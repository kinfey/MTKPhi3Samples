#include "common/logging.h"
#include "llm_helper/include/utils.h"
#include "llm_helper/include/ringbuffer_cache_manager.h"

#include <vector>

#define NO_EXPORT __attribute__((visibility("hidden")))

class NO_EXPORT RingBufferCacheContext {
public:
    explicit RingBufferCacheContext(const size_t cacheLength,
                                    const std::vector<size_t>& modelInputCacheSizesBytes,
                                    const std::vector<size_t>& cachesNumRows,
                                    const size_t strideSizeBytes,
                                    const size_t overheadSizeBytes)
        : kCacheLength(cacheLength), kModelInputCacheSizesBytes(modelInputCacheSizesBytes),
          kCachesNumRows(cachesNumRows), kStrideSizeBytes(strideSizeBytes),
          kOverheadSizeBytes(overheadSizeBytes) {
        DCHECK_EQ(kModelInputCacheSizesBytes.size(), kCachesNumRows.size());
    }

    ~RingBufferCacheContext() {}

    void setIoCacheBuffers(const std::vector<void*>& inputCacheRingBuffers,
                           const std::vector<void*>& outputCacheBuffers) {
        DCHECK_EQ(inputCacheRingBuffers.size(), outputCacheBuffers.size());
        DCHECK_EQ(inputCacheRingBuffers.size(), kCachesNumRows.size());
        mInputCacheRingBuffers = inputCacheRingBuffers;
        mOutputCacheBuffers = outputCacheBuffers;
    }

    // Get cache append stride size in bytes
    size_t getStrideSize() const { return kStrideSizeBytes; }

    // Get input ringbuffer cache row size in bytes
    size_t getRowSize() const { return kCacheLength * getStrideSize(); }

    // Get output cache row size in bytes
    size_t getOutCacheRowSize(const size_t tokenBatchSize) const {
        return tokenBatchSize * getStrideSize();
    }

    // Get copy size in bytes
    size_t getCopySize(const size_t tokenBatchSize, const size_t padLength) const {
        DCHECK_GE(tokenBatchSize, padLength);
        return (tokenBatchSize - padLength) * getStrideSize();
    }

    size_t getOverheadSizeBytes() const { return kOverheadSizeBytes; }

    size_t getRingOffset() const { return mRingBufferOffsetBytes; }

    size_t getNumRows(const size_t index) const { return kCachesNumRows[index]; }

    size_t getNumCaches() const { return mInputCacheRingBuffers.size(); }

    void setRingOffset(const size_t offsetBytes) { mRingBufferOffsetBytes = offsetBytes; }

    void addRingOffset(const size_t sizeBytes) { mRingBufferOffsetBytes += sizeBytes; }

    size_t getModelInputCacheSizeBytes(const size_t index) const {
        return kModelInputCacheSizesBytes[index];
    }

    char* getInputCacheRingBuffer(const size_t index) {
        return reinterpret_cast<char*>(mInputCacheRingBuffers[index]);
    }

    char* getOutputCacheBuffer(const size_t index) {
        return reinterpret_cast<char*>(mOutputCacheBuffers[index]);
    }

private:
    // Constants
    const size_t kCacheLength;
    const std::vector<size_t> kModelInputCacheSizesBytes; // The input cache size that the model actually sees
    const std::vector<size_t> kCachesNumRows; // Applies for both input and output caches
    const size_t kStrideSizeBytes;
    const size_t kOverheadSizeBytes;

    // Variables
    size_t mRingBufferOffsetBytes = 0;

    std::vector<void*> mInputCacheRingBuffers;
    std::vector<void*> mOutputCacheBuffers;
};

RingBufferCacheManager::RingBufferCacheManager() {}

RingBufferCacheManager::~RingBufferCacheManager() { delete mCtx; }

void RingBufferCacheManager::initialize(const std::vector<ShapeType>& cacheShapes,
                                        const size_t cacheConcatDim,
                                        const size_t cacheTypeSizeBytes,
                                        const size_t initTokenIndex, const size_t maxTokenLength) {
    DCHECK_GT(cacheShapes.size(), 0);

    // Get cache length, assume same for all caches
    const auto& firstCacheShape = cacheShapes[0];
    DCHECK_LT(cacheConcatDim, firstCacheShape.size());
    const size_t cacheLength = firstCacheShape[cacheConcatDim];

    // Compute size of each cache input used by the model
    std::vector<size_t> inputCacheSizesBytes;
    for (const auto& cacheShape : cacheShapes) {
        const auto cacheSizeBytes = reduce_prod(cacheShape, cacheTypeSizeBytes);
        inputCacheSizesBytes.push_back(cacheSizeBytes);
    }

    // Compute stride size, assume same for all caches
    const size_t strideSizeBytes = reduce_prod(firstCacheShape.begin() + cacheConcatDim + 1,
                                               firstCacheShape.end(),
                                               cacheTypeSizeBytes);

    // Compute num rows for each cache
    std::vector<size_t> cachesNumRows;
    for (const auto& cacheShape : cacheShapes) {
        const auto numRows = reduce_prod(cacheShape.begin(), cacheShape.begin() + cacheConcatDim);
        cachesNumRows.push_back(numRows);
    }

    // Compute overhead size, assume same for all caches
    const size_t firstRowUsage = std::max(1UL, initTokenIndex);
    const size_t overheadSizeBytes = (maxTokenLength - firstRowUsage) * strideSizeBytes;

    DCHECK_EQ(mCtx, nullptr);
    mCtx = new RingBufferCacheContext(cacheLength, inputCacheSizesBytes, cachesNumRows,
                                      strideSizeBytes, overheadSizeBytes);
    mIsInitialized = true;
}

void RingBufferCacheManager::setIoCacheBuffers(const std::vector<void*>& inputCacheRingBuffers,
                                               const std::vector<void*>& outputCacheBuffers) {
    DCHECK_EQ(inputCacheRingBuffers.size(), outputCacheBuffers.size());
    mCtx->setIoCacheBuffers(inputCacheRingBuffers, outputCacheBuffers);
}

size_t RingBufferCacheManager::getOverheadSizeBytes() const {
    return mCtx->getOverheadSizeBytes();
}

// Ring offset public interface
size_t RingBufferCacheManager::getRingOffset() const {
    ensureInit();
    const auto ringBufferOffsetBytes = mCtx->getRingOffset();
    DCHECK_LE(ringBufferOffsetBytes, mCtx->getOverheadSizeBytes());
    return ringBufferOffsetBytes;
}

void RingBufferCacheManager::resetRingOffset() {
    ensureInit();
    mCtx->setRingOffset(0);
}

void RingBufferCacheManager::advanceRingOffset(const size_t tokenCount) {
    ensureInit();
    mCtx->addRingOffset(tokenCount * mCtx->getStrideSize());
    CHECK_LE(mCtx->getRingOffset(), mCtx->getOverheadSizeBytes())
        << "Ring buffer offset overflow.";
}

// Ring buffer append
void RingBufferCacheManager::appendInOutCaches(const size_t tokenBatchSize,
                                               const size_t leftPadLength,
                                               const size_t rightPadLength,
                                               const bool isCacheEmpty) {
    // View cache buffer of shape [..., ringConcatDim, ...] as:
    //   [mNumRows, (ringConcatDim, strideSize)]
    //    <------>  <------------------------->
    //       row                col
    // Write strideSize number of values, then jump by rowSize.
    //
    // If init from zero, it will append to the last col of the first row and ring buffer offset
    // will remain at 0. Otherwise, it will start appending from the second row onwards and
    // ring buffer offset will begin to take effect.

    ensureInit();

    const size_t padLength = leftPadLength + rightPadLength;
    const auto copySizeBytes = mCtx->getCopySize(tokenBatchSize, padLength); // Padding is exclulded from copy
    const bool hasEnoughSpaceToAppend = [&]() {
        const auto extraNeededSizeBytes = isCacheEmpty ? 0 : mCtx->getRingOffset() + copySizeBytes;
        return extraNeededSizeBytes <= mCtx->getOverheadSizeBytes();
    }();
    if (!hasEnoughSpaceToAppend) {
        resetRingBuffer(); // Will change the output of getRingOffset()
    }

    // Append ring buffer
    const auto icRowSizeBytes = mCtx->getRowSize(); // Input cache row size
    const auto ocRowSizeBytes = mCtx->getOutCacheRowSize(tokenBatchSize); // Output cache row size
    const auto padOffset = leftPadLength * mCtx->getStrideSize();
    const size_t startOffset = isCacheEmpty ? icRowSizeBytes - copySizeBytes // Fill the end of row (aka last N cols)
                                            : icRowSizeBytes + mCtx->getRingOffset();

    auto appendSingleRBCache = [&](const size_t index) {
        auto inputCacheBuffer = mCtx->getInputCacheRingBuffer(index) + startOffset;
        const auto outputCacheBuffer = mCtx->getOutputCacheBuffer(index);
        for (size_t rowIdx = 0; rowIdx < mCtx->getNumRows(index); rowIdx++) {
            std::memcpy(inputCacheBuffer  + rowIdx * icRowSizeBytes,
                        outputCacheBuffer + rowIdx * ocRowSizeBytes + padOffset, copySizeBytes);
        }
    };
    for (size_t i = 0; i < mCtx->getNumCaches(); i++) {
        appendSingleRBCache(i);
    }
}

void RingBufferCacheManager::resetRingBuffer() {
    const auto ringOffsetBytes = mCtx->getRingOffset();
    if (ringOffsetBytes == 0) {
        return; // No need to reset
    }
    for (size_t i = 0; i < mCtx->getNumCaches(); i++) {
        auto cacheRingBuffer = mCtx->getInputCacheRingBuffer(i);
        const auto inputCacheSizeBytes = mCtx->getModelInputCacheSizeBytes(i);
        std::memcpy(cacheRingBuffer, cacheRingBuffer + ringOffsetBytes, inputCacheSizeBytes);
    }
    resetRingOffset(); // Reset ring buffer offsets to 0
}

// Returns true if rollback ring buffer is successful, false if otherwise.
bool RingBufferCacheManager::rollback(const size_t tokenCount) {
    ensureInit();
    const size_t rollbackSizeBytes = tokenCount * mCtx->getStrideSize();
    const auto ringOffsetBytes = mCtx->getRingOffset();

    // Rollback size is greater than the current ring offset
    if (ringOffsetBytes < rollbackSizeBytes) {
        return false;
    }

    const auto rowSizeBytes = mCtx->getRowSize();
    const size_t startClearOffset = ringOffsetBytes - rollbackSizeBytes;
    for (size_t i = 0; i < mCtx->getNumCaches(); i++) {
        auto inputCacheBuffer = mCtx->getInputCacheRingBuffer(i);
        const auto numRows = mCtx->getNumRows(i);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto curInputCacheBuffer = inputCacheBuffer + rowIdx * rowSizeBytes;
            std::memset(curInputCacheBuffer + startClearOffset, 0, rollbackSizeBytes);
        }
    }
    mCtx->addRingOffset(-rollbackSizeBytes);
    return true;
}

void RingBufferCacheManager::ensureInit() const {
    CHECK(mIsInitialized)
        << "Attempting to use RingBufferCacheManager without initialization.";
    CHECK_GT(mCtx->getNumCaches(), 0)
        << "Attempting to use RingBufferCacheManager without any cache buffers.";
}