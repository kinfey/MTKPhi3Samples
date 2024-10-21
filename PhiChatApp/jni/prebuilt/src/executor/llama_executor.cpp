#include "executor/llama_executor.h"
#include "llm_helper/include/utils.h"
#include "llm_helper/include/rotary_embedding.h"

#include "common/logging.h"

#include <algorithm>
#include <numeric>
#include <iterator>
#include <thread>

#include <arm_neon.h>
#include <stdint.h>

void LlamaExecutor::initialize() {
    buildRuntimeIdxMap();
    setDefaultModel();
    linkCacheIOs();
    setNumIOs();

    if (isSharedWeightsUsed()) {
        // Load shared weights and model in parallel
        const auto sharedWeightsInputIndex = getExpectedNumInputs() - 1;
        // Reserve to prevent `initBuffer()` from allocating in case it is called first
        reserveInputBuffer(sharedWeightsInputIndex);
        std::thread loadSharedWeightsThread([&] {
            ExecutorBackend::loadSharedWeights(sharedWeightsInputIndex);
        });
        std::thread initializeThread([&] {
            ExecutorBackend::initialize();
        });
        loadSharedWeightsThread.join();
        initializeThread.join();
    } else {
        ExecutorBackend::initialize(); // Not using shared weights
    }

    initMaskBuilder();
    initCache();
    applyLoraWeights(kDefaultLoraKey);
}

void LlamaExecutor::setNumIOs() {
    // Used because Neuron Adapter requires number of IO before runtime init
    this->setNumInputs(getExpectedNumInputs());
    this->setNumOutputs(getExpectedNumOutputs());
}

void LlamaExecutor::buildRuntimeIdxMap() {
    size_t runtimeIdx = 0;
    for (const auto& runtimeInfo : kRuntimeInfos) {
        const auto& batchSize = runtimeInfo.batchSize;
        const auto& cacheSize = runtimeInfo.cacheSize;
        mRuntimeIdxMap[batchSize][cacheSize] = runtimeIdx++;
    }
}

void LlamaExecutor::setDefaultModel() {
    // Select the model with largest batch size with smallest cache size

    auto keyLessThan = [](const auto& pairA, const auto& pairB) {
        return pairA.first < pairB.first;
    };
    auto getMaxKey = [&](const auto& map) {
        return std::max_element(map.begin(), map.end(), keyLessThan)->first;
    };
    auto getMinKey = [&](const auto& map) {
        return std::min_element(map.begin(), map.end(), keyLessThan)->first;
    };

    const auto maxBatchSize = getMaxKey(mRuntimeIdxMap);
    const auto minCacheSize = getMinKey(mRuntimeIdxMap[maxBatchSize]);
    const auto defaultRuntimeIndex = mRuntimeIdxMap[maxBatchSize][minCacheSize];
    this->setDefaultRuntimeIndex(defaultRuntimeIndex);

    mModelNumInputToken = maxBatchSize;
    mMaskLength = kCacheLength + mModelNumInputToken;
}

void LlamaExecutor::applyLoraWeights(const LoraKey& loraKey) {
    if (mCurrentLoraKey == loraKey) {
        return; // Already applied
    } else if (loraKey.empty()) {
        removeLoraWeights(); // Empty key, so clear Lora weights to zeros to use base weights
        return;
    } else if (kLoraWeightsPathMap.find(loraKey) == kLoraWeightsPathMap.end()) {
        LOG(ERROR) << "Invalid LoraKey: " << loraKey;
        return;
    }
    std::vector<void*> loraInputBuffers;
    std::vector<size_t> loraInputBufferSizes;
    for (const auto idx : kLoraWeightsInputIndexes) {
        const auto& input = this->getInput(idx);
        loraInputBuffers.push_back(input.buffer);
        loraInputBufferSizes.push_back(input.usedSizeBytes);
    }
    CHECK_EQ(kLoraInputCount, loraInputBuffers.size());
    LoraWeightsLoader loader(kLoraWeightsPathMap.at(loraKey));
    loader.loadLoraWeights(loraInputBuffers, loraInputBufferSizes);
    mCurrentLoraKey = loraKey;
    LOG(DEBUG) << "Successfully applied Lora weights with key: " << loraKey;
}

void LlamaExecutor::applyLoraWeights(const std::vector<char*>& loraWeights,
                                                     const std::vector<size_t>& sizes) {
    CHECK_EQ(kLoraInputCount, loraWeights.size());
    CHECK_EQ(sizes.size(), loraWeights.size());
    for (size_t i = 0; i < kLoraInputCount; i++) {
        const auto loraInputIdx = kLoraWeightsInputIndexes[i];
        auto& input = this->getInput(loraInputIdx);
        const auto loraWeight = loraWeights[i];
        const auto loraWeightSize = sizes[i];
        CHECK_LE(loraWeightSize, input.sizeBytes)
            << "Insufficient buffer allocation (size=" << input.sizeBytes << ") to load Lora input "
            << i << " weights (size=" << loraWeightSize << ")";
        if (loraWeightSize != input.usedSizeBytes) {
            LOG(WARN) << "Expected Lora input " << i << " size by model (" << input.usedSizeBytes
                      << ") != " << "provided Lora weights size (" << loraWeightSize << ")";
        }
        std::memcpy(input.buffer, loraWeight, loraWeightSize);
    }
    mCurrentLoraKey = ""; // Not using any predefined Lora keys
    LOG(DEBUG) << "Successfully applied Lora weights from user provided buffers";
}

void LlamaExecutor::removeLoraWeights() {
    // Memset Lora input buffers to zeros
    for (const auto idx : kLoraWeightsInputIndexes) {
        auto& input = this->getInput(idx);
        std::memset(input.buffer, 0, input.usedSizeBytes);
    }
    mCurrentLoraKey = "";
    LOG(DEBUG) << "Removed Lora weights";
}

void LlamaExecutor::preInitBufferProcess() {
    // Input cache shape
    const auto& cacheInputIdxs = this->getCacheInputIdxs();
    const auto numInputCaches = cacheInputIdxs.size();
    DCHECK_GT(numInputCaches, 0);
    DCHECK_EQ(numInputCaches, kCacheCount);
    mCacheShapes.resize(numInputCaches);
    for (size_t i = 0; i < numInputCaches; i++) {
        auto& cacheShape = mCacheShapes[i];
        this->getRuntimeInputShape(cacheInputIdxs[i], cacheShape.data());
        CHECK_EQ(cacheShape[kCacheLengthDim], kCacheLength)
            << "Please ensure the cache size option is set correctly.";
    }

    // Ensure all stride sizes are the same across cache inputs
    auto getStrideSize = [this](const auto& cacheShape) {
        return reduce_prod(cacheShape.begin() + kCacheLengthDim + 1, cacheShape.end(),
                           kCacheTypeSize);
    };
    const auto firstStrideSize = getStrideSize(mCacheShapes[0]);
    for (const auto& cacheShape : mCacheShapes) {
        CHECK_EQ(firstStrideSize, getStrideSize(cacheShape))
            << "Different stride size across caches are not supported.";
    }

    // Verify cache type size using the first cache
    const auto inputCacheSizeBytes = this->getModelInputSizeBytes(cacheInputIdxs[0]);
    const auto inputCacheSize = reduce_prod(mCacheShapes[0]);
    const auto modelCacheTypeSize = inputCacheSizeBytes / inputCacheSize;
    CHECK_EQ(kCacheTypeSize, modelCacheTypeSize)
        << "Mismatch between user provided cache type size (" << kCacheTypeSize << ") "
        << "and actual model cache type size (" << modelCacheTypeSize << ")";


    // Check number of IOs
    CHECK_EQ(getExpectedNumInputs(), this->getRuntimeNumInputs())
        << "Number of inputs does not match, please ensure the model is correct.";
    CHECK_EQ(getExpectedNumOutputs(), this->getRuntimeNumOutputs())
        << "Number of outputs does not match, please ensure the model is correct.";
}

void LlamaExecutor::initMaskBuilder() {
    const auto maskBuffer = this->getInputBuffer(getMaskInputIdx());
    const auto maskSizeBytes = this->getModelInputSizeBytes(getMaskInputIdx());
    mMaskBuilder = std::make_unique<MaskBuilder>(maskBuffer, maskSizeBytes, kMaskType,
                                                 kCacheLength);
    mMaskBuilder->buildMask(mModelNumInputToken, mCurrentTokenIndex);
}

bool LlamaExecutor::hotSwapModel(const size_t batchSize, const size_t cacheSize) {
    LOG_ENTER

    if (cacheSize != kUnusedSize && cacheSize != kCacheLength) {
        LOG(FATAL) << "Unimplemented: Variable cache size model swapping is not yet supported.";
    }

    // Save old values
    const auto oldRuntimeIdx = this->getRuntimeIndex();
    const size_t oldNumInputToken = mModelNumInputToken;

    auto mapHasKey = [](const auto& map, const auto& key) {
        return map.find(key) != map.end();
    };

    if (!mapHasKey(mRuntimeIdxMap, batchSize)) {
        LOG(ERROR) << "Model swap: No model with batchSize=" << batchSize << " is available";
        return false;
    }
    // Search for suitable runtime matching the requirements (batch size & cache size)
    const auto& runtimesWithBatchSize = mRuntimeIdxMap[batchSize];
    if (cacheSize != kUnusedSize && !mapHasKey(runtimesWithBatchSize, cacheSize)) {
        LOG(ERROR) << "Model swap: No model with batchSize=" << batchSize << " has cacheSize="
                   << cacheSize;
        return false;
    }
    const auto runtimeIdx = (cacheSize != kUnusedSize) ? runtimesWithBatchSize.at(cacheSize)
                                                       : runtimesWithBatchSize.begin()->second;
    if (cacheSize == kUnusedSize && runtimesWithBatchSize.size() > 1) {
        LOG(WARN) << "Model swap: No target cache size provided, selecting the first available "
                  << "model with cache size: " << runtimeIdx;
    }
    if (runtimeIdx == oldRuntimeIdx) {
        LOG(DEBUG) << "Model swapping to itself.";
        return true;
    }

    this->selectRuntime(runtimeIdx);

    const auto newRuntimeIdx = this->getRuntimeIndex();
    if (oldRuntimeIdx == newRuntimeIdx) {
        LOG(WARN) << "Failed to switch to model with batchSize=" << batchSize
                  << " and cacheSize=" << cacheSize << ". Model currently remain at "
                  << "(batchSize=" << oldNumInputToken << ", cacheSize=" << kCacheLength << "): "
                  << this->getModelPath();
        return false;
    }

    // Update model variables
    // Mask length = cache size (length) + num input token
    mModelNumInputToken = batchSize;
    mMaskLength = kCacheLength + mModelNumInputToken;

    this->updateModelIO();
    this->registerRuntimeIO(); // Attach IO buffers to model runtime

    // Rebuild mask because different batch/cache size values will produce different mask shapes
    mMaskBuilder->markMaskDirty();

    // Update mask size
    const auto newMaskSizeBytes = this->getModelInputSizeBytes(getMaskInputIdx());
    mMaskBuilder->updateMaskSize(newMaskSizeBytes);

    const auto numInputs = this->getNumInputs();
    const auto numOutputs = this->getNumOutputs();

    // Check that the buffer requirement is <= the existing ones
    for (size_t i = 0; i < numInputs; i++) {
        if (this->getInputBufferSizeBytes(i) < this->getModelInputSizeBytes(i)) {
            LOG(ERROR) << "Model hotswap failed. Insufficient originally allocated input buffer "
                          "size.";
            return false;
        }
    }
    for (size_t i = 0; i < numOutputs; i++) {
        if (this->getOutputBufferSizeBytes(i) < this->getModelOutputSizeBytes(i)) {
            LOG(ERROR) << "Model hotswap failed. Insufficient originally allocated output buffer "
                          "size.";
            return false;
        }
    }

    return true;
}

void LlamaExecutor::linkCacheIOs() {
    const size_t numCacheIn = kCacheInputIndexes.size();
    for (size_t i = 0; i < numCacheIn; i++) {
        this->linkModelIO(kCacheInputIndexes[i], kCacheOutputIndexes[i]);
    }
}

void LlamaExecutor::resetTokenIndex() {
    setTokenIndex(kInitTokenIndex);
    mMaskBuilder->markMaskDirty();
}

void LlamaExecutor::setTokenIndex(const size_t index) {
    if (index >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set token index (" << index << ") exceeding the supported max "
                      "token length (" << kMaxTokenLength << ")";
        return;
    }
    mCurrentTokenIndex = index;
}

void LlamaExecutor::advanceTokenIndex() {
    setTokenIndex(mCurrentTokenIndex + mModelNumInputToken);
}

size_t LlamaExecutor::getTokenIndex() const {
    return mCurrentTokenIndex;
}

int LlamaExecutor::alignInputTokens(const size_t numInputToken) {
    int rollbackCount = mModelNumInputToken - numInputToken;
    if (rollbackCount > 0) {
        CHECK_GE(mCurrentTokenIndex, rollbackCount) << "Total tok count < model input tok count";
        rollbackCache(rollbackCount);
        LOG(DEBUG) << "Tokens/Caches alignment rollback count = " << rollbackCount;

        // rollbackCache() requires original mCurrentTokenIndex value so only modify after the call
        mCurrentTokenIndex -= rollbackCount;

        // Rebuild mask as updateMask requires mCurrentTokenIndex to be monotonically increasing
        mMaskBuilder->markMaskDirty();
    }
    return rollbackCount;
}

// Also updates the token index
void LlamaExecutor::updatePosEmbAndMask(const size_t numInputToken) {
    if (mCurrentTokenIndex + numInputToken > kMaxTokenLength) {
        LOG(FATAL) << "Attempting to generate tokens exceeding the supported max token length ("
                   << kMaxTokenLength << ")";
    }
    if (mCurrentTokenIndex > 0 && getLeftPadding() > 0) {
        LOG(FATAL) << "Left-padding is only allowed in the first prompt pass.";
    }
    mMaskBuilder->updateMask(mModelNumInputToken, mCurrentTokenIndex, numInputToken);
    setPosEmbed(mCurrentTokenIndex);
    LOG_DONE
}

void LlamaExecutor::setPosEmbed(const size_t tokenIndex) {
    // Cut the array from master
    if (tokenIndex >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set rotaty embedding using index exceeding the supported "
                      "max token length (" << kMaxTokenLength << ")";
    }
    START_TIMER
    const auto& rotEmbInputIdxes = getRotEmbInputIdxs();
    DCHECK_EQ(rotEmbInputIdxes.size(), kRotEmbInputCount);

    auto getRotEmbInputs = [&]() {
        std::vector<void*> rotEmbInputs(kRotEmbInputCount);
        for (size_t i = 0; i < kRotEmbInputCount; i++)
            rotEmbInputs[i] = this->getInputBuffer(rotEmbInputIdxes[i]);
        return rotEmbInputs;
    };
    mRotEmbMasterLut->setEmbed(getRotEmbInputs(), tokenIndex, mModelNumInputToken, getLeftPadding(),
                               getRightPadding());
    LOG_LATENCY
    LOG_DONE
}

size_t LlamaExecutor::getLeftPadding() const {
    return (mPaddingMode == PaddingMode::LEFT) ? mCurrentPadSize : 0;
}

size_t LlamaExecutor::getRightPadding() const {
    return (mPaddingMode == PaddingMode::RIGHT) ? mCurrentPadSize : 0;
}

void LlamaExecutor::setLeftPadding(const size_t leftPadSize) {
    mCurrentPadSize = leftPadSize;
    mPaddingMode = PaddingMode::LEFT;

    // Notify mask builder about padding
    mMaskBuilder->notifyLeftPadding(leftPadSize);
}

void LlamaExecutor::setRightPadding(const size_t rightPadSize) {
    mCurrentPadSize = rightPadSize;
    mPaddingMode = PaddingMode::RIGHT;

    // Notify mask builder about padding
    mMaskBuilder->notifyRightPadding(rightPadSize);
}

void LlamaExecutor::paddingPostprocess() {
    if (mCurrentPadSize == 0) {
        return;
    }

    if (mPaddingMode == PaddingMode::RIGHT) {
        rightPaddingCachePostprocess();
    } else if (mPaddingMode == PaddingMode::LEFT) {
        leftPaddingCachePostprocess();
    }
    setTokenIndex(mCurrentTokenIndex - mCurrentPadSize); // Rollback by padding size
    mCurrentPadSize = 0; // Reset padding size
}

void LlamaExecutor::leftPaddingCachePostprocess() {
    // NOTE: This part might not actually be needed

    // Stride size is same across caches
    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = kCacheLength * strideSizeBytes;

    // Fill padded sections with zeros
    size_t cacheCounter = 0;
    for (const auto cacheInputIdx : getCacheInputIdxs()) {
        auto cacheBuffer = reinterpret_cast<char*>(this->getInputBuffer(cacheInputIdx));
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            const size_t offset = (kCacheLength - mModelNumInputToken) * strideSizeBytes;
            const size_t zeroCount = mCurrentPadSize * strideSizeBytes;
            std::memset(cacheBufRow + offset, 0, zeroCount);
        }
    }
}

void LlamaExecutor::rightPaddingCachePostprocess() {
    // advanceTokenIndex() has to be called first to set mCurrentTokenIndex for rollbackCache()
    rollbackCache(mCurrentPadSize);
}

size_t LlamaExecutor::getCacheNumRows(const size_t index) const {
    CHECK_GT(mCacheShapes.size(), 0) << "Cache shapes have not been initialized.";
    CHECK_LT(index, mCacheShapes.size());
    const auto& cacheShape = mCacheShapes[index];
    // NOTE: cacheShape[0] is the batch dim
    return reduce_prod(cacheShape.begin(), cacheShape.begin() + kCacheLengthDim);
}

size_t LlamaExecutor::getCacheStrideSize() const {
    CHECK_GT(mCacheShapes.size(), 0) << "Cache shapes have not been initialized.";
    const auto& cacheShape = mCacheShapes[0];
    return reduce_prod(cacheShape.begin() + kCacheLengthDim + 1, cacheShape.end(), kCacheTypeSize);
}

void LlamaExecutor::initCache() {
    START_TIMER
    resetTokenIndex();
    if (kInitCachePath.size() == 0) {
        // Use default zero initialization if no cache path provided
        for (const auto cacheIdx : getCacheInputIdxs()) {
            auto inputCache = this->getInput(cacheIdx);
            char* cacheBuffer = reinterpret_cast<char*>(inputCache.buffer);
            const size_t cacheSizeBytes = inputCache.sizeBytes;
            std::memset(cacheBuffer, 0, cacheSizeBytes);
        }
        LOG(DEBUG) << "initCache: zero initialization";
        LOG_LATENCY
        LOG_DONE
        return;
    }

    LOG(DEBUG) << "initCache: precomputed cache initialization";

    auto fin = std::ifstream(kInitCachePath, std::ios::binary);
    if (!fin) {
        LOG(FATAL) << "Init cache file not found: " << kInitCachePath;
    }
    const auto& cacheInputIdxs = getCacheInputIdxs();
    DCHECK_EQ(cacheInputIdxs.size(), kCacheCount);
    for (size_t i = 0; i < kCacheCount; i++) {
        const auto cacheInputIdx = cacheInputIdxs[i];
        const auto cacheSizeBytes = this->getModelInputSizeBytes(cacheInputIdx);
        auto cacheBuffer = this->getInputBuffer(cacheInputIdx);
        fin.read(reinterpret_cast<char*>(cacheBuffer), cacheSizeBytes);
        if (fin.gcount() != cacheSizeBytes) {
            LOG(WARN) << "Expected cache[" << i << "] size=" << cacheSizeBytes << ", but "
                      << "actual size read from file is " << fin.gcount();
        }
    }

    LOG_LATENCY
    LOG_DONE
}

std::vector<char*> LlamaExecutor::getCacheBuffers() {
    const auto& cacheInputIdxs = getCacheInputIdxs();
    const size_t numCacheInputs = cacheInputIdxs.size();
    std::vector<char*> cacheBuffers(numCacheInputs);
    for (size_t i = 0; i < numCacheInputs; i++) {
        cacheBuffers[i] = reinterpret_cast<char*>(this->getInputBuffer(cacheInputIdxs[i]));
    }
    return cacheBuffers;
}

void LlamaExecutor::getCacheBuffersWithSize(std::vector<char*>& cacheBuffers,
                                                            size_t& byteSizePerCache) {
    cacheBuffers = getCacheBuffers();
    byteSizePerCache = this->getModelInputSizeBytes(getCacheInputIdxs()[0]);
}

void LlamaExecutor::rollbackCache(const size_t tokenCount) {
    if (tokenCount == 0) {
        return; // do nothing
    }
    START_TIMER

    // View cache buffer of shape [..., kCacheLength, ...] as:
    //   [numRows, (kCacheLength, strideSizeBytes)]
    //    <----->  <----------------------------->
    //      row                   col

    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = kCacheLength * strideSizeBytes;
    const size_t firstNonEmptyIdx = kCacheLength - std::min(mCurrentTokenIndex, kCacheLength);

    auto cacheBuffers = getCacheBuffers();

    // Shift right and truncate tokenCount, then fill left with zeros
    size_t cacheCounter = 0;
    for (auto cacheBuffer : cacheBuffers) {
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            // Copy from back until srcOffset reaches empty segment in the cache
            for (size_t tokenIdx = kCacheLength - 1; tokenIdx >= firstNonEmptyIdx + tokenCount;
                 tokenIdx--) {
                const size_t dstOffset = tokenIdx * strideSizeBytes;
                const size_t srcOffset = (tokenIdx - tokenCount) * strideSizeBytes;
                std::memcpy(cacheBufRow + dstOffset, cacheBufRow + srcOffset, strideSizeBytes);
            }
            const size_t offset = firstNonEmptyIdx * strideSizeBytes;
            const size_t zeroCount = tokenCount * strideSizeBytes;
            std::memset(cacheBufRow + offset, 0, zeroCount);
        }
    }
    LOG_LATENCY
    LOG_DONE
}