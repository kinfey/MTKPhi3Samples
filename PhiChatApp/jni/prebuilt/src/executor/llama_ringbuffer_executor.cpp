#include "executor/llama_ringbuffer_executor.h"
#include "runtime/neuron_runtime.h"

#include "llm_helper/include/ringbuffer_cache_manager.h"

#include "common/logging.h"

#include <algorithm>
#include <cmath>

#include <string>
#include <vector>

void LlamaRingBufferExecutor::initCache() {
    LlamaExecutor::initCache();
    mRingBufferCacheManager.resetRingOffset();
}

void LlamaRingBufferExecutor::setOffsetedCacheInputs() {
    START_TIMER
    const auto ringOffsetBytes = mRingBufferCacheManager.getRingOffset();
    for (const auto i : this->getCacheInputIdxs()) {
        this->setRuntimeOffsetedInput(i, ringOffsetBytes);
    }
    LOG_LATENCY
    LOG_DONE
}

void LlamaRingBufferExecutor::preInitBufferProcess() {
    if (mDoneInitRingBuffer) {
        // Do nothing, especially after model swap.
        return;
    }

    LlamaExecutor::preInitBufferProcess();

    // Prepare cache shapes
    std::vector<RingBufferCacheManager::ShapeType> cacheShapesForRB;
    for (const auto& cacheShape : mCacheShapes) {
        cacheShapesForRB.emplace_back(cacheShape.begin(), cacheShape.end());
    }

    mRingBufferCacheManager.initialize(cacheShapesForRB, kRingConcatDim, this->kCacheTypeSize,
                                       this->kInitTokenIndex, this->kMaxTokenLength);

    // Expand the required size for each inputs
    const auto ringBufferOverheadSizeBytes = mRingBufferCacheManager.getOverheadSizeBytes();
    DCHECK_GT(ringBufferOverheadSizeBytes, 0);
    for (const auto cacheIdx : this->getCacheInputIdxs()) {
        this->getInput(cacheIdx).sizeBytes += ringBufferOverheadSizeBytes;
    }

    mDoneInitRingBuffer = true;
}

void LlamaRingBufferExecutor::postInitBufferProcess() {
    // Prepare input/output cache buffers
    const auto inputCacheIdxes = this->getCacheInputIdxs();
    const auto outputCacheIdxes = this->getCacheOutputIdxs();
    DCHECK_EQ(inputCacheIdxes.size(), inputCacheIdxes.size());
    const auto numCaches = inputCacheIdxes.size();
    std::vector<void*> inputCaches(numCaches), outputCaches(numCaches);
    for (size_t i = 0; i < numCaches; i++) {
        inputCaches[i] = this->getInputBuffer(inputCacheIdxes[i]);
        outputCaches[i] = this->getOutputBuffer(outputCacheIdxes[i]);
    }
    mRingBufferCacheManager.setIoCacheBuffers(inputCaches, outputCaches);
}

void LlamaRingBufferExecutor::runInferenceImpl() {
    // Rough flow:
    //  1. Change cache buffer reading offset using setOffsetedInput.
    //  2. Append each cache output to the ring buffer.
    //  3. But if the current pass overshoots the ring buffer, reset back to 0 and memcpy.

    // Set the offseted inputs then run inference as usual
    setOffsetedCacheInputs();
    LlamaExecutor::runInferenceImpl();

    const bool isCacheEmpty = (this->mCurrentTokenIndex == 0);

    mRingBufferCacheManager.appendInOutCaches(this->getModelNumInputToken(), this->getLeftPadding(),
                                              this->getRightPadding(), isCacheEmpty);

    // Advance ring buffer offsets by copy size. Also, skip offset for the first pass.
    if (!isCacheEmpty) {
        mRingBufferCacheManager.advanceRingOffset(this->getValidModelNumInputToken());
    }
}

void LlamaRingBufferExecutor::rollbackCache(const size_t tokenCount) {
    if (!mRingBufferCacheManager.rollback(tokenCount)) {
        // Fallback to use the naive rollbackCache approach.
        LlamaExecutor::rollbackCache(tokenCount);
    }
}

std::vector<char*> LlamaRingBufferExecutor::getCacheBuffers() {
    const auto& inputCacheIdxs = this->getCacheInputIdxs();
    const size_t numInputCaches = inputCacheIdxs.size();
    std::vector<char*> cacheBuffers(numInputCaches);
    const auto ringOffsetBytes = mRingBufferCacheManager.getRingOffset();
    for (size_t i = 0; i < numInputCaches; i++) {
        const auto inputCacheIdx = inputCacheIdxs[i];
        const auto cacheRingBuffer = reinterpret_cast<char*>(this->getInputBuffer(inputCacheIdx));
        cacheBuffers[i] = cacheRingBuffer + ringOffsetBytes;
    }
    return cacheBuffers;
}