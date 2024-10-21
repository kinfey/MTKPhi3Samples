#pragma once

#include "executor/llama_executor.h"
#include "llm_helper/include/ringbuffer_cache_manager.h"
#include "llm_helper/include/rotary_embedding.h"

#include "llm_types.h"

#include <string>
#include <vector>

// Use ring buffer for caches.
class LlamaRingBufferExecutor : public LlamaExecutor {
private:
    static constexpr size_t kRingConcatDim = LlamaExecutor::kCacheLengthDim;

public:
    // Inherit parent class constructor
    using LlamaExecutor::LlamaExecutor;

    // For cache reset usage
    virtual void initCache() override;

private:
    virtual void linkCacheIOs() override {} // Don't link the cache IOs

    // Call Neuron API to read inputs with offsets
    void setOffsetedCacheInputs();

    // Initialize ring buffer related variables and constants
    virtual void preInitBufferProcess() override;
    virtual void postInitBufferProcess() override;

    virtual void runInferenceImpl() override;

    // Cache post-processing is not needed due to the padding-aware ring append
    virtual void leftPaddingCachePostprocess() override {} // Do nothing
    virtual void rightPaddingCachePostprocess() override {} // Do nothing

    virtual void rollbackCache(const size_t tokenCount) override;

protected:
    // Return offseted cache buffers
    virtual std::vector<char*> getCacheBuffers() override;

private:
    using LlamaExecutor::mCacheShapes;

    RingBufferCacheManager mRingBufferCacheManager;

    bool mDoneInitRingBuffer = false;
};