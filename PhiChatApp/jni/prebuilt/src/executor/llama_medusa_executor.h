#pragma once

#include "executor/llama_executor.h"
#include "executor/llama_ringbuffer_executor.h"

#include <vector>

#ifdef DISABLE_RING_BUFFER
using LlamaExecutorBase = LlamaExecutor;
#else
using LlamaExecutorBase = LlamaRingBufferExecutor;
#endif

class LlamaMedusaExecutor : public LlamaExecutorBase {
public:
    // Inherit parent class constructor
    using LlamaExecutorBase::LlamaExecutorBase;

    // Override functions
    virtual void setPosEmbed(const size_t tokenIndex) override;

    virtual void resetTokenIndex() override;

    // Medusa
    void setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                           const std::vector<size_t>& positions);

    void resetMedusaTreeAttn();

    void rollbackTreeCache(const std::vector<size_t>& acceptedIndices);

private:
    std::vector<size_t> mMedusaTreePositions;
};