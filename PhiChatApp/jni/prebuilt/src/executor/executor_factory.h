#pragma once

#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"
#include "executor/tflite_executor.h"
#include "executor/llama_executor.h"
#include "executor/llama_medusa_executor.h"
#include "executor/llama_ringbuffer_executor.h"

#include "common/logging.h"

#include <type_traits>

#ifdef USE_USDK_BACKEND
using NeuronModelExecutor = NeuronUsdkExecutor;
#else
using NeuronModelExecutor = NeuronExecutor;
#endif

#ifdef DISABLE_RING_BUFFER
using LlamaModelExecutor = LlamaExecutor;
#else
using LlamaModelExecutor = LlamaRingBufferExecutor;
#endif

using LlamaMedusaModelExecutor = LlamaMedusaExecutor;

using TFLiteModelExecutor = TfliteExecutor;

#define GetExecutorClass(ExecType) ExecType##ModelExecutor


enum class ExecutorType { Neuron, TFLite, Llama, LlamaMedusa };

class ExecutorFactory {
public:
    explicit ExecutorFactory(const ExecutorType executorType = ExecutorType::Llama)
        : mExecutorType(executorType) {}

    ExecutorFactory& setType(const ExecutorType executorType) {
        mExecutorType = executorType;
        return *this;
    }

    template <typename... Args>
    Executor* create(Args&&... args) const {
#define __DECL__(ExecType)                                                              \
    case ExecutorType::ExecType: {                                                      \
        auto executor = create<ExecType##ModelExecutor>(std::forward<Args>(args)...);   \
        DCHECK(executor != nullptr)                                                     \
            << "Unable to create '" #ExecType "' executor with the given "              \
            << sizeof...(Args) << " arguments.";                                        \
        return executor;                                                                \
    }

        switch (mExecutorType) {
            __DECL__(Neuron)
            __DECL__(TFLite)
            __DECL__(Llama)
            __DECL__(LlamaMedusa)
        }

#undef __DECL__
    }

    // Can be constructed with the provided arguments
    template <typename ExecutorClass, typename... Args>
    static std::enable_if_t<std::is_constructible_v<ExecutorClass, Args...>, Executor*>
    create(Args&&... args) {
        return new ExecutorClass(std::forward<Args>(args)...);
    }

    // Cannot be constructed with the provided arguments
    template <typename ExecutorClass, typename... Args>
    static std::enable_if_t<!std::is_constructible_v<ExecutorClass, Args...>, Executor*>
    create(Args&&... args) {
        return nullptr;
    }

private:
    ExecutorType mExecutorType;
};