#include "executor/neuron_executor.h"
#include "common/logging.h"
#include "common/file_mem_mapper.h"

void NeuronExecutor::loadSharedWeights(const size_t inputIndex) {
    if (!isSharedWeightsUsed()) {
        return; // FMS shared weights not used
    }
    START_TIMER
    FileMemMapper sharedWeightsFile(kSharedWeightsPath);
    const auto [swData, swSize] = sharedWeightsFile.get();

    auto& swBuffer = getInput(inputIndex); // Shared weights IO buffer
    DCHECK(!swBuffer.isAllocated());
    swBuffer.sizeBytes = swSize;
    swBuffer.usedSizeBytes = swSize;
    if (!allocateMemory(swBuffer)) {
        LOG(ERROR) << "Failed to allocate memory for shared weights on input[" << inputIndex << "] "
                   << "with size=" << swSize;
    }
    mAllocatedBuffers.push_back(swBuffer);
    std::memcpy(swBuffer.buffer, swData, swSize);

    LOG_LATENCY
    LOG_DONE
}

bool NeuronExecutor::isSharedWeightsUsed() const {
    return kSharedWeightsPath.size() > 0;
}

void NeuronExecutor::runInferenceImpl() {
    START_TIMER
    CHECK_NEURON_ERROR(fnNeuronRuntime_inference(this->getRuntime()));
    LOG_LATENCY
    LOG_DONE
}

void* NeuronExecutor::createRuntime(const std::string& modelPath) {
    START_TIMER
    if (!checkFile(modelPath)) {
        LOG(FATAL) << "File not found: " << modelPath;
    }

    // Create runtime
    void* runtime;
    CHECK_NEURON_ERROR(fnNeuronRuntime_create_with_options(
        "--apusys-config \"{ \\\"high_addr\\\": true, \\\"import_forever\\\": true }\"",
        &mEnvOptions, &runtime
    ));

    // Load model
    CHECK_NEURON_ERROR(fnNeuronRuntime_loadNetworkFromFile(runtime, modelPath.c_str()));

    // Set QoS option
    mQosOptions.preference = NEURONRUNTIME_PREFER_PERFORMANCE;
    mQosOptions.boostValue = 100;
    mQosOptions.priority = NEURONRUNTIME_PRIORITY_HIGH;
    mQosOptions.powerPolicy = NEURONRUNTIME_POWER_POLICY_SUSTAINABLE;
    CHECK_NEURON_ERROR(fnNeuronRuntime_setQoSOption(runtime, &mQosOptions));

    LOG_LATENCY
    LOG_DONE
    return runtime;
}

void NeuronExecutor::releaseRuntime(void* runtime) {
    START_TIMER
    // Release current runtime
    fnNeuronRuntime_release(runtime);
    LOG_LATENCY
    LOG_DONE
}

void NeuronExecutor::registerRuntimeInputsImpl() {
    START_TIMER
    #define NEURON_RUNTIME_SET_INPUT(inputIdx, ioBuf, size)                                 \
        CHECK_NEURON_ERROR(fnNeuronRuntime_setInput(this->getRuntime(),                     \
                                                    inputIdx,                               \
                                                    reinterpret_cast<void*>(ioBuf.buffer),  \
                                                    size,                                   \
                                                    {ioBuf.fd}));
    for (int i = 0; i < this->getNumInputs(); i++) {
        const auto sizeAllocated = this->getInputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelInputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Input[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_RUNTIME_SET_INPUT(i, this->getInput(i), sizeAllocated)
    }

    #undef NEURON_RUNTIME_SET_INPUT
    LOG_LATENCY
    LOG_DONE
}

void NeuronExecutor::registerRuntimeOutputsImpl() {
    START_TIMER
    #define NEURON_RUNTIME_SET_OUTPUT(outputIdx, ioBuf, size)                                   \
        CHECK_NEURON_ERROR(fnNeuronRuntime_setOutput(this->getRuntime(),                        \
                                                     outputIdx,                                 \
                                                     reinterpret_cast<void*>(ioBuf.buffer),     \
                                                     size,                                      \
                                                     {ioBuf.fd}));
    for (int i = 0; i < this->getNumOutputs(); i++) {
        const auto sizeAllocated = this->getOutputBufferSizeBytes(i);
        const auto sizeRequired = this->getModelOutputSizeBytes(i);
        if (sizeAllocated < sizeRequired) {
            LOG(ERROR) << "Insufficient buffer allocated for Output[" << i << "]: Allocated "
                       << sizeAllocated << " but need " << sizeRequired;
        }
        // import_forever requires the full allocated size during the first call to set input/output
        NEURON_RUNTIME_SET_OUTPUT(i, this->getOutput(i), sizeAllocated)
    }

    #undef NEURON_RUNTIME_SET_OUTPUT
    LOG_LATENCY
    LOG_DONE
}

void NeuronExecutor::setRuntimeOffsetedInput(const size_t index, const size_t offset) {
    const auto& ioBuf = this->getInput(index);
    CHECK_NEURON_ERROR(
        fnNeuronRuntime_setOffsetedInput(this->getRuntime(),
                                         index,
                                         reinterpret_cast<void*>(ioBuf.buffer),
                                         this->getModelInputSizeBytes(index),
                                         {ioBuf.fd},
                                         offset));
}

size_t NeuronExecutor::getRuntimeNumInputs() const {
    size_t numInputs;
    CHECK_NEURON_ERROR(fnNeuronRuntime_getInputNumber(this->getRuntime(), &numInputs));
    return numInputs;
}

size_t NeuronExecutor::getRuntimeNumOutputs() const {
    size_t numOutputs;
    CHECK_NEURON_ERROR(fnNeuronRuntime_getOutputNumber(this->getRuntime(), &numOutputs));
    return numOutputs;
}

size_t NeuronExecutor::getRuntimeInputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t inputSizeBytes;
    CHECK_NP_ERROR(fnNeuronRuntime_getInputPaddedSize(this->getRuntime(), index, &inputSizeBytes));

    RuntimeAPIDimensions dims;
    fnNeuronRuntime_getInputPaddedDimensions(this->getRuntime(), index, &dims);
    LOG(DEBUG) << this->getModelPath() << ":\n Input[" << index << "] Size (padded): "
               << inputSizeBytes << "\n Input[" << index << "] Dims (padded): "
               << dims.dimensions[0] << "x"
               << dims.dimensions[1] << "x"
               << dims.dimensions[2] << "x"
               << dims.dimensions[3];
    return inputSizeBytes;
}

size_t NeuronExecutor::getRuntimeOutputSizeBytes(const size_t index) const {
    // NOTE: Assume user model is always with suppress-io
    size_t outputSizeBytes;
    CHECK_NP_ERROR(fnNeuronRuntime_getOutputPaddedSize(this->getRuntime(), index, &outputSizeBytes));

    RuntimeAPIDimensions dims;
    fnNeuronRuntime_getOutputPaddedDimensions(this->getRuntime(), index, &dims);
    LOG(DEBUG) << this->getModelPath() << ":\n Output[" << index << "] Size (padded): "
               << outputSizeBytes << "\n Output[" << index << "] Dims (padded): "
               << dims.dimensions[0] << "x"
               << dims.dimensions[1] << "x"
               << dims.dimensions[2] << "x"
               << dims.dimensions[3];
    return outputSizeBytes;
}

void NeuronExecutor::getRuntimeInputShape(const size_t index, uint32_t* shape) const {
    RuntimeAPIDimensions cacheShape;
    fnNeuronRuntime_getInputPaddedDimensions(this->getRuntime(), index, &cacheShape);
    std::memcpy(shape, cacheShape.dimensions, sizeof(RuntimeAPIDimensions::dimensions));
}
