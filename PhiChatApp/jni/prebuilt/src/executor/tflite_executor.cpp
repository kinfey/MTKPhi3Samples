#include "executor/tflite_executor.h"
#include "common/logging.h"

// NOTE: Must include "NeuralNetworksTypes.h" before "NeuroPilotTFLiteShim.h"
#include <android/NeuralNetworksTypes.h>
#include "runtime/api/tflite/NeuroPilotTFLiteShim.h"

using OptionType = ANeuralNetworksTFLiteOptions;

// Override for loop-based dynamic shape workaround
void TfliteExecutor::runInference(const void* input, size_t inputSize) {
    // ============ORIGINAL============
    // requiresInit();
    // setModelInput(input, inputSize);
    // registerRuntimeInputs();
    // runInference();
    // ================================
    this->requiresInit();
    this->setModelInput(input, inputSize); // Will update model input size

    const auto modelInputSize = this->getModelInputSizeBytes(0);
    const auto runtimeInputSize = getRuntimeInputSizeBytes(0);
    const size_t loopCount = modelInputSize / runtimeInputSize;
    if (modelInputSize % runtimeInputSize != 0) {
        LOG(FATAL) << "Provided input size (" << modelInputSize << ") is not a multiple of "
                   << "the expected model input size (" << runtimeInputSize << ").";
    }
    for (size_t loopIdx = 0; loopIdx < loopCount; loopIdx++) {
        for (size_t inputIdx = 0; inputIdx < this->getNumInputs(); inputIdx++) {
            mInputBufferOffsetBytes[inputIdx] = getRuntimeInputSizeBytes(inputIdx) * loopIdx;
        }
        for (size_t outputIdx = 0; outputIdx < this->getNumOutputs(); outputIdx++) {
            mOutputBufferOffsetBytes[outputIdx] = getRuntimeOutputSizeBytes(outputIdx) * loopIdx;
        }
        this->registerRuntimeInputs();
        runInference();
    }
}

void TfliteExecutor::runInferenceImpl() {
    START_TIMER
    // check
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_invoke(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime())
    ));

    for (size_t outputIdx = 0; outputIdx < this->getNumOutputs(); outputIdx++) {
        auto outBuffer = reinterpret_cast<char*>(this->getOutputBuffer(outputIdx));
        const auto runtimeOutputSize = getRuntimeOutputSizeBytes(outputIdx);
        // Workaround for manual loop-based dynamic shape
        const size_t outBufferOffset = mOutputBufferOffsetBytes[outputIdx];

        CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_getOutputTensorData(
            reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()), outputIdx,
            reinterpret_cast<void*>(outBuffer + outBufferOffset), runtimeOutputSize
        ));
        if (kOutputDequantFp16Scale > 0) {
            dequantToFp16(reinterpret_cast<int16_t*>(outBuffer + outBufferOffset), runtimeOutputSize / 2);
        }
    }
    LOG_LATENCY
    LOG_DONE
}

void TfliteExecutor::dequantToFp16(int16_t* buffer, const size_t numValToDequant) {
    auto fp16buffer = reinterpret_cast<__fp16*>(buffer);
    auto int16buffer = reinterpret_cast<int16_t*>(buffer);
    for (size_t i = 0; i < numValToDequant; i++) {
        auto dequant_fp32 = static_cast<float>(int16buffer[i]) * kOutputDequantFp16Scale;
        fp16buffer[i] = static_cast<__fp16>(dequant_fp32);
    }
}

void TfliteExecutor::releaseRuntime(void* runtime) {
    START_TIMER
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_free(reinterpret_cast<OptionType*>(mOptions)));
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_free(
        reinterpret_cast<ANeuralNetworksTFLite*>(runtime)
    ));
    LOG_LATENCY
    LOG_DONE
}

void* TfliteExecutor::createRuntime(const std::string& modelPath) {
    START_TIMER
    if (!checkFile(modelPath)) {
        LOG(FATAL) << "File not found: " << modelPath;
    }
    setTfliteOptions();
    void* runtime;
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_makeAdvTFLite(
        reinterpret_cast<ANeuralNetworksTFLite**>(&runtime),
        modelPath.c_str(), reinterpret_cast<OptionType*>(mOptions)
    ));
    LOG_LATENCY
    LOG_DONE
    return runtime;
}

// Workaround for manual loop-based dynamic shape (allocate max size)
void TfliteExecutor::preInitBufferProcess() {
    if (kMaxLoopCount > 1) {
        auto adjustIoBufSizes = [this](auto& ioBuf) {
            ioBuf.sizeBytes *= kMaxLoopCount;
        };
        std::for_each(this->mInputs.begin(), this->mInputs.end(), adjustIoBufSizes);
        std::for_each(this->mOutputs.begin(), this->mOutputs.end(), adjustIoBufSizes);
    }
    mInputBufferOffsetBytes.resize(kMaxLoopCount);
    mOutputBufferOffsetBytes.resize(kMaxLoopCount);
}

void TfliteExecutor::registerRuntimeInputsImpl() {
    START_TIMER
    for (size_t inputIdx = 0; inputIdx < this->getNumInputs(); inputIdx++) {
        //get input tensor byte size and set it
        auto inBuffer = reinterpret_cast<char*>(this->getInputBuffer(inputIdx));
        const auto runtimeInputSize = getRuntimeInputSizeBytes(inputIdx);
        // Workaround for manual loop-based dynamic shape
        const size_t inBufferOffset = mInputBufferOffsetBytes[inputIdx];

        // Use getModelInputSizeBytes instead of getRuntimeInputSizeBytes to get accurate input size
        // for dynamic shape TFLite case. The function getModelInputSizeBytes can reflect the
        // realtime input size changes set by setModelInput.
        CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_setInputTensorData(
            reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()), inputIdx,
            reinterpret_cast<const void*>(inBuffer + inBufferOffset), runtimeInputSize
        ));
    }
    this->registerRuntimeOutputs(); // dummy call this since tflite runtime doesn't require registering model output
    LOG_LATENCY
    LOG_DONE
}

size_t TfliteExecutor::getRuntimeNumInputs() const {
    int32_t numInputs;
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorCount(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()),
        TFLITE_BUFFER_TYPE_INPUT, &numInputs
    ));
    return numInputs;
}

size_t TfliteExecutor::getRuntimeNumOutputs() const {
    int32_t numOutputs;
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorCount(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()),
        TFLITE_BUFFER_TYPE_OUTPUT, &numOutputs
    ));
    return numOutputs;
}

size_t TfliteExecutor::getRuntimeInputSizeBytes(const size_t index) const {
    size_t inputSizeBytes;
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()),
        TFLITE_BUFFER_TYPE_INPUT, index, &inputSizeBytes
    ));
    return inputSizeBytes;
}

size_t TfliteExecutor::getRuntimeOutputSizeBytes(const size_t index) const {
    size_t outputSizeBytes;
    CHECK_NP_ERROR(ANeuroPilotTFLiteWrapper_getTensorByteSize(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()),
        TFLITE_BUFFER_TYPE_OUTPUT, index, &outputSizeBytes
    ));
    return outputSizeBytes;
}

void TfliteExecutor::getRuntimeInputShape(const uint64_t index, uint32_t* shape) const {
    int tmpShape[4];
    ANeuroPilotTFLiteWrapper_getTensorDimensions(
        reinterpret_cast<ANeuralNetworksTFLite*>(this->getRuntime()),
        TFLITE_BUFFER_TYPE_INPUT, index, tmpShape
    );
    // FIXME: Assumes shape is 4 dim
    for (size_t i = 0; i < 4; i++) {
        const auto dim = tmpShape[i];
        if (dim < 0) {
            LOG(WARN) << "TFLite tensor shape[" << i << "] contains negative dim (" << dim << ")";
        }
        shape[i] = static_cast<uint32_t>(dim);
    }
}

void TfliteExecutor::setTfliteOptions() {
    START_TIMER
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_create(reinterpret_cast<OptionType**>(&mOptions)));
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_setAccelerationMode(
        reinterpret_cast<OptionType*>(mOptions), NpAccelerationMode::NP_ACCELERATION_CPU
    ));
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_setDisallowNnApiCpu(
        reinterpret_cast<OptionType*>(mOptions), false
    ));
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_setAllowFp16PrecisionForFp32(
        reinterpret_cast<OptionType*>(mOptions), 0
    ));
    CHECK_NP_ERROR(ANeuralNetworksTFLiteOptions_setPreference(
        reinterpret_cast<OptionType*>(mOptions), ExecutionPreference::kSustainedSpeed
    ));
    LOG_LATENCY
    LOG_DONE
}
