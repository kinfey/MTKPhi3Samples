#include "executor/executor.h"
#include "runtime/neuron_runtime.h"
#include "common/logging.h"

#include <algorithm>

#include <filesystem>
#include <fstream>
#include <sstream>

namespace fs = std::filesystem;

void Executor::initialize() {
    if (isInitialized()) {
        return;
    }
    initRuntimes(); // Need to know the number of IO here for usdk
    initModelIOInfo(); // Get number of IO, and their sizes in bytes
    preInitBufferProcess(); // Optional preprocessing after model IO info is obtained.
    initBuffer(); // Allocate buffer based on the model IO info
    postInitBufferProcess(); // Optional postprocessing after model IO buffers have been allocated.
    mIsInitialized = true;
    LOG_DONE
}

void Executor::release() {
    releaseRuntimes();
    releaseBuffer();
}

void Executor::updateModelIO() {
    preInitBufferProcess(); // Optional preprocessing after model IO info is obtained.

    // Update actual used IO sizes by the model
    const size_t numInputs = getRuntimeNumInputs();
    const size_t numOutputs = getRuntimeNumOutputs();
    if (numInputs != getNumInputs()) {
        LOG(WARN) << "updateModelIO: Existing num inputs (" << getNumInputs() << ") != "
                     "new num inputs (" << numInputs << ").";
    }
    if (numOutputs != getNumOutputs()) {
        LOG(WARN) << "updateModelIO: Existing num outputs (" << getNumOutputs() << ") != "
                     "new num outputs (" << numOutputs << ").";
    }
    mInputs.resize(numInputs);
    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        auto& sizeAllocated = mInputs[inputIdx].sizeBytes;
        auto& sizeRequired = mInputs[inputIdx].usedSizeBytes;
        const auto before = sizeRequired;

        sizeRequired = getRuntimeInputSizeBytes(inputIdx); // Update
        if (sizeAllocated < sizeRequired) {
            LOG(WARN) << "updateModelIO: Insufficient buffer size for input[" << inputIdx << "]. "
                      << "Requires " << sizeRequired << " but only allocated " << sizeAllocated;
        }
        if (before != sizeRequired) {
            LOG(DEBUG) << "Update Input[" << inputIdx << "] size: " << before << " -> "
                       << sizeRequired;
        }
    }
    mOutputs.resize(numOutputs);
    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        auto& sizeAllocated = mOutputs[outputIdx].sizeBytes;
        auto& sizeRequired = mOutputs[outputIdx].usedSizeBytes;
        const auto before = sizeRequired;

        sizeRequired = getRuntimeOutputSizeBytes(outputIdx); // Update
        if (sizeAllocated < sizeRequired) {
            LOG(WARN) << "updateModelIO: Insufficient buffer size for output[" << outputIdx << "]. "
                      << "Requires " << sizeRequired << " but only allocated " << sizeAllocated;
        }
        if (before != sizeRequired) {
            LOG(DEBUG) << "Update Output[" << outputIdx << "] size: " << before << " -> "
                       << sizeRequired;
        }
    }
}

bool Executor::isInitialized() const {
    return mIsInitialized;
}

void Executor::registerRuntimeIO() {
    registerRuntimeInputs();
    registerRuntimeOutputs();
    LOG_DONE
}

void Executor::registerRuntimeInputs() {
    // Ensure all input buffers are allocated/initialized
    auto isNotAllocated = [](const IOBuffer& ioBuf) {
        return !ioBuf.isAllocated();
    };
    const auto it = std::find_if(mInputs.begin(), mInputs.end(), isNotAllocated);
    if (it != mInputs.end()) {
        const size_t notAllocatedIdx = std::distance(mInputs.begin(), it);
        LOG(FATAL) << "[registerRuntimeInputs] Attempting to register an uninitialized input "
                   << "buffer (index=" << notAllocatedIdx << ")";
    }
    CHECK_GT(getNumInputs(), 0) << "[registerRuntimeInputs] No model input allocated. "
                                   "Please check if the model has been loaded properly.";
    registerRuntimeInputsImpl();
    mIsInputRegistered = true;
}

void Executor::registerRuntimeOutputs() {
    // Ensure all output buffers are allocated/initialized
    auto isNotAllocated = [](const IOBuffer& ioBuf) {
        return !ioBuf.isAllocated();
    };
    const auto it = std::find_if(mOutputs.begin(), mOutputs.end(), isNotAllocated);
    if (it != mOutputs.end()) {
        const size_t notAllocatedIdx = std::distance(mOutputs.begin(), it);
        LOG(FATAL) << "[registerRuntimeOutputs] Attempting to register an uninitialized "
                   << "output buffer (index=" << notAllocatedIdx << ")";
    }
    CHECK_GT(getNumOutputs(), 0) << "[registerRuntimeOutputs] No model output allocated. "
                                    "Please check if the model has been loaded properly.";
    registerRuntimeOutputsImpl();
    mIsOutputRegistered = true;
}

template <typename T>
void Executor::runInference(const std::vector<T>& input) {
    runInference(input.data(), input.size() * sizeof(T));
    LOG_DONE
}

void Executor::runInference(const void* input, size_t inputSize) {
    requiresInit();
    setModelInput(input, inputSize);
    registerRuntimeInputs();
    runInference();
    LOG_DONE
}

void Executor::runInference() {
    requiresInit();
    if (!mIsInputRegistered) {
        LOG(FATAL) << "[runInference] Model input has not been registered to runtime";
    } else if (!mIsOutputRegistered) {
        LOG(FATAL) << "[runInference] Model output has not been registered to runtime";
    }
    runInferenceImpl();
}

void Executor::setNumInputs(const size_t numInputs) {
    const auto oldNumInputs = getNumInputs();
    if (oldNumInputs > numInputs) {
        LOG(WARN) << "Reducing the number of inputs from " << oldNumInputs << " to "
                  << numInputs;
    }
    mInputs.resize(numInputs);
}

void Executor::setNumOutputs(const size_t numOutputs) {
    const auto oldNumOutputs = getNumOutputs();
    if (oldNumOutputs > numOutputs) {
        LOG(WARN) << "Reducing the number of Outputs from " << oldNumOutputs << " to "
                  << numOutputs;
    }
    mOutputs.resize(numOutputs);
}

// Either assign buffer pointer or copy buffer content. If the index is out of input buffer vector,
// the buffer will be set aside to the preallocate buffer container.
void Executor::setModelInput(const IOBuffer& buffer, const size_t index) {
    // If current buffer (mInputs[index]) is not allocated then copy the buffer POINTER,
    // otherwise copy the content from buffer to current buffer.

    if (index >= mInputs.size()) {
        // initModelIOInfo() has not yet been called
        mInputs.resize(index + 1);
    }
    auto& curBuffer = mInputs[index];
    if (!curBuffer.isAllocated()) {
        // Not yet allocated, so share the buffer
        curBuffer = buffer;
        mIsInputRegistered = false;
    } else {
        // Already allocated, do memcpy
        setModelInput(buffer.buffer, buffer.sizeBytes, index);
    }
    LOG_DONE
}

// Copy buffer content
void Executor::setModelInput(const void* buffer, const size_t sizeBytes, const size_t index) {
    auto& input = getInput(index);
    if (input.sizeBytes < sizeBytes) {
        LOG(ERROR) << "[setModelInput] Insufficient buffer size ("
                   << input.sizeBytes << ") to hold the required target data size ("
                   << sizeBytes << ")";
        return;
    }
    std::memcpy(input.buffer, buffer, sizeBytes);
    auto& curInputSizeBytes = getInput(index).usedSizeBytes;
    if (curInputSizeBytes != sizeBytes) {
        LOG(DEBUG) << "[setModelInput]: Update model input[" << index << "] size bytes from "
                   <<  curInputSizeBytes << " to " << sizeBytes;
        curInputSizeBytes = sizeBytes;
    }
    mIsInputRegistered = false;
    LOG_DONE
}

void Executor::reserveInputBuffer(const size_t index) {
    mReservedInputBuffers.insert(index);
}

void Executor::reserveOutputBuffer(const size_t index) {
    mReservedOutputBuffers.insert(index);
}

size_t Executor::getNumInputs() const {
    return mInputs.size();
}

size_t Executor::getNumOutputs() const {
    return mOutputs.size();
}

//=================//
// Get model input //
//=================//

const IOBuffer& Executor::getInput(const size_t index) const {
    CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
    return mInputs[index];
}

IOBuffer& Executor::getInput(const size_t index) {
    CHECK_LT(index, getNumInputs()) << "getInput(): Index out of range.";
    return mInputs[index];
}

void* Executor::getInputBuffer(const size_t index) {
    return getInput(index).buffer;
}

size_t Executor::getInputBufferSizeBytes(const size_t index) const {
    return getInput(index).sizeBytes; // Actual allocated buffer size
}

size_t Executor::getModelInputSizeBytes(const size_t index) const {
    return getInput(index).usedSizeBytes; // Actual size used by the model
}

//==================//
// Get model output //
//==================//

const IOBuffer& Executor::getOutput(const size_t index) const {
    CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
    return mOutputs[index];
}

IOBuffer& Executor::getOutput(const size_t index) {
    CHECK_LT(index, getNumOutputs()) << "getOutput(): Index out of range.";
    return mOutputs[index];
}

void* Executor::getOutputBuffer(const size_t index) {
    return getOutput(index).buffer;
}

size_t Executor::getOutputBufferSizeBytes(const size_t index) const {
    return getOutput(index).sizeBytes; // Actual allocated buffer size
}

size_t Executor::getModelOutputSizeBytes(const size_t index) const {
    return getOutput(index).usedSizeBytes; // Actual size used by the model
}

//==================//
// Model IO linkage //
//==================//

void Executor::linkModelIO(const size_t inputIndex, const size_t outputIndex) {
    mModelInToOutIndexLinks.emplace(inputIndex, outputIndex);
    LOG_DONE
}

void Executor::setModelIOLink(const std::unordered_map<size_t, size_t>& modelIOLinks) {
    mModelInToOutIndexLinks = modelIOLinks;
    LOG_DONE
}

bool Executor::inputHasLinkToOutput(const size_t inputIndex) const {
    // return (inputIndex in mModelInToOutIndexLinks)
    return mModelInToOutIndexLinks.find(inputIndex) != mModelInToOutIndexLinks.end();
}

size_t Executor::getLinkedOutputIndex(const size_t inputIndex) {
    if (inputHasLinkToOutput(inputIndex)) {
        LOG_DONE
        return mModelInToOutIndexLinks[inputIndex];
    }
    LOG_DONE
    return kInvalidIndex;
}

void Executor::initModelIOInfo() {
    // Inputs
    const size_t numInputs = getRuntimeNumInputs();
    setNumInputs(numInputs);
    LOG(DEBUG) << "numInputs = " << numInputs;
    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        size_t inputSize = getRuntimeInputSizeBytes(inputIdx);
        auto& input = mInputs[inputIdx];
        input.sizeBytes = inputSize; // Initialize buffer size to input size
        input.usedSizeBytes = inputSize;
    }
    if (numInputs == 0) {
        LOG(FATAL) << "[Executor] Failed to get model input info.";
    }

    // Outputs
    const size_t numOutputs = getRuntimeNumOutputs();
    setNumOutputs(numOutputs);
    LOG(DEBUG) << "numOutputs = " << numOutputs;
    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        size_t outputSize = getRuntimeOutputSizeBytes(outputIdx);
        auto& output = mOutputs[outputIdx];
        output.sizeBytes = outputSize; // Initialize buffer size to output size
        output.usedSizeBytes = outputSize;
    }
    if (numOutputs == 0) {
        LOG(FATAL) << "[Executor] Failed to get model output info.";
    }
    LOG_DONE
}

//=================================//
// Buffer Initialization & Release //
//=================================//

void Executor::initBuffer() {
    const size_t numInputs = getNumInputs();
    const size_t numOutputs = getNumOutputs();
    if (numInputs == 0 || numOutputs == 0) {
        LOG(FATAL) << "Attempt to init buffer before model IO info is retrieved.";
    }

    auto isInputBufferReserved = [&](const size_t index) {
        return mReservedInputBuffers.find(index) != mReservedInputBuffers.end();
    };

    auto isOutputBufferReserved = [&](const size_t index) {
        return mReservedOutputBuffers.find(index) != mReservedOutputBuffers.end();
    };

    for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
        auto& output = getOutput(outputIdx);
        if (output.isAllocated()) {
            LOG(DEBUG) << "Init Buffer: Reusing preallocated output buffer " << outputIdx;
            continue;
        }
        if (isOutputBufferReserved(outputIdx)) {
            LOG(DEBUG) << "Init Buffer: Skip allocation for reserved output buffer " << outputIdx;
            continue;
        }
        if (!allocateMemory(output)) {
            LOG(ERROR) << "Failed to allocate memory for output[" << outputIdx << "]";
        }
        LOG(DEBUG) << "Init Buffer: allocating output[" << outputIdx << "]";
        mAllocatedBuffers.push_back(output);
    }

    for (size_t inputIdx = 0; inputIdx < numInputs; inputIdx++) {
        auto& input = getInput(inputIdx);
        if (input.isAllocated()) {
            LOG(DEBUG) << "Init Buffer: Reusing preallocated input buffer " << inputIdx;
            continue;
        }
        if (isInputBufferReserved(inputIdx)) {
            LOG(DEBUG) << "Init Buffer: Skip allocation for reserved input buffer " << inputIdx;
            continue;
        }
        if (!inputHasLinkToOutput(inputIdx)) {
            if (!allocateMemory(input)) {
                LOG(ERROR) << "Failed to allocate memory for input[" << inputIdx << "]";
            }
            LOG(DEBUG) << "Init Buffer: allocating input[" << inputIdx << "] with size="
                       << input.sizeBytes;
            mAllocatedBuffers.push_back(input);
            continue;
        }

        const size_t inputSizeBytes =  input.sizeBytes;
        const size_t linkedOutputIdx = getLinkedOutputIndex(inputIdx);
        const auto& linkedOutput = getOutput(linkedOutputIdx);
        const size_t linkedOutputSizeBytes = linkedOutput.sizeBytes;
        if (inputSizeBytes != linkedOutputSizeBytes) {
            LOG(FATAL) << "Init Buffer: Mismatch size between linked input/output! Input["
                       << inputIdx << "].size=" << inputSizeBytes << ", Output[" << linkedOutputIdx
                       << "].size=" << linkedOutputSizeBytes;
        }
        // Reuse the same buffer from output since they are linked
        input = linkedOutput;
        LOG(DEBUG) << "Init Buffer: input[" << inputIdx << "] reuse output["
                   <<  linkedOutputIdx << "]";
    }
    LOG_DONE
}

void Executor::releaseBuffer() {
    for (size_t i = 0; i < mAllocatedBuffers.size(); i++) {
        if (!releaseMemory(mAllocatedBuffers[i])) {
            LOG(ERROR) << "Failed to release memory for mAllocatedBuffers[" << i << "]";
        }
    }
    mAllocatedBuffers.clear();
    LOG_DONE
}

//===================//
// Init verification //
//===================//

void Executor::requiresInit() const {
    if (!mIsInitialized) {
        LOG(FATAL) << "Executor is not initialized. Please call initialize().";
    }
    LOG_DONE
}

//===================//
// Export IO to disk //
//===================//

bool Executor::saveInputs(const std::string& directory, const std::string& name) const {
    const std::string outName = name.size() > 0 ? name : fs::path(getModelPath()).stem().string();
    fs::path outdir = fs::path(directory) / outName;
    if (!fs::exists(outdir)) {
        LOG(INFO) << "[saveInputs] Creating directory: " << outdir;
        fs::create_directories(outdir); // mkdir -p <outdir>
    }

    for (int i = 0; i < getNumInputs(); i++) {
        std::stringstream filename;
        filename << "input" << i << ".bin";
        const auto outpath = outdir / filename.str();
        std::fstream fout(outpath, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!fout) {
            LOG(ERROR) << "[saveInputs] Unable to create the file: " << outpath;
            return false;
        }
        const auto input = getInput(i);
        const auto buffer = reinterpret_cast<char*>(input.buffer);
        const auto size = input.usedSizeBytes;
        fout.write(buffer, size);
        if (!fout) {
            LOG(ERROR) << "[saveInputs] Unable to write to file: " << outpath;
            return false;
        }
    }
    LOG(INFO) << "[saveInputs] Exported model inputs for " << outName;
    return true;
}

bool Executor::saveOutputs(const std::string& directory, const std::string& name) const {
    const std::string outName = name.size() > 0 ? name : fs::path(getModelPath()).stem().string();
    fs::path outdir = fs::path(directory) / outName;
    if (!fs::exists(outdir)) {
        LOG(INFO) << "[saveOutputs] Creating directory: " << outdir;
        fs::create_directories(outdir); // mkdir -p <outdir>
    }

    for (int i = 0; i < getNumOutputs(); i++) {
        std::stringstream filename;
        filename << "output" << i << ".bin";
        const auto outpath = outdir / filename.str();
        std::fstream fout(outpath, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!fout) {
            LOG(ERROR) << "[saveOutputs] Unable to create the file: " << outpath;
            return false;
        }
        const auto output = getOutput(i);
        const auto buffer = reinterpret_cast<char*>(output.buffer);
        const auto size = output.usedSizeBytes;
        fout.write(buffer, size);
        if (!fout) {
            LOG(ERROR) << "[saveOutputs] Unable to write to file: " << outpath;
            return false;
        }
    }
    LOG(INFO) << "[saveOutputs] Exported model outputs for " << outName;
    return true;
}

// Explicit instantiation of runInference for some integral types
template void Executor::runInference<int>(const std::vector<int>&);
template void Executor::runInference<int16_t>(const std::vector<int16_t>&);
template void Executor::runInference<float>(const std::vector<float>&);
template void Executor::runInference<__fp16>(const std::vector<__fp16>&);