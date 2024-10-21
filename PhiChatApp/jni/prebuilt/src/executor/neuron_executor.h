
#pragma once

#include "executor/executor.h"
#include "runtime/neuron_runtime.h"

// Executor class that allows user to access one input and one output
class NeuronExecutor : public Executor {
public:
    explicit NeuronExecutor(const std::vector<std::string>& modelPaths,
                            const std::string& sharedWeightsPath = "")
        : Executor(modelPaths),
          kSharedWeightsPath(sharedWeightsPath) {}

    ~NeuronExecutor() {}

    virtual void runInferenceImpl() override;

protected:
    virtual void* createRuntime(const std::string& modelPath) override;
    virtual void releaseRuntime(void* runtime) override;

protected:
    virtual void registerRuntimeInputsImpl() override;
    virtual void registerRuntimeOutputsImpl() override;

    virtual void setRuntimeOffsetedInput(const size_t index, const size_t offset) override;

    virtual size_t getRuntimeNumInputs() const override;
    virtual size_t getRuntimeNumOutputs() const override;

    virtual size_t getRuntimeInputSizeBytes(const size_t index) const override;
    virtual size_t getRuntimeOutputSizeBytes(const size_t index) const override;

    virtual void getRuntimeInputShape(const uint64_t index, uint32_t* shape) const override;

protected:
    bool isSharedWeightsUsed() const;
    void loadSharedWeights(const size_t inputIndex);

private:
    virtual bool canRuntimesCoexist() const override { return isSharedWeightsUsed(); }

private:
    const std::string kSharedWeightsPath;

    // Runtimes delayed for release, which will be used when FMS (weight sharing) is enabled so each
    // where only negligible memory space is allocated to each runtime object in this case.
    std::vector<void*> mDelayedReleaseRuntimes;

    // NOTE: Starting from NP7, mEnvOptions will be ignored.
    EnvOptions mEnvOptions = {
        .deviceKind = kEnvOptHardware,
        .MDLACoreOption = MDLACoreMode::Auto
    };

    QoSOptions mQosOptions = {};
};
