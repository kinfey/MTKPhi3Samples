#pragma once

#include <memory>
#include <mutex>

#include "executor/executor.h"
#include "runtime/api/neuron/NeuronAdapter.h"
#include "runtime/api/neuron/NeuronAdapterShim.h"
#include "runtime/api/neuron/Types.h"

class NeuronUsdkExecutor : public Executor {
private:
    struct UsdkRuntime {
        NeuronModel* model = nullptr;
        NeuronCompilation* compilation = nullptr;
        NeuronExecution* execution = nullptr;
    };

public:
    explicit NeuronUsdkExecutor(const std::vector<std::string>& modelPaths,
                                const std::string& sharedWeightsPath = "")
        : Executor(modelPaths),
          kSharedWeightsPath(sharedWeightsPath) {}

    virtual ~NeuronUsdkExecutor() {}

    virtual void initialize() override;

    virtual void release() override;

    virtual void runInferenceImpl() override;

    virtual void updateModelIO() override;

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
    bool loadDla(void* buffer, size_t size, UsdkRuntime* runtime);

    virtual bool canRuntimesCoexist() const override { return isSharedWeightsUsed(); }

    UsdkRuntime* getUsdkRuntime() { return reinterpret_cast<UsdkRuntime*>(this->getRuntime()); }

    UsdkRuntime* getUsdkRuntime() const { return reinterpret_cast<UsdkRuntime*>(this->getRuntime()); }

    void createUsdkNeuronMemory();

private:
    // A single mutex shared across all instances of NeuronUsdkExecutor for multi-threaded init
    inline static std::mutex mMutex;

    std::vector<NeuronMemory*> mCreatedNeuronMems;

    const std::string kSharedWeightsPath;

    const std::string kOptions = "--apusys-config \"{ \\\"high_addr\\\": true, \\\"import_forever\\\": true }\"";
};
