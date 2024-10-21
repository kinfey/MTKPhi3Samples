#pragma once

#include <string>
#include <vector>

class MultiRuntimeHandler {
public:
    explicit MultiRuntimeHandler(const std::vector<std::string>& modelPaths,
                                 const size_t defaultRuntimeIdx = 0)
        : mModelPaths(modelPaths), mDefaultRuntimeIdx(defaultRuntimeIdx) {}

    explicit MultiRuntimeHandler(const std::string& modelPath, const size_t defaultRuntimeIdx = 0)
        : mModelPaths({modelPath}), mDefaultRuntimeIdx(defaultRuntimeIdx) {}

    virtual ~MultiRuntimeHandler() {}

protected:
    // Initialize all runtimes if they can coexist, otherwise initialize the default runtime
    void initRuntimes();

    // Release all active runtimes
    void releaseRuntimes();

    // Get the current runtime
    void* getRuntime() const;

    // Set the current runtime
    void setRuntime(void* runtime);

    // Set the default active runtime for use by initRuntimes()
    void setDefaultRuntimeIndex(const size_t index);

    // Get the current runtime index
    size_t getRuntimeIndex() const;

    // Select the runtime of given index
    void selectRuntime(const size_t index);

    // Get total number of runtimes
    size_t getNumRuntimes() const;

    // Get the model path of the current runtime
    std::string getModelPath() const;

    // Add new runtime post initialization
    size_t addRuntime(const std::string& modelPath);

private:
    // Create and returns a runtime given a model path. To be implemented by subclass.
    virtual void* createRuntime(const std::string& modelPath) = 0;

    // Release a runtime. To be implemented by subclass.
    virtual void releaseRuntime(void* runtime) = 0;

    // Determine whether multiple runtimes are allowed to be active concurrently.
    virtual bool canRuntimesCoexist() const { return false; }

private:
    std::vector<std::string> mModelPaths;
    std::vector<void*> mRuntimes;
    size_t mCurrentRuntimeIdx = 0;
    size_t mDefaultRuntimeIdx = 0;
};
