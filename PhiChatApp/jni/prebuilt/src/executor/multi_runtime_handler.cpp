#include "executor/common.h"
#include "multi_runtime_handler.h"

#include <string>
#include <vector>

void MultiRuntimeHandler::initRuntimes() {
    const size_t numRuntimes = mModelPaths.size();
    mRuntimes.resize(numRuntimes);
    if (!canRuntimesCoexist()) {
        selectRuntime(mDefaultRuntimeIdx);
        DCHECK_EQ(getRuntime(), nullptr) << "Runtime[" << mDefaultRuntimeIdx << "] is "
                                            "initialized before calling initRuntimes!";
        void* runtime = createRuntime(mModelPaths[mDefaultRuntimeIdx]);
        setRuntime(runtime);
        LOG(DEBUG) << "initRuntimes(): Loaded single exclusive model (Total=" << numRuntimes << ")";
        return;
    }

    for (size_t i = 0; i < numRuntimes; i++) {
        selectRuntime(i);
        DCHECK_EQ(getRuntime(), nullptr) << "Runtime[" << i << "] is initialized before "
                                            "calling initRuntimes!";
        void* runtime = createRuntime(mModelPaths[i]);
        setRuntime(runtime);
    }
    selectRuntime(mDefaultRuntimeIdx); // Select the default runtime
    LOG(DEBUG) << "initRuntimes(): Loaded multiple models (Total=" << numRuntimes << ")";
}

void MultiRuntimeHandler::releaseRuntimes() {
    if (!canRuntimesCoexist()) {
        // Select the current runtime
        releaseRuntime(getRuntime());
        setRuntime(nullptr);
        LOG(DEBUG) << "releaseRuntimes(): Released single runtime";
        return;
    }

    const size_t numRuntimes = getNumRuntimes();
    for (size_t i = 0; i < numRuntimes; i++) {
        selectRuntime(i);
        releaseRuntime(getRuntime());
        setRuntime(nullptr);
    }
    LOG(DEBUG) << "releaseRuntimes(): Released multiple runtimes (Total=" << getNumRuntimes() << ")";
}

void* MultiRuntimeHandler::getRuntime() const {
    DCHECK_LT(mCurrentRuntimeIdx, getNumRuntimes()) << "Index out of range.";
    return mRuntimes[mCurrentRuntimeIdx];
}

void MultiRuntimeHandler::setRuntime(void* runtime) {
    DCHECK_LT(mCurrentRuntimeIdx, getNumRuntimes()) << "Index out of range.";
    mRuntimes[mCurrentRuntimeIdx] = runtime;
}

void MultiRuntimeHandler::setDefaultRuntimeIndex(const size_t index) {
    mDefaultRuntimeIdx = index;
}

size_t MultiRuntimeHandler::getRuntimeIndex() const {
    return mCurrentRuntimeIdx;
}

void MultiRuntimeHandler::selectRuntime(const size_t index) {
    CHECK_LT(index, getNumRuntimes()) << "selectRuntime(): Index out of range.";

    if (mCurrentRuntimeIdx == index) {
        return; // Do nothing
    } else if (canRuntimesCoexist()) {
        mCurrentRuntimeIdx = index;
        LOG(DEBUG) << "Selected runtime[" << index << "]: " << mModelPaths[index];
        return;
    }

    // Release current runtime if already loaded
    if (getRuntime() != nullptr) {
        releaseRuntime(getRuntime());
        setRuntime(nullptr);
    }

    // Load new runtime
    mCurrentRuntimeIdx = index;
    void* runtime = createRuntime(mModelPaths[index]);
    setRuntime(runtime);
    LOG(DEBUG) << "Selected exclusive runtime[" << index << "]: " << mModelPaths[index];
}

size_t MultiRuntimeHandler::getNumRuntimes() const {
    DCHECK_EQ(mRuntimes.size(), mModelPaths.size())
        << "Please ensure that initRuntimes() is called first.";
    return mRuntimes.size();
}

std::string MultiRuntimeHandler::getModelPath() const {
    DCHECK_LT(mCurrentRuntimeIdx, getNumRuntimes()) << "Index out of range.";
    return mModelPaths[mCurrentRuntimeIdx];
}

size_t MultiRuntimeHandler::addRuntime(const std::string& modelPath) {
    mModelPaths.push_back(modelPath);
    mRuntimes.push_back(nullptr);
    const size_t newRuntimeIdx = mRuntimes.size() - 1;
    if (canRuntimesCoexist()) {
        // Create runtime immediately
        const auto oldRuntimeIdx = getRuntimeIndex();
        selectRuntime(newRuntimeIdx);
        void* runtime = createRuntime(mModelPaths[mDefaultRuntimeIdx]);
        setRuntime(runtime);
        // Switch back to original runtime
        selectRuntime(oldRuntimeIdx);
    }
    return newRuntimeIdx;
}
