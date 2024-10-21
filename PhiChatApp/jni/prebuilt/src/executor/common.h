#pragma once
#include "common/logging.h"
#include "common/timer.h"
#include <android/log.h>
#include <chrono>
#include <fstream>

// Use either __PRETTY_FUNCTION__ or __FUNCTION__
#define BUILT_IN_FUNC_NAME __FUNCTION__

#define LOG_ENTER LOG(DEBUG) << "[" << BUILT_IN_FUNC_NAME << "] entered";
#define LOG_DONE LOG(DEBUG) << "[" << BUILT_IN_FUNC_NAME << "] done";
#define LOG_LATENCY LOG(DEBUG, "llm_sdk_latency") << BUILT_IN_FUNC_NAME << ": " \
                                                  << this->mTimer.reset() * 1000 << " ms";
#define START_TIMER this->mTimer.reset();

#ifdef NDEBUG   // release build
static constexpr bool kEnableLogOk = false;
#else
static constexpr bool kEnableLogOk = true;
#endif

#define LOG_TAG "CHECK_API"
#define CHECK_NEURON_ERROR(code)                                                                \
do {                                                                                            \
    const auto _ret = (code);                                                                   \
    if (_ret != NEURONRUNTIME_NO_ERROR) {                                                       \
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "NEURON API error (%d, %s, line %d)",   \
                _ret, __FILE__, __LINE__);                                                      \
    } else if constexpr (kEnableLogOk) {                                                      \
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "NEURON API OK (%d, %s, line %d)",      \
        _ret, __FILE__, __LINE__);                                                              \
    }                                                                                           \
  } while (0)

#define CHECK_NP_ERROR(code)                                                                     \
do {                                                                                             \
    const auto _ret = (code);                                                                    \
    if (_ret != 0) {                                                                             \
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "NeuroPilot API error (%d, %s, line %d)",\
                _ret, __FILE__, __LINE__);                                                       \
    } else if constexpr (kEnableLogOk) {                                                       \
        __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, "NeuroPilot API OK (%d, %s, line %d)",   \
                _ret, __FILE__, __LINE__);                                                       \
    }                                                                                            \
  } while (0)

#define CALLOC_BUFFER(_buffer, length)                                              \
do {                                                                                \
    _buffer = reinterpret_cast<void*>(calloc(1, length));                           \
    if (_buffer == nullptr) {                                                       \
        __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, "calloc fail (%s, line %d)",\
                   __FILE__, __LINE__);                                             \
    }                                                                               \
  } while (0)

#define RELEASE_BUFFER(buffer)              \
do {                                        \
    if (buffer != nullptr) {                \
        free(buffer);                       \
        buffer = nullptr;                   \
    }                                       \
  } while (0)

// Returns true if file can be opened, false if otherwise.
inline bool checkFile(const std::string& path) {
    std::ifstream ifile(path);
    return !ifile.fail();
}