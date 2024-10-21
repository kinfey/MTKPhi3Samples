#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <type_traits>

#define __LLM_SDK_LOGTAG__ "llm_sdk"

// All log severity will log to logcat and stdout/stderr, except DEBUG will only log to logcat.
enum class LogSeverity {
    DEBUG = 0,
    INFO,
    WARN,
    ERROR,
    FATAL
};

class StreamLogger {
public:
    StreamLogger(const LogSeverity logSeverity, const char* tag, const char* file = nullptr,
                 const size_t line = 0);
    ~StreamLogger();
    std::ostream& stream();

private:
    bool shouldAbort() const;
    std::ostream& getOutStream() const;
    static std::ostream& getNullStream();

private:
    std::ostringstream mMsgStream;
    const LogSeverity kLogSeverity;
    const char* kTag;

    const char* kFile = nullptr;
    const size_t kLine = 0;
};

bool runtimeShouldLog(const LogSeverity logSeverity);

#ifdef NDEBUG   // release build
static constexpr bool kEnableDChecks = false;
#else
static constexpr bool kEnableDChecks = true;
#endif

#define ASSERT_LOGGER \
    StreamLogger(LogSeverity::FATAL, "ASSERT_FAILED", __FILE__, __LINE__).stream()

#define CHECK(cond) \
    if (!(cond)) ASSERT_LOGGER << "Check failed: " #cond " "

#define CHECK_OP(LHS, RHS, OP) \
    if (!(LHS OP RHS)) ASSERT_LOGGER << "Check failed: " << #LHS " " #OP " " #RHS \
                                     << " (" #LHS "=" << LHS << ", " #RHS "=" << RHS << "). "

#define CHECK_EQ(LHS, RHS) CHECK_OP(LHS, RHS, ==)
#define CHECK_NE(LHS, RHS) CHECK_OP(LHS, RHS, !=)
#define CHECK_LT(LHS, RHS) CHECK_OP(LHS, RHS, <)
#define CHECK_LE(LHS, RHS) CHECK_OP(LHS, RHS, <=)
#define CHECK_GE(LHS, RHS) CHECK_OP(LHS, RHS, >=)
#define CHECK_GT(LHS, RHS) CHECK_OP(LHS, RHS, >)

#define DCHECK(cond)        if constexpr (kEnableDChecks) CHECK(cond)
#define DCHECK_EQ(LHS, RHS) if constexpr (kEnableDChecks) CHECK_EQ(LHS, RHS)
#define DCHECK_NE(LHS, RHS) if constexpr (kEnableDChecks) CHECK_NE(LHS, RHS)
#define DCHECK_LT(LHS, RHS) if constexpr (kEnableDChecks) CHECK_LT(LHS, RHS)
#define DCHECK_LE(LHS, RHS) if constexpr (kEnableDChecks) CHECK_LE(LHS, RHS)
#define DCHECK_GE(LHS, RHS) if constexpr (kEnableDChecks) CHECK_GE(LHS, RHS)
#define DCHECK_GT(LHS, RHS) if constexpr (kEnableDChecks) CHECK_GT(LHS, RHS)


// Macro overloading. See https://stackoverflow.com/a/11763277
#define _GET_ARG2(_0, _1, _2, ...) _2

// Expands the following:
// - LOG(SEVERITY,TAG) to LOG_WITH_TAG(SEVERITY,TAG)
// - LOG(SEVERITY)     to LOG_DEFAULT_TAG(SEVERITY)
#define LOG(...) _GET_ARG2(__VA_ARGS__, LOG_WITH_TAG, LOG_DEFAULT_TAG)(__VA_ARGS__)

// Logging with provided tag
#define LOG_WITH_TAG(SEVERITY, TAG) \
    if (runtimeShouldLog(LogSeverity::SEVERITY)) \
        StreamLogger(LogSeverity::SEVERITY, TAG, __FILE__, __LINE__).stream()

// Logging using default tag
#define LOG_DEFAULT_TAG(SEVERITY) LOG_WITH_TAG(SEVERITY, __LLM_SDK_LOGTAG__)


// Support vector in ostream
template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    if (vec.empty()) {
        stream << "{}";
        return stream;
    }
    auto iter = vec.cbegin();
    auto insertElem = [&]() {
        if constexpr (std::is_convertible_v<T, std::string> ||
                      std::is_convertible_v<T, std::string_view>)
            stream << '"' << *iter++ << '"';
        else
            stream << *iter++;
    };
    stream << "{";
    insertElem();
    while (iter != vec.cend()) {
        stream << ", ";
        insertElem();
    }
    stream << "}";
    return stream;
}