#include "utils.h"
#include "common/logging.h"
#include "third_party/include/yaml-cpp/yaml.h"

#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <unordered_set>
#include <cctype>
#include <regex>
#include <sstream>
#include <random>
#include <filesystem>

namespace fs = std::filesystem;

// Buffers used by `argmaxFrom16bitLogits`
static std::vector<int> argmaxBufferInt;
static std::vector<float> argmaxBufferFloat;
static std::vector<float> argmaxBufferFloatForDraft;
static const size_t randomSeed = 12345678;

bool matchArgument(const std::string& target, const std::string& argPattern,
                   const std::string& argPatternShort, bool normalizeUnderscore) {
    // Replace '_' with '-'
    auto normalize = [&](const std::string& arg) {
        return normalizeUnderscore ? std::regex_replace(arg, std::regex("_"), "-") : arg;
    };
    auto getRegexPattern = [&]() {
        const auto argPatternNorm = normalize(argPattern);
        const auto argPatternShortNorm = normalize(argPatternShort);
        if (argPatternShort.size() > 0) {
            std::ostringstream patStream;
            patStream << "(" << argPatternNorm << ")|(" << argPatternShortNorm << ")";
            return std::regex(patStream.str());
        } else {
            return std::regex(argPatternNorm);
        }
    };
    return std::regex_match(normalize(target), getRegexPattern());
}

bool isWhiteLine(const std::string& line) {
    return (line.size() == 1 && (line[0] == '\n' || line[0] == '\r'));
}

bool UTF8CharResolver::hasResolved() {
    return mResolved.size() > 0;
}

std::string UTF8CharResolver::getResolvedStr() {
    if (mConcatMultibyteMode)
        return "";
    return mResolved;
}

size_t UTF8CharResolver::utf8Len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

size_t UTF8CharResolver::getUTF8FullLength(const std::string& str, size_t startIndex) {
    size_t strLength = str.size();
    size_t curIdx = startIndex;
    size_t totalLength = 0;
    while (curIdx < strLength) {
        size_t curUtf8Length = utf8Len(str[curIdx]);
        curIdx += curUtf8Length;
        totalLength += curUtf8Length;
    }
    return totalLength;
}

bool UTF8CharResolver::addBytes(const std::string& byteStr) {
    size_t curStrLength = byteStr.size();
    LOG(DEBUG) << "UTF8: mConcatMultibyteMode=" << mConcatMultibyteMode << ", mUtfLengthRemaining="
               << mUtfLengthRemaining << ", mAccum.size()=" << mAccum.size()
               << ", curStrLength=" << curStrLength;
    if (!mConcatMultibyteMode) {
        mAccum = byteStr;
        LOG(DEBUG) << "UTF8: Concat mode false, check multibyte";
        mUtfLengthRemaining = getUTF8FullLength(byteStr);

        if (mUtfLengthRemaining > curStrLength) {
            // Start concat mode
            LOG(DEBUG) << "UTF8: Set concat mode true";
            mConcatMultibyteMode = true;
            mUtfLengthRemaining -= curStrLength;
        } else if (mUtfLengthRemaining < curStrLength) {
            LOG(FATAL) << "UTF8: Unreachable case: mUtfLengthRemaining < curStrLength ("
                       << mUtfLengthRemaining << " < " << curStrLength << ")";
        } else {
            if (curStrLength == 0) { // bos or eos, just update
                LOG(DEBUG) << "UTF8: bos/eos, just update";
            } else if (mUtfLengthRemaining == curStrLength) { // Is sub-byte
                LOG(DEBUG) << "UTF8: token == byte, update";
            }
            mUtfLengthRemaining = 0;
            setResolved();
            return true;
        }
    } else {
        mAccum += byteStr;
        LOG(DEBUG) << "UTF8: Concat mode true. Concat response: " << mAccum;

        if (mUtfLengthRemaining == curStrLength) {
            // Exit concat
            LOG(DEBUG) << "UTF8: Concat done, update. Set concat mode to false";
            mUtfLengthRemaining = 0;
            mConcatMultibyteMode = false;
            setResolved();
            return true;
        } else if (mUtfLengthRemaining < curStrLength) {
            // The current byteStr overshoots what was required to form a usable utf8 word
            setResolvedPartial(curStrLength - mUtfLengthRemaining); // Extract the resolved part from `mAccum` to `resolved`
            mUtfLengthRemaining = getUTF8FullLength(byteStr, mUtfLengthRemaining);
            return true; // next round will continue in concat mode
        } else {
            // continue to concat
            mUtfLengthRemaining -= curStrLength;
            LOG(DEBUG) << "UTF8: utf_len=" << mUtfLengthRemaining << ", curStrLength="
                       << curStrLength;
        }
    }
    mResolved.clear();
    return false;
}

float* repeatPenalty(float* logits, TokenType tokenId, const float repetition_penalty) {
    if (tokenId < 0)
        return logits;

    float score = logits[tokenId];
    if (score < 0)
        score *= repetition_penalty;
    else
        score /= repetition_penalty;
    logits[tokenId] = score;

    return logits;
}

int16_t* suppressLogits(int16_t* logits, const std::vector<TokenType>& tokenIds) {
    for (const auto tokenId : tokenIds) {
        if (tokenId < 0)
            continue;
        logits[tokenId] = INT16_MIN;
    }
    return logits;
}

float* suppressLogits(float* logits, const std::vector<TokenType>& tokenIds) {
    for (const auto tokenId : tokenIds) {
        if (tokenId < 0)
            continue;
        logits[tokenId] = -FLT_MAX;
    }
    return logits;
}

size_t samplingFromDistribution(const std::vector<float>& probs) {
    std::random_device device;
    std::mt19937 engine(device());
    engine.seed(randomSeed);
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    const size_t sampledIndex = dist(engine);
    return sampledIndex;
}

template <typename LogitsType>
std::pair<TokenType, LogitsType> argmaxWithMax(const LogitsType* array, const size_t vocabSize) {
    // Find argmax and corresponding value
    LogitsType max = array[0];
    size_t index = 0;
    for (size_t i = 1; i < vocabSize; i++) {
        const auto current = array[i];
        if (current > max) {
            index = i;
            max = current;
        }
    }
    return {index, max};
}

TokenType argmaxFrom16bitLogits(const LLMType logitsType, const void* logits,
                                const size_t vocabSize) {
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize);
        case LLMType::FP16:
            return argmaxFrom16bitLogits(reinterpret_cast<const __fp16*>(logits), vocabSize);
        default:
            LOG(ERROR) << "argmaxFrom16bitLogits function only supports INT16 and FP16 logits.";
            return 0;
    }
}

TokenType argmaxFrom16bitLogits(const int16_t* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferInt.size() < vocabSize) {
        argmaxBufferInt.resize(vocabSize);
    }
    int* tmp = argmaxBufferInt.data();
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    return argmaxWithMax(tmp, vocabSize).first;
}

TokenType argmaxFrom16bitLogits(const __fp16* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* tmp = argmaxBufferFloat.data();
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    return argmaxWithMax(tmp, vocabSize).first;
}

void convertToSoftmax(float* array, const size_t vocabSize, const float max,
                      const float temperature = 0.0) {
    if (temperature == 0) {
        const auto top1Token = argmaxWithMax(array, vocabSize).first;
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] = 0;
        }
        array[top1Token] = 1;
    } else {
        // Apply softmax on array
        const float lowerBound = 1e-8;
        const float temperatureForSoftmax = std::max(lowerBound, temperature);
        float total = 0;
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] = expf((array[j] - max) / temperatureForSoftmax);
            total += array[j];
        }
        for (size_t j = 0; j < vocabSize; j++) {
            array[j] /= total;
        }
    }
};

ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale) {
    // This function returns the highest probability and the corresponding index.
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const int16_t*>(logits),
                                             vocabSize, modelOutputQuantScale);
        case LLMType::FP16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const __fp16*>(logits),
                                             vocabSize);
        default:
            LOG(ERROR) << "argmaxProbFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferInt.size() < vocabSize) {
        argmaxBufferInt.resize(vocabSize);
    }
    int* tmp = argmaxBufferInt.data();
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf((tmp[j] - max) * modelOutputQuantScale);
    }
    const float prob = 1 / total;
    return {index, prob};
}

ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* tmp = argmaxBufferFloat.data();
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf(tmp[j] - max);
    }
    const float prob = 1 / total;
    return {index, prob};
}

ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature) {
    // This function returns the index that has highest probability and its probability.
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                               modelOutputQuantScale, temperature);
        case LLMType::FP16:
            return randomSampleFrom16bitLogits(reinterpret_cast<const __fp16*>(logits), vocabSize,
                                               temperature);
        default:
            LOG(ERROR) << "randomSampleFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale,
                                       const float temperature) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (temperature == 0) {
        if (argmaxBufferInt.size() < vocabSize) {
            argmaxBufferInt.resize(vocabSize);
        }
        int* tmp = argmaxBufferInt.data();
        for (size_t i = 0; i < vocabSize; i++) {
            tmp[i] = logits[i];
        }
        return {argmaxWithMax(tmp, vocabSize).first, 1};
    }
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* distWithTemperature = argmaxBufferFloat.data();
    for (size_t j = 0; j < vocabSize; j++) {
        distWithTemperature[j] = logits[j] * modelOutputQuantScale;
    }
    const auto [index, max] = argmaxWithMax(distWithTemperature, vocabSize);

    convertToSoftmax(distWithTemperature, vocabSize, max, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(argmaxBufferFloat);
    return {sampledToken, distWithTemperature[sampledToken]};
}

ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* distWithTemperature = argmaxBufferFloat.data();
    for (size_t i = 0; i < vocabSize; i++) {
        distWithTemperature[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(distWithTemperature, vocabSize);

    if (temperature == 0) {
        return {index, 1};
    }

    convertToSoftmax(distWithTemperature, vocabSize, max, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(argmaxBufferFloat);
    return {sampledToken, distWithTemperature[sampledToken]};
}

ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb) {
    // This function returns the index that has highest probability and the probability of a given token 'tokenIdForProb'.
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const int16_t*>(logits),
                                             vocabSize, modelOutputQuantScale, tokenIdForProb);
        case LLMType::FP16:
            return argmaxProbFrom16bitLogits(reinterpret_cast<const __fp16*>(logits),
                                             vocabSize, tokenIdForProb);
        default:
            LOG(ERROR) << "argmaxProbFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferInt.size() < vocabSize) {
        argmaxBufferInt.resize(vocabSize);
    }
    int* tmp = argmaxBufferInt.data();
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf((tmp[j] - max) * modelOutputQuantScale);
    }
    const float prob = expf((tmp[tokenIdForProb] - max) * modelOutputQuantScale) / total;
    return {index, prob};
}

ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                     const TokenType tokenIdForProb) {
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* tmp = argmaxBufferFloat.data();
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(tmp, vocabSize);
    for (size_t j = 0; j < vocabSize; j++) {
        total += expf(tmp[j] - max);
    }
    const float prob = expf(tmp[tokenIdForProb] - max) / total;
    return {index, prob};
}

ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature, const TokenType tokenIdForProb) {
    // This function returns the index that has highest probability and the probability of a given token 'tokenIdForProb'.
    switch (logitsType) {
        case LLMType::INT16:
            return
                randomSampleFrom16bitLogits(reinterpret_cast<const int16_t*>(logits), vocabSize,
                                            modelOutputQuantScale, temperature, tokenIdForProb);
        case LLMType::FP16:
            return
                randomSampleFrom16bitLogits(reinterpret_cast<const __fp16*>(logits), vocabSize,
                                            temperature, tokenIdForProb);
        default:
            LOG(ERROR) << "randomSampleFrom16bitLogits function only supports INT16/FP16 logits.";
            return {0, 0};
    }
}

ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature,
                                       const TokenType tokenIdForProb) {
    if (temperature == 0) {
        const auto top1Token = argmaxWithMax(logits, vocabSize).first;
        return {top1Token, (float)(top1Token == tokenIdForProb)};
    }
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* distWithTemperature = argmaxBufferFloat.data();
    for (size_t j = 0; j < vocabSize; j++) {
        distWithTemperature[j] = logits[j] * modelOutputQuantScale;
    }
    const auto [index, max] = argmaxWithMax(distWithTemperature, vocabSize);

    convertToSoftmax(distWithTemperature, vocabSize, max, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(argmaxBufferFloat);
    return {sampledToken, distWithTemperature[tokenIdForProb]};
}

ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature, const TokenType tokenIdForProb) {
    if (temperature == 0) {
        const auto top1Token = argmaxWithMax(logits, vocabSize).first;
        return {top1Token, (float)(top1Token == tokenIdForProb)};
    }
    // Comparisons is significantly faster in 32-bit,
    // so we store a temp 32-bit casted buffer and do comparisons in 32-bit.
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    float* distWithTemperature = argmaxBufferFloat.data();
    for (size_t i = 0; i < vocabSize; i++) {
        distWithTemperature[i] = logits[i];
    }
    const auto [index, max] = argmaxWithMax(distWithTemperature, vocabSize);

    convertToSoftmax(distWithTemperature, vocabSize, max, temperature);

    // temperature sampling
    const size_t sampledToken = samplingFromDistribution(argmaxBufferFloat);
    return {sampledToken, distWithTemperature[tokenIdForProb]};
}

TokenType argmaxFromAdjustDistSpecDec(const LLMType logitsType, const void* targetLogits,
                                      const void* draftLogits, const size_t vocabSize,
                                      const float targetOutputQuantScale,
                                      const float draftOutputQuantScale) {
    switch (logitsType) {
        case LLMType::INT16:
            return argmaxFromAdjustDistSpecDec(reinterpret_cast<const int16_t*>(targetLogits),
                                               reinterpret_cast<const int16_t*>(draftLogits),
                                               vocabSize, targetOutputQuantScale,
                                               draftOutputQuantScale);
        case LLMType::FP16:
            return argmaxFromAdjustDistSpecDec(reinterpret_cast<const __fp16*>(targetLogits),
                                               reinterpret_cast<const __fp16*>(draftLogits),
                                               vocabSize, targetOutputQuantScale,
                                               draftOutputQuantScale);
        default:
            LOG(ERROR) << "argmaxFromAdjustDistSpecDec function only supports INT16/FP16 logits.";
            return 0;
    }
}

template <typename LogitsType>
TokenType argmaxFromAdjustDistSpecDec(const LogitsType* targetLogits, const LogitsType* draftLogits,
                                      const size_t vocabSize, const float targetOutputQuantScale,
                                      const float draftOutputQuantScale) {
    // Initialization
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    if (argmaxBufferFloatForDraft.size() < vocabSize) {
        argmaxBufferFloatForDraft.resize(vocabSize);
    }
    float* targetTmp = argmaxBufferFloat.data();
    float* draftTmp = argmaxBufferFloatForDraft.data();
    // Dequantize target and draft logits
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] = targetLogits[i] * targetOutputQuantScale;
        draftTmp[i] = draftLogits[i] * draftOutputQuantScale;
    }
    // Large model
    auto max = argmaxWithMax(targetTmp, vocabSize).second;
    convertToSoftmax(targetTmp, vocabSize, max);
    // Small model
    max = argmaxWithMax(draftTmp, vocabSize).second;
    convertToSoftmax(draftTmp, vocabSize, max);
    // Adjust the distribution
    // p'(x) = max(p(x)-q(x), 0)
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] = std::max(targetTmp[i] - draftTmp[i], 0.0f);
        total += targetTmp[i];
    }
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] /= total;
    }
    // argmax(p'(x))
    return argmaxWithMax(targetTmp, vocabSize).first;
}

TokenType randomSampleFromAdjustDistSpecDec(const LLMType logitsType, const void* targetLogits,
                                            const void* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature) {
    switch (logitsType) {
        case LLMType::INT16:
            return randomSampleFromAdjustDistSpecDec(
                reinterpret_cast<const int16_t*>(targetLogits),
                reinterpret_cast<const int16_t*>(draftLogits), vocabSize, targetOutputQuantScale,
                draftOutputQuantScale, targetSamplingTemperature, draftSamplingTemperature);
        case LLMType::FP16:
            return randomSampleFromAdjustDistSpecDec(
                reinterpret_cast<const __fp16*>(targetLogits),
                reinterpret_cast<const __fp16*>(draftLogits), vocabSize, targetOutputQuantScale,
                draftOutputQuantScale, targetSamplingTemperature, draftSamplingTemperature);
        default:
            LOG(ERROR)
                << "randomSampleFromAdjustDistSpecDec function only supports INT16/FP16 logits.";
            return 0;
    }
}

template <typename LogitsType>
TokenType randomSampleFromAdjustDistSpecDec(const LogitsType* targetLogits,
                                            const LogitsType* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature) {
    // Initialization
    if (argmaxBufferFloat.size() < vocabSize) {
        argmaxBufferFloat.resize(vocabSize);
    }
    if (argmaxBufferFloatForDraft.size() < vocabSize) {
        argmaxBufferFloatForDraft.resize(vocabSize);
    }
    float* targetTmp = argmaxBufferFloat.data();
    float* draftTmp = argmaxBufferFloatForDraft.data();
    // Dequantize target and draft logits
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] = targetLogits[i] * targetOutputQuantScale;
        draftTmp[i] = draftLogits[i] * draftOutputQuantScale;
    }
    // Large model
    auto max = argmaxWithMax(targetTmp, vocabSize).second;
    convertToSoftmax(targetTmp, vocabSize, max, targetSamplingTemperature);
    // Small model
    max = argmaxWithMax(draftTmp, vocabSize).second;
    convertToSoftmax(draftTmp, vocabSize, max, draftSamplingTemperature);
    // Adjust the distribution
    // p'(x) = max(p(x)-q(x), 0)
    float total = 0;
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] = std::max(targetTmp[i] - draftTmp[i], 0.0f);
        total += targetTmp[i];
    }
    for (size_t i = 0; i < vocabSize; i++) {
        targetTmp[i] /= total;
    }
    // sample from (p'(x))
    const size_t sampledToken = samplingFromDistribution(argmaxBufferFloat);
    return sampledToken;
}

std::vector<size_t> getTopkArgmaxV2(const LLMType logitsType, const int16_t* logits,
                                    const size_t vocabSize, const size_t k) {
    if (logitsType != LLMType::INT16) {
        LOG(FATAL) << "`getTopkArgmaxV2` currently only supports INT16.";
    }
    if (argmaxBufferInt.size() < vocabSize) {
        argmaxBufferInt.resize(vocabSize);
    }
    int* tmp = argmaxBufferInt.data();
    for (size_t i = 0; i < vocabSize; i++) {
        tmp[i] = logits[i];
    }

    std::vector<size_t> result(k);
    int topi;
    for (size_t i = 0; i < k; i++) {
        if (i > 0) {
            tmp[topi] = -32768;
        }
        topi = argmaxWithMax(tmp, vocabSize).first;
        result[i] = topi;
    }
    return result;
}

template <typename LogitsType>
std::vector<TokenType> getTopkArgmax(const LogitsType* logits, const size_t vocabSize,
                                     const size_t k) {
    using ValIdxPair = std::pair<LogitsType, size_t>;
    std::priority_queue<ValIdxPair,
                        std::vector<ValIdxPair>,
                        std::greater<ValIdxPair>> q;
    for (size_t i = 0; i < vocabSize; ++i) {
        if(q.size() < k)
            q.push(ValIdxPair(logits[i], i));
        else if(q.top().first < logits[i]) {
            q.pop();
            q.push(ValIdxPair(logits[i], i));
        }
    }
    std::vector<TokenType> result(k);
    for (size_t i = 0; i < k; ++i) {
        result[k - i - 1] = static_cast<TokenType>(q.top().second);
        q.pop();
    }
    return result;
}

// Returns true if preformatter has been successfully added, false if otherwise.
bool addPreformatter(const std::string& prefName, std::string& prompt) {
    if (prefName.empty())
        return false;

    #define DISPATCH(NAME)                      \
    if (prefName == #NAME) {                    \
        prompt = addPreformatter_##NAME(prompt);\
        return true;                            \
    }
    DISPATCH(AlpacaNoInput)
    DISPATCH(OneShotConversation)
    DISPATCH(VicunaNoInput)
    DISPATCH(QwenNoInput)
    DISPATCH(Llama3NoInput)
    DISPATCH(Phi3NoInput)
    #undef DISPATCH
    return false;
}

std::string addPreformatter_AlpacaNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n"
       << prompt << "\n\n### Response:\n";
    return ss.str();
}

std::string addPreformatter_OneShotConversation(const std::string& prompt) {
    std::stringstream ss;
    ss << "A chat between a curious human and an artificial intelligence assistant. "
       << "The assistant gives helpful, detailed, and polite answers to the human's questions.\n"
       << "### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: "
       << "Of course! Here are some creative ideas for a 10-year-old's birthday party:\n"
       << "1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n"
       << "2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n"
       << "3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n"
       << "Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: "
       << prompt << "\n### Assistant:";
    return ss.str();
}

std::string addPreformatter_VicunaNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "A chat between a curious user and an artificial intelligence assistant. "
       << "The assistant gives helpful, detailed, and polite answers to the user's questions. USER: "
       << prompt << " ASSISTANT:";
    return ss.str();
}

std::string addPreformatter_QwenNoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
       << prompt << "<|im_end|>\n<|im_start|>assistant\n";
    return ss.str();
}

std::string addPreformatter_Llama3NoInput(const std::string& prompt) {
    std::stringstream ss;
    ss << "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
       << prompt << "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    return ss.str();
}

std::string addPreformatter_Phi3NoInput(const std::string& prompt){
    std::stringstream ss;
    ss << "<|system|>\nYou are a helpful AI assistant. Please provide safe, ethical and accurate information to the user.\n<|user|>\n "
       << prompt
       << " \n <|assistant|>";
    return ss.str();
}

std::vector<std::string> split(const std::string& str, const std::string& sep) {
    std::vector<std::string> substrings;
    const std::regex noSepPattern("([^" + sep + "]+)");
    std::smatch match;
    auto cur = str.cbegin();
    while (std::regex_search(cur, str.cend(), match, noSepPattern)) {
        substrings.push_back(match[0].str());
        cur = match.suffix().first;
    }
    return substrings;
}

// Parses non-digit separated integers (the separator can be anything other than numbers)
std::vector<TokenType> parseTokenString(const std::string& tokenString) {
    std::vector<TokenType> tokens;
    const std::regex tokenPattern("([0-9-]+)"); // Matches a signed integer
    std::smatch match;
    auto cur = tokenString.cbegin();
    while (std::regex_search(cur, tokenString.cend(), match, tokenPattern)) {
        tokens.push_back(std::stoi(match[0].str()));
        cur = match.suffix().first;
    }
    return tokens;
}

std::vector<std::string> readPromptFiles(const std::vector<std::string>& promptPaths,
                                         const bool onePromptPerLine) {
    std::vector<std::string> prompts;

    // Replace "\\n" with '\n'
    auto replaceNewLines = [&](std::string& str) -> decltype(auto) {
        constexpr std::string_view literalNewLine = "\\n";
        constexpr std::string_view realNewLine = "\n";
        size_t start_pos = 0;
        while ((start_pos = str.find(literalNewLine, start_pos)) != std::string::npos) {
            str.replace(start_pos, literalNewLine.size(), realNewLine);
            start_pos += realNewLine.size();
        }
        return str;
    };

    auto readLines = [&](std::ifstream& fin) {
        std::string prompt;
        while (std::getline(fin, prompt) && !prompt.empty()) {
            if (isWhiteLine(prompt))
                continue;
            prompts.emplace_back(std::move(replaceNewLines(prompt)));
        }
    };

    auto readFile = [&](std::ifstream& fin) {
        std::stringstream buffer;
        buffer << fin.rdbuf();
        prompts.push_back(buffer.str());
    };

    for (const auto& path : promptPaths) {
        std::ifstream fin(path);
        if (!fin) {
            LOG(ERROR) << "Unable to open the prompt file: " << path;
            continue;
        } else {
            LOG(INFO) << "Reading prompt from file: " << path;
        }
        if (onePromptPerLine) {
            readLines(fin);
        } else {
            readFile(fin);
        }
    }
    return prompts;
}

void parseLlamaConfigYaml(const std::string configYamlPath, LlamaModelOptions& modelOptions,
                          LlamaRuntimeOptions& runtimeOptions) {
    const auto config = YAML::LoadFile(configYamlPath);
    const auto& modelOptYaml = config["llamaModelOptions"];
    const auto& runtimeOptYaml = config["llamaRuntimeOptions"];

    const auto& specialTokensYaml = runtimeOptYaml["specialTokens"];
    const auto& tokenizerRegexYaml = runtimeOptYaml["tokenizerRegex"];
    const auto& vocabPathYaml = runtimeOptYaml["vocabPath"]; // TODO: Deprecated
    const auto& tokenizerPathYaml = runtimeOptYaml["tokenizerPath"];
    const auto& tfliteEmbPathYaml = runtimeOptYaml["tfliteEmbPath"]; // TODO: Deprecated
    const auto& tokenEmbPathYaml = runtimeOptYaml["tokenEmbPath"];
    const auto& dlaPromptPathsYaml = runtimeOptYaml["dlaPromptPaths"];
    const auto& dlaGenPathsYaml = runtimeOptYaml["dlaGenPaths"];
    const auto& dlaLmHeadPathYaml = runtimeOptYaml["dlaLmHeadPath"];
    const auto& dlaMedusaHeadsPathYaml = runtimeOptYaml["dlaMedusaHeadsPath"];
    const auto& cachePathsYaml = runtimeOptYaml["cachePaths"];
    const auto& sharedWeightsPathsYaml = runtimeOptYaml["sharedWeightsPaths"];
    const auto& loraWeightsPathsYaml = runtimeOptYaml["loraWeightsPaths"];
    const auto& initWithLoraKeyYaml = runtimeOptYaml["initWithLoraKey"];
    const auto& loraInputCountYaml = runtimeOptYaml["loraInputCount"];
    const size_t numPromptDla = dlaPromptPathsYaml ? dlaPromptPathsYaml.size() : 0;
    const size_t numGenDla = dlaGenPathsYaml ? dlaGenPathsYaml.size() : 0;
    const size_t numCachePaths = cachePathsYaml ? cachePathsYaml.size() : 0;
    const size_t numSharedWeightsPaths = sharedWeightsPathsYaml ? sharedWeightsPathsYaml.size() : 0;

    // Error checking:
    //   - Both 'llamaModelOptions' and 'llamaRuntimeOptions' have to be defined.
    //   - 'tokenizerPath' has to be defined. Otherwise will look for 'vocabPath' (deprecated).
    //   - 'llamaRuntimeOptions.tokenEmbPath' has to be defined.
    //   - At least one of 'dlaPromptPathsYaml' and 'dlaGenPathsYaml' has to be defined.
    //   - The number of cache paths (if provided) must match the dla chunk count.
    //   - The number of paths per LoRA (if provided) must match the dla chunk count.
    if (!modelOptYaml || !runtimeOptYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'llamaModelOptions' or 'llamaRuntimeOptions'"
                   << " is not found in the config.";
    }
    if (!tokenizerPathYaml && !vocabPathYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'tokenizerPath' is not defined in the yaml "
                      "config.";
    }
    if (!tokenEmbPathYaml && !tfliteEmbPathYaml) {
        LOG(FATAL) << "Invalid yaml config file: 'tokenEmbPath' is not defined in the yaml config";
    }
    if (!numPromptDla && !numGenDla) {
        LOG(FATAL) << "Invalid yaml config file: At least one of 'dlaPromptPaths' and 'dlaGenPaths'"
                   << " is not defined in the yaml config.";
    }
    const size_t numDlaChunks = std::max(numPromptDla, numGenDla);
    if (numCachePaths > 0 && numCachePaths != numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: The number of provided cache paths ("
                   << numCachePaths << ") does not " << "match the number of dla chunks ("
                   << numDlaChunks << ").";
    }
    if (numSharedWeightsPaths > 0 && numSharedWeightsPaths != numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: The number of provided shared weights paths ("
                   << numSharedWeightsPaths << ") does not " << "match the number of dla chunks ("
                   << numDlaChunks << ").";
    }

    const size_t numLoraWeightsPaths = [&]() {
        std::unordered_set<size_t> numChunkSet;
        for (const auto& kv : loraWeightsPathsYaml) {
            const auto& paths = kv.second;
            numChunkSet.insert(paths.size());
        }
        CHECK_LE(numChunkSet.size(), 1)
            << "Invalid yaml config file: Inconsistent chunk size for LoRA weights paths.";
        return numChunkSet.empty() ? 0 : *numChunkSet.cbegin();
    }();
    if (numLoraWeightsPaths > 0 && numLoraWeightsPaths != numDlaChunks) {
        LOG(FATAL) << "Invalid yaml config file: The number of provided LoRA weights paths"
                   << "(" << numLoraWeightsPaths << ") does not " << "match the number of dla "
                   << "chunks " << numDlaChunks << ").";
    }

    // Parse llama model options
    #define PARSE_OPTION(type, key) \
    if (modelOptYaml[#key]) { modelOptions.key = modelOptYaml[#key].as<type>(); }
    PARSE_OPTION(size_t, promptTokenBatchSize)
    PARSE_OPTION(size_t, genTokenBatchSize)
    PARSE_OPTION(size_t, cacheSize)
    PARSE_OPTION(size_t, hiddenSize)
    PARSE_OPTION(size_t, numHead)
    PARSE_OPTION(size_t, numLayer)
    PARSE_OPTION(size_t, maxTokenLength)
    PARSE_OPTION(size_t, numMedusaHeads)
    PARSE_OPTION(float, rotEmbBase)
    PARSE_OPTION(float, embOutputQuantScale)
    PARSE_OPTION(float, modelOutputQuantScale)
    #undef PARSE_OPTION

    #define PARSE_OPTION_LLMTYPE(key) \
    if (modelOptYaml[#key]) { \
        modelOptions.key = getLLMTypeFromName(modelOptYaml[#key].as<std::string>().c_str()); \
    }
    PARSE_OPTION_LLMTYPE(modelInputType)
    PARSE_OPTION_LLMTYPE(modelOutputType)
    PARSE_OPTION_LLMTYPE(cacheType)
    PARSE_OPTION_LLMTYPE(maskType)
    PARSE_OPTION_LLMTYPE(rotEmbType)
    #undef PARSE_OPTION_LLMTYPE

    if (modelOptions.embOutputQuantScale != 0) {
        // TODO: Deprecated
        LOG(WARN) << "The use of 'embOutputQuantScale' is deprecated. Please ensure the token "
                     "embedding Lut value type matches with the model embedding input type.";
    }

    // Override `promptTokenBatchSize` to 1 if only gen mode dla path is provided
    if (!numPromptDla) {
        modelOptions.promptTokenBatchSize = 1;
    }

    const auto outputType = modelOptYaml["modelOutputType"].as<std::string>();
    const auto outputScale = modelOptions.modelOutputQuantScale;
    if (outputType == "FP16" && outputScale != 1.0) {
        modelOptions.modelOutputQuantScale = 1.0;
        LOG(WARN) << "Overriding scale to 1.0 for FP16 output.";
    }

    auto yamlSeqToStrVec = [](const YAML::Node& yamlSeqNode) -> std::vector<std::string> {
        const auto size = yamlSeqNode.size();
        std::vector<std::string> strVec(size);
        for (size_t i = 0; i < size; i++)
            strVec[i] = yamlSeqNode[i].template as<std::string>();
        return strVec;
    };

    auto parseScalarOrSeq = [&](const YAML::Node& node) -> std::vector<std::string> {
        if (node.IsSequence())
            return node.as<std::vector<std::string>>();
        return {node.as<std::string>()};
    };

    // Parse llama runtime options
    if (runtimeOptYaml["startTokenIndex"]) {
        runtimeOptions.startTokenIndex = runtimeOptYaml["startTokenIndex"].as<size_t>();
    }

    auto hasValue = [](const YAML::Node& node) { return node.IsDefined() && !node.IsNull(); };

    if (!tokenEmbPathYaml) {
        // TODO: Deprecated
        LOG(WARN) << "Th use of 'tfliteEmbPath' in YAML config is deprecated. "
                     "Please rename it to 'tokenEmbPath' instead.";
        runtimeOptions.tokenEmbPath = tfliteEmbPathYaml.as<std::string>();
        if (fs::path(runtimeOptions.tokenEmbPath).extension() == ".tflite") {
            LOG(ERROR) << "Token embedding file has '.tflite' extension. "
                          "Please note that '.tflite' embedding has been replaced with '.bin' "
                          "lookup table.";
        }
    } else {
        runtimeOptions.tokenEmbPath = tokenEmbPathYaml.as<std::string>();
    }

    if (!tokenizerPathYaml && vocabPathYaml) {
        // TODO: Deprecated
        LOG(WARN) << "Th use of 'vocabPath' in YAML config is deprecated. "
                     "Please use 'tokenizerPath' instead.";
        runtimeOptions.tokenizerPath = parseScalarOrSeq(vocabPathYaml);
    } else {
        runtimeOptions.tokenizerPath = parseScalarOrSeq(tokenizerPathYaml);
    }
    if (hasValue(tokenizerRegexYaml)) {
        runtimeOptions.tokenizerRegex = tokenizerRegexYaml.as<std::string>();
    }

    // Special tokens:
    //   - Both 'bosId' and 'eosId' have to be defined.
    //   - 'addBos' is default to false, unless set to true in config.
    //   - if 'stopToken' is not defined, then it is default to eosId.
    if (!hasValue(specialTokensYaml)) {
        LOG(FATAL) << "The runtime option 'specialTokens' is required.";
    }
    const auto& bosIdYaml = specialTokensYaml["bosId"];
    const auto& eosIdYaml = specialTokensYaml["eosId"];
    const auto& addBosYaml = specialTokensYaml["addBos"];
    const auto& stopTokenYaml = specialTokensYaml["stopToken"];
    if (!hasValue(bosIdYaml) || !hasValue(eosIdYaml)) {
        LOG(FATAL) << "Both 'bosId' & 'eosId' special tokens have to be defined in the config.";
    } else {
        auto& specialTokens = runtimeOptions.specialTokens;
        specialTokens.bosId = bosIdYaml.as<TokenType>();
        specialTokens.eosId = eosIdYaml.as<TokenType>();
        specialTokens.addBos = hasValue(addBosYaml) ? addBosYaml.as<bool>() : false;
        // Stop token set
        if (!hasValue(stopTokenYaml)) {
            specialTokens.stopToken = {specialTokens.eosId};
            LOG(DEBUG) << "The option 'stopToken' is not specified, defaulting to EoS token: "
                       << specialTokens.eosId;
        } else if (stopTokenYaml.IsSequence()) {
            const auto& stopTokenVec = stopTokenYaml.as<std::vector<TokenType>>();
            specialTokens.stopToken = {stopTokenVec.begin(), stopTokenVec.end()};
        }else {
            specialTokens.stopToken = {stopTokenYaml.as<TokenType>()};
        }
    }

    runtimeOptions.dlaPromptPaths.resize(numPromptDla);
    for (size_t i = 0; i < numPromptDla; i++) {
        runtimeOptions.dlaPromptPaths[i] = dlaPromptPathsYaml[i].as<std::string>();
    }

    runtimeOptions.dlaGenPaths.resize(numGenDla);
    for (size_t i = 0; i < numGenDla; i++) {
        runtimeOptions.dlaGenPaths[i] = dlaGenPathsYaml[i].as<std::string>();
    }

    if (dlaLmHeadPathYaml) {
        runtimeOptions.dlaLmHeadPath = dlaLmHeadPathYaml.as<std::string>();
    }

    if (dlaMedusaHeadsPathYaml) {
        runtimeOptions.dlaMedusaHeadsPath = dlaMedusaHeadsPathYaml.as<std::string>();
    }

    // Cache path
    runtimeOptions.cachePaths.resize(numCachePaths);
    for (size_t i = 0; i < numCachePaths; i++) {
        runtimeOptions.cachePaths[i] = cachePathsYaml[i].as<std::string>();
    }

    // Shared weights path
    runtimeOptions.sharedWeightsPaths.resize(numSharedWeightsPaths);
    for (size_t i = 0; i < numSharedWeightsPaths; i++) {
        runtimeOptions.sharedWeightsPaths[i] = sharedWeightsPathsYaml[i].as<std::string>();
    }

    // LoRA weights path
    for (const auto& kv : loraWeightsPathsYaml) {
        const auto& loraKey = kv.first.as<std::string>();
        const auto& weightsPaths = yamlSeqToStrVec(kv.second);
        if (weightsPaths.empty())
            continue;
        runtimeOptions.loraWeightsPaths.emplace(loraKey, weightsPaths);
    }

    // The default value of `runtimeOptions.initWithLoraKey` is an empty string.
    if (initWithLoraKeyYaml && !initWithLoraKeyYaml.IsNull()) {
        runtimeOptions.initWithLoraKey = initWithLoraKeyYaml.as<std::string>();
    }

    // The default value of `runtimeOptions.loraInputCount` is 0.
    if (loraInputCountYaml && !loraInputCountYaml.IsNull()) {
        runtimeOptions.loraInputCount = loraInputCountYaml.as<size_t>();
    }
}

// Explicit instantiation of getTopkArgmax for some logits types
template std::vector<TokenType> getTopkArgmax<int16_t>(const int16_t* logits, const size_t vocabSize,
                                                       const size_t k);
template std::vector<TokenType> getTopkArgmax<__fp16>(const __fp16* logits, const size_t vocabSize,
                                                      const size_t k);
template std::vector<TokenType> getTopkArgmax<float>(const float* logits, const size_t vocabSize,
                                                     const size_t k);
