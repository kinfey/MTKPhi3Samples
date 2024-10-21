#pragma once

#include "tokenizer/tokenizer.h"

#include "llm_types.h"
#include "llm_llama.h"
#include <string>
#include <sstream>
#include <queue>
#include <iostream>

using TokenType = Tokenizer::TokenType;
using ArgmaxProb = std::pair<TokenType, float>;

#define ENSURE_NEXT_ARG_EXISTS(curArgIdx) \
    if (curArgIdx + 1 >= argc) { \
        std::cout << "No value provided for for argument '" << argv[curArgIdx] << "'." << std::endl; \
        continue; \
    }

bool matchArgument(const std::string& target, const std::string& argPattern,
                   const std::string& argPatternShort="", bool normalizeUnderscore = true);

bool isWhiteLine(const std::string& line);

class UTF8CharResolver {
public:
    UTF8CharResolver() {}
    bool addBytes(const std::string& byteStr);
    bool hasResolved();
    std::string getResolvedStr();
    static size_t utf8Len(char src);
    static size_t getUTF8FullLength(const std::string& str, size_t start_index = 0);
private:
    void setResolved() {
        mResolved = mAccum;
        mAccum.clear();
    }
    void setResolvedPartial(size_t unresolvedSize) {
        size_t resolved_size = mAccum.size() - unresolvedSize;
        mResolved = mAccum.substr(0, resolved_size);
        mAccum = mAccum.substr(resolved_size);
    }
private:
    size_t mUtfLengthRemaining = 0;
    bool mConcatMultibyteMode = false;
    std::string mAccum;
    std::string mResolved;
};

// Logits processors
float* repeatPenalty(float* logits, TokenType input_token, const float repetition_penalty = 1.2);

int16_t* suppressLogits(int16_t* logits, const std::vector<TokenType> tokenIds);
float* suppressLogits(float* logits, const std::vector<TokenType> tokenIds);

size_t samplingFromDistribution(const std::vector<float>& probs);

template <typename LogitsType>
std::pair<TokenType, LogitsType> argmaxWithMax(const LogitsType* array);

void convertToSoftmax(float* array, const size_t vocabSize, const float max,
                      const float temperature);
TokenType argmaxFrom16bitLogits(const LLMType logitsType, const void* logits, const size_t vocabSize);
TokenType argmaxFrom16bitLogits(const int16_t* logits, const size_t vocabSize);
TokenType argmaxFrom16bitLogits(const __fp16* logits, const size_t vocabSize);
ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale);
ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale);
ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize);
ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature);
ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature);
ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float tempertature);
ArgmaxProb argmaxProbFrom16bitLogits(const LLMType logitsType, const void* logits,
                                     const size_t vocabSize, const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb);
ArgmaxProb argmaxProbFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                     const float modelOutputQuantScale,
                                     const TokenType tokenIdForProb);
ArgmaxProb argmaxProbFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                     const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const LLMType logitsType, const void* logits,
                                       const size_t vocabSize, const float modelOutputQuantScale,
                                       const float temperature, const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const int16_t* logits, const size_t vocabSize,
                                       const float modelOutputQuantScale, const float temperature,
                                       const TokenType tokenIdForProb);
ArgmaxProb randomSampleFrom16bitLogits(const __fp16* logits, const size_t vocabSize,
                                       const float temperature, const TokenType tokenIdForProb);
TokenType argmaxFromAdjustDistSpecDec(const LLMType LogitsType, const void* targetLogits,
                                      const void* draftLogits, const size_t vocabSize,
                                      const float targetOutputQuantScale,
                                      const float draftOutputQuantScale);
template <typename LogitsType>
TokenType argmaxFromAdjustDistSpecDec(const LogitsType* targetLogits, const LogitsType* draftLogits,
                                      const size_t vocabSize, const float targetOutputQuantScale,
                                      const float draftOutputQuantScale);
TokenType randomSampleFromAdjustDistSpecDec(const LLMType LogitsType, const void* targetLogits,
                                            const void* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature);
template <typename LogitsType>
TokenType randomSampleFromAdjustDistSpecDec(const LogitsType* targetLogits,
                                            const LogitsType* draftLogits, const size_t vocabSize,
                                            const float targetOutputQuantScale,
                                            const float draftOutputQuantScale,
                                            const float targetSamplingTemperature,
                                            const float draftSamplingTemperature);

std::vector<size_t> getTopkArgmaxV2(const LLMType logitsType, const int16_t* logits,
                                    const size_t vocabSize, const size_t k);
template <typename LogitsType>
std::vector<size_t> getTopkArgmax(const LogitsType* logits, const size_t vocabSize, const size_t k);

// Preformatters
bool addPreformatter(const std::string& prefName, std::string& prompt);
std::string addPreformatter_AlpacaNoInput(const std::string& prompt);
std::string addPreformatter_OneShotConversation(const std::string& prompt);
std::string addPreformatter_VicunaNoInput(const std::string& prompt);
std::string addPreformatter_QwenNoInput(const std::string& prompt);
std::string addPreformatter_Llama3NoInput(const std::string& prompt);
std::string addPreformatter_Phi3NoInput(const std::string& prompt);

std::vector<std::string> split(const std::string& str, const std::string& sep);

std::vector<TokenType> parseTokenString(const std::string& tokenString);

std::vector<std::string> readPromptFiles(const std::vector<std::string>& promptPaths,
                                         const bool onePromptPerLine);

void parseLlamaConfigYaml(const std::string configYamlPath, LlamaModelOptions& modelOptions,
                          LlamaRuntimeOptions& runtimeOptions);