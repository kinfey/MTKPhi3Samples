#include "llm_llama.h"
#include "utils/utils.h"
#include "common/timer.h"
#include "common/dump.h"
#include "common/logging.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"

#include <string>
#include <vector>
#include <numeric>
#include <random>

#include <sstream>
#include <iostream>
#include <filesystem>
#include <thread>

namespace fs = std::filesystem;

using TokenType = Tokenizer::TokenType;
using TokenizerUPtr = std::unique_ptr<Tokenizer>;

LlamaModelOptions llamaModelOpt;
LlamaModelOptions draftModelOpt;
LlamaRuntimeOptions llamaRuntimeOpt;
LlamaRuntimeOptions draftRuntimeOpt;

enum class SpecDecInferType : int {
    UnionMethodV1
};

struct SpecDecContext {
    void* targetRuntime;
    void* draftRuntime;
    size_t inferenceStep;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution;
    const size_t draftLength;
    const size_t maxResponse;
    const SpecDecInferType inferType;
    const TokenizerUPtr tokenizer;
    const float targetSamplingTemperature;
    const float draftSamplingTemperature;
};

static const size_t randomSeed = 20240402;

TokenizerUPtr prepare_tokenizer(const LlamaRuntimeOptions& runtimeOpt) {
    auto tokenizerInstance = TokenizerFactory().create(runtimeOpt.tokenizerPath,
                                                       runtimeOpt.tokenizerRegex);
    const auto& specialTokens = runtimeOpt.specialTokens;
    if (specialTokens.addBos)
        tokenizerInstance->enableBosToken(specialTokens.bosId);
    return tokenizerInstance;
}

bool isStopToken(const TokenType token) {
    const auto& stopTokenSet = llamaRuntimeOpt.specialTokens.stopToken;
    return stopTokenSet.find(token) != stopTokenSet.end();
};

std::tuple<std::string, std::vector<TokenType>>
get_prompt_and_tokens(const std::string& inputString, const TokenizerUPtr& tokenizer,
                      const bool parsePromptTokens) {
    // Parse or tokenize input
    const auto inputTokens = parsePromptTokens ? parseTokenString(inputString)
                                               : tokenizer->tokenize(inputString);

    const auto& inputPrompt = parsePromptTokens ? tokenizer->detokenize(inputTokens)
                                                : inputString;
    return {inputPrompt, inputTokens};
}

void llm_llama_init_spec_dec(void** targetRuntime, void** draftRuntime,
                             const std::string& yamlConfigPath,
                             const std::string& yamlConfigPathDraft) {
    Timer timer;
    timer.start();
    LOG(INFO) << "Begin target model init...";
    parseLlamaConfigYaml(yamlConfigPath, llamaModelOpt, llamaRuntimeOpt);
    bool status = neuron_llama_init(targetRuntime, llamaModelOpt, llamaRuntimeOpt);
    if (!status) {
        LOG(FATAL) << "LLM init failed";
    }

    LOG(INFO) << "Begin draft model init...";
    parseLlamaConfigYaml(yamlConfigPathDraft, draftModelOpt, draftRuntimeOpt);
    status = neuron_llama_init(draftRuntime, draftModelOpt, draftRuntimeOpt);
    if (!status) {
        LOG(FATAL) << "LLM init failed";
    }

    double elapsed = timer.reset();
    LOG(INFO) << "Done model init. (Time taken: " << elapsed << "s)";
}

void llm_llama_swap_model(void* llamaRuntime, const size_t batchSize = 1) {
    Timer timer;
    timer.start();
    LOG(INFO) << "Hot swapping to " << batchSize << "t model...";
    neuron_llama_swap_model(llamaRuntime, batchSize);
    double elapsed = timer.reset();
    LOG(INFO) << "Done model hot swapping. (Time taken: " << elapsed << "s)";
}

TokenType llm_llama_digest_prompt(SpecDecContext& ctx, const bool isTarget,
                                 const std::vector<TokenType>& inputTokens,
                                 size_t& numModelInputToken, double& promptTokPerSec) {
    const auto logitsType = llamaModelOpt.modelOutputType;
    void* lastLogits;
    const auto inpBeginIt = inputTokens.begin();
    const auto inputTokenCount = inputTokens.size();
    size_t inputTokenIndex = 0;

    void* llamaRuntime = isTarget ? ctx.targetRuntime : ctx.draftRuntime;

    auto getNewTokens = [&]() {
        // Calculate prompt tokens size for current step
        const size_t numInputTokenLeft = inputTokenCount - inputTokenIndex;
        const size_t remainder = numInputTokenLeft % numModelInputToken;
        // Construct subset prompt tokens
        const auto numNewTok = remainder ? remainder : numModelInputToken;
        const auto tokIdxStart = inputTokenIndex; // inclusive
        const auto tokIdxEnd = tokIdxStart + numNewTok; // exclusive
        const auto newTokens = std::vector(inpBeginIt + tokIdxStart, inpBeginIt + tokIdxEnd);
        LOG(DEBUG) << "Feeding model with prompt tokens [" << tokIdxStart << " - "
                   << tokIdxEnd << "] (numToken=" << numNewTok << "): " << newTokens;
        return newTokens;
    };

    Timer promptTimer;
    promptTimer.start();
    while (inputTokenIndex < inputTokenCount) {
        SET_DUMP_INDEX(ctx.inferenceStep++);
        LOG(DEBUG) << "Token position: " << inputTokenIndex << ": "
                   << inputTokens[inputTokenIndex];

        const auto curInputTokens = getNewTokens();
        const auto numNewTok = curInputTokens.size();
        DUMP(INPUTS).fromVector("input_tokens", curInputTokens);
        DUMP(INPUTS).fromString("input_string", ctx.tokenizer->detokenize(curInputTokens));
        lastLogits = neuron_llama_inference_once(llamaRuntime, curInputTokens);

        inputTokenIndex += numNewTok;
    }
    double promptTimeTaken = promptTimer.reset();

    // Ideal prompt size is a multiple of prompt batch size
    const size_t idealPromptSize = std::ceil(float(inputTokenCount) / numModelInputToken)
                                 * numModelInputToken;
    DCHECK_EQ(idealPromptSize % numModelInputToken, 0);
    promptTokPerSec = idealPromptSize / promptTimeTaken;

    LOG(INFO) << "Done analyzing prompt in " << promptTimeTaken << "s" << " (" << promptTokPerSec
              << " tok/s)";
    // Prompt mode ended, take the output and feed as input
    // Argmax to generate the token first
    const auto outputToken = argmaxFrom16bitLogits(logitsType, lastLogits,
                                                   ctx.tokenizer->vocabSize());
    return outputToken;
}

std::tuple<std::vector<TokenType>, TokenType>
llm_llama_spec_dec_per_step(SpecDecContext& ctx, const TokenType inputToken, size_t& acceptNum,
                            double& meanDraftElapsed, double& targetElapsed,
                            double& rollbackElapsed, double& verifyElapsed) {
    Timer timerDraft, timerTarget, timerVerify, timerRollback;
    rollbackElapsed = 0;

    const auto targetLogitsType = llamaModelOpt.modelOutputType;
    const auto draftLogitsType = draftModelOpt.modelOutputType;

    const float targetOutputQuantScale = llamaModelOpt.modelOutputQuantScale;
    const float draftOutputQuantScale = draftModelOpt.modelOutputQuantScale;
    const size_t genTokenBatchSize = llamaModelOpt.genTokenBatchSize;
    const float targetSamplingTemperature = ctx.targetSamplingTemperature;
    const float draftSamplingTemperature = ctx.draftSamplingTemperature;

    const auto vocabSize = ctx.tokenizer->vocabSize();

    TokenType outputToken = inputToken;

    double allDraftTimePerStep = 0;
    std::vector<TokenType> draftTokens, targetTokens, tokensToVerify;
    void* targetLogits;
    std::vector<float> draftProbs;
    std::vector<char*> allDraftLogits;
    std::vector<TokenType> acceptedTokens;

    const size_t draftLogitsSize = neuron_llama_get_per_token_logits_size(ctx.draftRuntime);

    double draftElapsed;

    LOG(DEBUG) << "[Spec-Dec]: The newest token (confirmedNewToken) is: " << inputToken;

    // Draft model generates `draftLength` draft tokens.
    for (size_t t = 0; t < ctx.draftLength; t++) {
        timerDraft.start();
        void* draftLastLogits = neuron_llama_inference_once(ctx.draftRuntime, {outputToken});
        char* draftLogitsCopy = new char[draftLogitsSize];
        std::memcpy(draftLogitsCopy, draftLastLogits, draftLogitsSize);
        allDraftLogits.push_back(draftLogitsCopy);

        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            const auto [updatedToken, tokenProb] =
                randomSampleFrom16bitLogits(draftLogitsType, draftLastLogits, vocabSize,
                                            draftOutputQuantScale, draftSamplingTemperature);

            outputToken = updatedToken;
            draftProbs.push_back(tokenProb);
        }
        draftElapsed = timerDraft.reset() * 1000;
        LOG(DEBUG) << "[Spec-Dec][Draft]: Generate the "<< t << "-th draft token. Time "
                   << "elapsed: " << draftElapsed;
        draftTokens.push_back(outputToken);
        allDraftTimePerStep += draftElapsed;
    }

    meanDraftElapsed = allDraftTimePerStep / ctx.draftLength;

    LOG(DEBUG) << "[Spec-Dec][Draft]: Complete the generation. Tokens:" << draftTokens;

    // Target model verifies the draft tokens.
    tokensToVerify.push_back(inputToken);
    tokensToVerify.insert(tokensToVerify.end(), draftTokens.begin(), draftTokens.end());
    DCHECK_EQ(tokensToVerify.size(), ctx.draftLength + 1);
    LOG(DEBUG) << "[Spec-Dec][Target] Input Tokens: " << tokensToVerify;

    timerTarget.start();
    std::vector<float> targetProbs;
    targetLogits = neuron_llama_inference_once(ctx.targetRuntime, tokensToVerify, false);

    targetElapsed = timerTarget.elapsed() * 1000;
    LOG(DEBUG) << "[Spec-Dec][Target]: Latency of Target(" << genTokenBatchSize << "-T): "
               << 1000 * targetElapsed << " ms.";
    const size_t logitsSize = neuron_llama_get_per_token_logits_size(ctx.targetRuntime);
    // gen target tokens and probs
    for (size_t t = 0; t < ctx.draftLength; t++) {
        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            const auto draftTokenId = draftTokens[t];
            const auto curTargetLogits = reinterpret_cast<char*>(targetLogits) + logitsSize * t;
            const auto [updatedToken, tokenProb] =
                randomSampleFrom16bitLogits(targetLogitsType, curTargetLogits, vocabSize,
                                            targetOutputQuantScale, targetSamplingTemperature,
                                            draftTokenId);

            outputToken = updatedToken;
            targetProbs.push_back(tokenProb);
        }
        targetTokens.push_back(outputToken);
    }
    LOG(DEBUG) << "[Spec-Dec][Target]: Latency for Target(" << genTokenBatchSize << "-T) + "
               << "argmax: " << 1000 * timerTarget.reset() << " ms.";
    LOG(DEBUG) << "[Spec-Dec][Target]: Target tokens:" << targetTokens;
    timerVerify.start();
    DCHECK_EQ(draftTokens.size(), targetTokens.size());
    acceptNum = 0;
    // Verification
    for (size_t t = 0; t < draftTokens.size(); t++) {
        bool acceptDraftToken = false;
        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            acceptDraftToken =
                ((draftTokens[t] == targetTokens[t]) ||
                 (ctx.distribution(ctx.generator) < targetProbs[t] / draftProbs[t]));
        }
        if (acceptDraftToken) {
            const auto acceptedToken = draftTokens[t];
            LOG(DEBUG) << "[Spec-Dec][Verifying] Accept the " << t+1 << "-th draft token";
            acceptNum++;
            if (acceptedToken == 2) {
                outputToken = acceptedToken;
                break;
            }
            acceptedTokens.push_back(acceptedToken);

            if (t == (ctx.draftLength - 1)) {  // All draft tokens are accepted.
                const auto curTargetLogits = reinterpret_cast<char*>(targetLogits)
                                             + logitsSize * ctx.draftLength;
                const auto [tmpToken, tmpTokenProb] =
                    randomSampleFrom16bitLogits(targetLogitsType, curTargetLogits, vocabSize,
                                                targetOutputQuantScale, targetSamplingTemperature);
                outputToken = tmpToken;
                neuron_llama_inference_once(ctx.draftRuntime, {acceptedToken});
            }
        } else {
            LOG(DEBUG) << "[Spec-Dec][Verifying] Reject " << t+1 << "-th draft token";
            if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
                const auto curTargetLogits = reinterpret_cast<char*>(targetLogits)
                                             + logitsSize * t;
                outputToken = randomSampleFromAdjustDistSpecDec(targetLogitsType, curTargetLogits,
                                                                allDraftLogits[t], vocabSize,
                                                                targetOutputQuantScale,
                                                                draftOutputQuantScale,
                                                                targetSamplingTemperature,
                                                                draftSamplingTemperature);
            }
            break;
        }
    }
    verifyElapsed = 1000 * timerVerify.reset();
    LOG(DEBUG) << "[Spec-Dec][Verifying]: Latency for verification: "
               << verifyElapsed << " ms.";
    LOG(DEBUG) << "[Spec-Dec][Verifying]: Accepted tokens: " << acceptNum;

    // Stop when output is a stop token (default to EoS if not set)
    if (isStopToken(outputToken)) {
        std::cout << "</eos>";
        return {acceptedTokens, outputToken};
    }
    // Manipulate the cache (rollback if necessary).
    if (acceptNum < ctx.draftLength) {
        timerRollback.start();
        neuron_llama_rollback(ctx.draftRuntime, ctx.draftLength - 1 - acceptNum);
        neuron_llama_rollback(ctx.targetRuntime, ctx.draftLength - acceptNum);
        rollbackElapsed = timerRollback.reset() * 1000;
        LOG(DEBUG) << "[Spec-Dec][Rollback]: Latency overhead: " << rollbackElapsed << " ms.";
    }
    for (auto draftLogits : allDraftLogits) {
        delete[] draftLogits;
    }
    return {acceptedTokens, outputToken};
}

void llm_llama_gen_response(SpecDecContext& ctx, const TokenType firstInputToken,
                            double& genTokPerSec) {
    const size_t maxTokenLength = llamaModelOpt.maxTokenLength;
    auto curTokenIndex = neuron_llama_get_token_index(ctx.targetRuntime);
    const size_t& sequenceLength = curTokenIndex;

    double elapsed = 0, genTotalTime = 0;
    genTokPerSec = 0;
    size_t genTokCount = 0, specDecCount = 0;
    TokenType totalAcceptNum = 0, allAcceptNum = 0;
    double totalDraftTime = 0, totalTargetTime = 0, totalRollbackTime = 0, totalVerifyTime = 0;

    std::string fullResponse;
    UTF8CharResolver utf8Resolver;
    TokenType outputToken = firstInputToken;

    Timer timer;
    timer.start();
    while (genTokCount < ctx.maxResponse && sequenceLength < maxTokenLength) {
        if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
            SET_DUMP_INDEX(ctx.inferenceStep++);
            // Save and print outputToken
            const std::string tokStr = ctx.tokenizer->detokenize(outputToken);
            const bool isTokStrResolved = utf8Resolver.addBytes(tokStr);
            if (isTokStrResolved) {
                const std::string response = utf8Resolver.getResolvedStr();
                std::cout << response << std::flush;
                fullResponse += response;
                DUMP(RESPONSE).fromValue("sampled_token", outputToken);
                DUMP(RESPONSE).fromString("sampled_text", tokStr);
                DUMP(RESPONSE).fromString("full_response", fullResponse);
            }
            size_t acceptNum;
            double meanDraftElapsed, targetElapsed, rollbackElapsed, verifyElapsed;
            auto [acceptedTokens, lastAcceptToken] =
                llm_llama_spec_dec_per_step(ctx, outputToken, acceptNum, meanDraftElapsed,
                                            targetElapsed, rollbackElapsed, verifyElapsed);
            outputToken = lastAcceptToken;

            // Save and print all accepted tokens in this decoding step.
            for (const auto acceptedToken : acceptedTokens) {
                const std::string tokStr = ctx.tokenizer->detokenize(acceptedToken);
                const bool isTokStrResolved = utf8Resolver.addBytes(tokStr);
                if (isTokStrResolved) {
                    const std::string response = utf8Resolver.getResolvedStr();
                    std::cout << response << std::flush;
                    fullResponse += response;
                    DUMP(RESPONSE).fromValue("sampled_token", acceptedToken);
                    DUMP(RESPONSE).fromString("sampled_text", tokStr);
                    DUMP(RESPONSE).fromString("full_response", fullResponse);
                }
            }
            specDecCount++;
            genTokCount += (acceptNum + 1);
            curTokenIndex += (acceptNum + 1);

            if (acceptNum == ctx.draftLength)
                allAcceptNum++;
            totalAcceptNum += acceptNum;
            totalDraftTime += meanDraftElapsed;
            totalTargetTime += targetElapsed;
            totalRollbackTime += rollbackElapsed;
            totalVerifyTime += verifyElapsed;

            elapsed = timer.reset();
            genTotalTime += elapsed;
            LOG(DEBUG) << "Single loop time taken: " << elapsed * 1000 << " ms";

            // Stop when output is a stop token (default to EoS if not set)
            if (isStopToken(outputToken)) {
                std::cout << "</eos>";
                break;
            }
        }
    }
    std::cout << "</end>" << std::endl;
    genTokPerSec = double(genTokCount) / genTotalTime;

    if (ctx.inferType == SpecDecInferType::UnionMethodV1) {
        std::cout << "\n[Full Response]\n" << fullResponse << std::endl;
        std::cout << "\n[Info]" << std::endl;
        std::cout << "        Avg. Acceptance: "
                  << double(totalAcceptNum) / (specDecCount * ctx.draftLength) << std::endl;
        std::cout << "        All-accept Rate: "
                  << double(allAcceptNum) / specDecCount << std::endl;
        std::cout << "       Draft 1t latency: "
                  << totalDraftTime / specDecCount << " ms" << std::endl;
        std::cout << "         Target latency: "
                  << totalTargetTime / specDecCount << " ms" << std::endl;
        std::cout << "   Verification latency: "
                  << totalVerifyTime / specDecCount << " ms" << std::endl;
        std::cout << "       Rollback latency: "
                  << totalRollbackTime / specDecCount << " ms" << std::endl;
    }
}

std::tuple<double, double>
llm_llama_inference_spec_dec(void* targetRuntime, void* draftRuntime,
                             const SpecDecInferType inferType, const size_t draftLength,
                             const std::string& inputString, const size_t maxResponse = 50,
                             const bool parsePromptTokens = false, const float upperBound = 1.0,
                             const float targetSamplingTemperature = 0.0,
                             const float draftSamplingTemperature = 0.0) {
    SpecDecContext ctx = {
        .targetRuntime = targetRuntime,
        .draftRuntime = draftRuntime,
        .generator = std::default_random_engine(randomSeed),
        .distribution = std::uniform_real_distribution<float>(0.0, upperBound),
        .draftLength = draftLength,
        .maxResponse = maxResponse,
        .inferType = inferType,
        .tokenizer = prepare_tokenizer(llamaRuntimeOpt),
        .targetSamplingTemperature = targetSamplingTemperature,
        .draftSamplingTemperature = draftSamplingTemperature
    };

    // Prepare tokenizers for both models
    const auto draftTokenizer = prepare_tokenizer(draftRuntimeOpt);

    const auto& tokenizer = ctx.tokenizer;

    CHECK_EQ(tokenizer->vocabSize(), draftTokenizer->vocabSize())
        << "Different vocab size for the target and the draft model.";

    // Convert string to tokens
    auto [draftInputPrompt, draftInputTokens] =
        get_prompt_and_tokens(inputString, draftTokenizer, parsePromptTokens);

    auto [inputPrompt, inputTokens] =
        get_prompt_and_tokens(inputString, tokenizer, parsePromptTokens);

    CHECK_EQ(inputPrompt, draftInputPrompt)
        << "target model and the draft model may be using different tokenizers!";
    CHECK_EQ(inputTokens, draftInputTokens)
        << "target model and the draft model may be using different tokenizers!";
    DUMP(PROMPT).fromVector("prompt_tokens", inputTokens);
    DUMP(PROMPT).fromString("prompt_text", inputPrompt);

    std::cout << "\n[Prompt]\n" << inputPrompt << '\n' << std::endl;

    // Draft Model: input prompt caching
    ctx.inferenceStep = 0;
    size_t draftNumModelInputToken = draftModelOpt.promptTokenBatchSize;
    double draftPromptTokPerSec;
    llm_llama_digest_prompt(ctx, /*isTarget*/ false, draftInputTokens, draftNumModelInputToken,
                            draftPromptTokPerSec);

    // Draft Model: Swap to gen mode if model is still in prompt mode.
    const size_t draftGenTokenBatchSize = draftModelOpt.genTokenBatchSize;
    if (draftNumModelInputToken > draftGenTokenBatchSize) {
        llm_llama_swap_model(ctx.draftRuntime, draftGenTokenBatchSize);
        draftNumModelInputToken = draftGenTokenBatchSize;
    }

    // Target Model: input prompt caching
    ctx.inferenceStep = 0;
    size_t numModelInputToken = llamaModelOpt.promptTokenBatchSize;
    double promptTokPerSec;
    const TokenType outputToken = llm_llama_digest_prompt(ctx, /*isTarget*/ true, inputTokens,
                                                          numModelInputToken, promptTokPerSec);
    // Target Model: Swap to gen mode if model is still in prompt mode.
    const size_t genTokenBatchSize = llamaModelOpt.genTokenBatchSize;
    CHECK_GT(genTokenBatchSize, ctx.draftLength)
        << "genTokenBatchSize in target model config should be larger than draftlen!";
    if (numModelInputToken > genTokenBatchSize) {
        llm_llama_swap_model(ctx.targetRuntime, genTokenBatchSize);
        numModelInputToken = genTokenBatchSize;
    }

    const double totalPromptTokPerSec = 1.0 / ((1.0 / promptTokPerSec)
                                               + (1.0 / draftPromptTokPerSec));

    // Generation process
    std::cout << "\nResponse [Max Length = " << ctx.maxResponse << "]:" << std::endl;
    double genTokPerSec;
    llm_llama_gen_response(ctx, outputToken, genTokPerSec);
    std::cout << "\n[Latency]" << std::endl;
    std::cout << "      Prompt Mode: " << totalPromptTokPerSec << " tok/s" << std::endl;
    std::cout << "  Generative Mode: " << genTokPerSec << " tok/s" << std::endl;
    return {totalPromptTokPerSec, genTokPerSec};
}

void llm_llama_reset(void* llamaRuntime) {
    neuron_llama_reset(llamaRuntime);
}

void llm_llama_release(void* llamaRuntime) {
    neuron_llama_release(llamaRuntime);
}

int main(int argc, char* argv[]) {
    std::string yamlConfigPath = "config.yaml";
    std::string yamlConfigPathDraft = "";
    SpecDecInferType inferType = SpecDecInferType::UnionMethodV1;
    size_t maxResponse = 200;
    bool parsePromptTokens = false; // Read prompt as a string of tokens
    bool onePromptPerLine = false; // Treat each line in prompt text as a single prompt. Will replace literal "\n" with new line char '\n'.
    std::string preformatterName = "";
    size_t draftLen = 0;
    float upperBound = 1.0; // Threshold ~ U[0, upperBound]
    std::vector<std::string> promptPaths; // Paths containing the prompt text
    std::vector<std::string> prompts;
    // std::string prompt = "Once upon a time,";
    const std::string defaultPrompt = "Tell me about alpacas";
    float targetSamplingTemperature = 0.0;
    float draftSamplingTemperature = 0.0;

    // Process command line.
    //  -m or --max to set the max response.
    //  -p or --prompt to set the input prompt.
    //  -i or --input-file to set the path to the text containing the input prompt.
    //  --read-tokens to read the input prompt as a string of tokens.
    //  --one-prompt-per-line to treat each line in prompt file as one prompt. The literal "\n" is treated as new line.
    //  --infer-type to set inference type of Spec-Dec.
    //  '-d or --draft' and '-r or --draft-len' should be initialized if --infer-type is 0.
    for (int i = 1; i < argc; i++) {
        std::string curArg(argv[i]);
        if (matchArgument(curArg, "--max", "-m")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            maxResponse = std::atoi(argv[++i]);
        } else if (matchArgument(curArg, "--prompt", "-p")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            prompts.emplace_back(argv[++i]);
        } else if (matchArgument(curArg, "--input-file", "-i")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            promptPaths.emplace_back(argv[++i]);
        } else if (matchArgument(curArg, "--infer-type")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            inferType = static_cast<SpecDecInferType>(std::atoi(argv[++i]));
        } else if (matchArgument(curArg, "--draft", "-d")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            yamlConfigPathDraft = argv[++i];
            LOG(INFO) << "Using yaml config file for draft model: " << yamlConfigPathDraft;
        } else if (matchArgument(curArg, "--draft-len", "-r")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            draftLen = std::atoi(argv[++i]);
            LOG(INFO) << "Draft length: " << draftLen;
        } else if (fs::path(curArg).extension() == ".yaml") {
            LOG(INFO) << "Using yaml config file: " << curArg;
            yamlConfigPath = curArg;
        } else if (matchArgument(curArg, "--read-tokens", "-t")) {
            parsePromptTokens = true;
        } else if (matchArgument(curArg, "--one-prompt-per-line")) {
            onePromptPerLine = true;
        } else if (matchArgument(curArg, "--preformatter")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            preformatterName = argv[++i];
        } else if (matchArgument(curArg, "--upper-bound")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            upperBound = std::atof(argv[++i]);
            LOG(INFO) << "Using upper bound: " << upperBound;
        } else if (matchArgument(curArg, "--target-temperature")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            targetSamplingTemperature = std::atof(argv[++i]);
            LOG(INFO) << "Using temperature for target model: " << targetSamplingTemperature;
            LOG(WARN) << "Remember to specify the modelOutputQuantScale in the target yaml file,"
                      << " or the results maybe incorrect in some cases! (e.g. 4w16a model)";
        } else if (matchArgument(curArg, "--draft-temperature")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            draftSamplingTemperature = std::atof(argv[++i]);
            LOG(INFO) << "Using temperature for draft model: " << draftSamplingTemperature;
            LOG(WARN) << "Remember to specify the modelOutputQuantScale in the draft yaml file,"
                      << " or the results maybe incorrect in some cases! (e.g. 4w16a model)";
        } else {
            LOG(INFO) << "Unrecognized argument: " << curArg;
        }
    }

    prompts = readPromptFiles(promptPaths, onePromptPerLine);

    if (prompts.empty())
        prompts.push_back(defaultPrompt);  // Use the default example.

    double allPromptTokPerSec = 0, allGenTokPerSec = 0;
    const size_t numPrompt = prompts.size();
    void* llamaRuntime = nullptr;
    void* draftLlamaRuntime = nullptr;
    switch(inferType) {
        case SpecDecInferType::UnionMethodV1:
            llm_llama_init_spec_dec(&llamaRuntime, &draftLlamaRuntime, yamlConfigPath,
                                    yamlConfigPathDraft);
            break;
        default:
            LOG(FATAL) << "Wrong INFER_METHOD initialized the bat file, or this main file doesn't "
                          "suppoort this method";
    }

    for (size_t i = 0; i < numPrompt; i++) {
        std::cout << "============ Processing the " << i << "-th input. ============" << std::endl;
        std::string prompt = prompts[i];
        DUMP(PROMPT).fromString("text", prompt);
        if (!parsePromptTokens && !preformatterName.empty()) {
            if(addPreformatter(preformatterName, prompt)) {
                LOG(INFO) << "Preformatted prompt with '" << preformatterName << "'";
                DUMP(PROMPT).fromString("text_preformatted", prompt);
            } else {
                LOG(ERROR) << "Invalid preformatter: '" << preformatterName << "'";
            }
        }
        switch (inferType) {
            case SpecDecInferType::UnionMethodV1: {
                LOG(INFO) << "Sanity check...";
                CHECK_GT(draftLen, 0) << "Need to specify draft_len in bat file.";
                CHECK(!yamlConfigPathDraft.empty())
                    << "Need to specify draft model (--draft) in bat file.";

                const auto [promptTokPerSec, genTokPerSec] =
                    llm_llama_inference_spec_dec(llamaRuntime, draftLlamaRuntime, inferType,
                                                 draftLen, prompt, maxResponse, parsePromptTokens,
                                                 upperBound, targetSamplingTemperature,
                                                 draftSamplingTemperature);
                allPromptTokPerSec += promptTokPerSec;
                allGenTokPerSec += genTokPerSec;
                llm_llama_reset(llamaRuntime);
                llm_llama_reset(draftLlamaRuntime);
                llm_llama_swap_model(llamaRuntime, llamaModelOpt.promptTokenBatchSize);
                llm_llama_swap_model(draftLlamaRuntime, draftModelOpt.promptTokenBatchSize);
                break;
            } default: {
                LOG(FATAL) << "Wrong INFER_METHOD initialized the bat file.";
            }
        }
        if ((i + 1) % 10 == 0) {
            LOG(INFO) << "Phone is sleeping now ... (5 seconds)";
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }

    }
    llm_llama_release(llamaRuntime);
    if (inferType == SpecDecInferType::UnionMethodV1) {
        llm_llama_release(draftLlamaRuntime);
    }
    std::cout << "\n[Average Performance among the given " << numPrompt << " prompts]\n";
    std::cout << "      Prompt Mode: " << allPromptTokPerSec / numPrompt << " tok/s\n";
    std::cout << "  Generative Mode: " << allGenTokPerSec / numPrompt << " tok/s\n";
}