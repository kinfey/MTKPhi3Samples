#include "llm_llama.h"
#include "medusa_config.h"
#include "utils/utils.h"
#include "common/timer.h"
#include "common/dump.h"
#include "common/logging.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"

#include <string>
#include <vector>
#include <math.h>
#include <numeric>
#include <functional>

#include <sstream>
#include <iostream>
#include <filesystem>

#include <thread>

namespace fs = std::filesystem;

using TokenType = Tokenizer::TokenType;
using TokenizerUPtr = std::unique_ptr<Tokenizer>;

LlamaModelOptions llamaModelOpt;
LlamaRuntimeOptions llamaRuntimeOpt;

size_t inferenceStep = 0; // Global counter

TokenizerUPtr prepare_tokenizer() {
    auto tokenizer = TokenizerFactory().create(llamaRuntimeOpt.tokenizerPath,
                                               llamaRuntimeOpt.tokenizerRegex);
    const auto& specialTokens = llamaRuntimeOpt.specialTokens;
    if (specialTokens.addBos)
        tokenizer->enableBosToken(specialTokens.bosId);
    return tokenizer;
}

std::tuple<std::string, std::vector<TokenType>>
get_prompt_and_tokens(const std::string& inputString, const TokenizerUPtr& tokenizer,
                      const bool parsePromptTokens) {
    // Parse or tokenize input
    auto inputTokens = parsePromptTokens ? parseTokenString(inputString)
                                         : tokenizer->tokenize(inputString);

    const auto& inputPrompt = parsePromptTokens ? tokenizer->detokenize(inputTokens)
                                                : inputString;
    return {inputPrompt, inputTokens};
}

void llm_llama_init(void** llamaRuntime, const std::string& yamlConfigPath) {
    Timer timer;
    timer.start();
    LOG(INFO) << "Begin model init...";

    // Force reset config to default values
    llamaModelOpt = {};
    llamaRuntimeOpt = {};

    // Load yaml config
    parseLlamaConfigYaml(yamlConfigPath, llamaModelOpt, llamaRuntimeOpt);

    bool status = neuron_llama_init(llamaRuntime, llamaModelOpt, llamaRuntimeOpt);
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

// Return the first generated token id and the hidden state of the last token in the last decoder layer.
std::tuple<TokenType, void*> llm_llama_digest_prompt(void* llamaRuntime,
                                                     const TokenizerUPtr& tokenizer,
                                                     const std::vector<TokenType>& inputTokens,
                                                     size_t& numModelInputToken,
                                                     double& promptTokPerSec) {
    const auto logitsType = llamaModelOpt.modelOutputType;
    void* lastLogits;
    void* hiddenStates;
    const auto inpBeginIt = inputTokens.cbegin();
    const auto inputTokenCount = inputTokens.size();
    size_t inputTokenIndex = 0;

    const auto startTokenIndex = neuron_llama_get_token_index(llamaRuntime);

    // Warn cache overflow
    if (startTokenIndex + inputTokenCount > llamaModelOpt.cacheSize) {
        LOG(WARN) << "Input prompt length (" << inputTokenCount << ") is longer than the available "
                  << "context length (cur token index = " << startTokenIndex << ", cache size = "
                  << llamaModelOpt.cacheSize << "). Cache will be overflowed.";
    }

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
        SET_DUMP_INDEX(inferenceStep++);
        LOG(DEBUG) << "Token position: " << inputTokenIndex << ": " << inputTokens[inputTokenIndex];

        const auto curInputTokens = getNewTokens();
        const auto numNewTok = curInputTokens.size();
        DUMP(INPUTS).fromVector("input_tokens", curInputTokens);
        DUMP(INPUTS).fromString("input_string", tokenizer->detokenize(curInputTokens));
        std::tie(lastLogits, hiddenStates) = neuron_llama_inference_once_return_hidden(
            llamaRuntime, curInputTokens
        );

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
    auto outputToken = argmaxFrom16bitLogits(logitsType, lastLogits, tokenizer->vocabSize());

    const size_t hiddenStatesSize = neuron_llama_get_per_token_hidden_states_size(llamaRuntime);
    const auto lastHiddenState = reinterpret_cast<char*>(hiddenStates)
                               + hiddenStatesSize * (numModelInputToken - 1);

    return {outputToken, lastHiddenState};
}

// Return tree candidates
std::vector<TokenType>
llm_medusa_gen_tree_candidates(void* llamaRuntime, const size_t vocabSize,
                               const TokenType acceptedToken, void* hiddenState,
                               const std::vector<std::vector<size_t>> medusaHeadTopK,
                               double& medusaHeadElapsed, double& topKElapsed) {
    Timer timerMedusaHead, timerTopK;

    std::vector<TokenType> treeCandidates;
    treeCandidates.push_back(acceptedToken);
    std::vector<size_t> topkPerHead;

    timerMedusaHead.start();
    auto medusaLogits = neuron_medusa_heads_inference_once(llamaRuntime, hiddenState);
    medusaHeadElapsed = timerMedusaHead.elapsed() * 1000;

    timerTopK.start();
    for (size_t headId = 0; headId < medusaHeadTopK.size(); headId++) {
        for (const auto topK : medusaHeadTopK[headId]) {
            if (llamaModelOpt.modelOutputType != LLMType::INT16) {
                LOG(FATAL) << "Medusa currently only supports INT16 activation datatype.";
            }
            topkPerHead = getTopkArgmaxV2(
                llamaModelOpt.modelOutputType,
                reinterpret_cast<const int16_t*>(medusaLogits) + headId * vocabSize,
                vocabSize,
                topK
            );
            treeCandidates.insert(treeCandidates.end(), topkPerHead.begin(), topkPerHead.end());
        }
    }
    topKElapsed = timerTopK.elapsed() * 1000;

    return treeCandidates;
}

// Return the accepted indices of the "tree_candidates", the length will be the generated length.
std::vector<size_t>
llm_medusa_verify(const int16_t* logits, std::vector<TokenType> treeCandidates,
                  const double modelOutputQuantScale, const size_t vocabSize,
                  const std::vector<int> parentId,
                  const std::vector<std::vector<int>> retrieveIndices,
                  double& dequantLogitElapsed, double& softmaxElapsed, double& verifyingElapsed,
                  const float temperature = 0, const float posteriorThreshold = 0.3,
                  const float posteriorAlpha = 0.09) {
    Timer timerDequantLogits, timerSoftmax, timerVerifying;
    dequantLogitElapsed = 0;
    softmaxElapsed = 0;

    // previousParentId should be initialized as a value which doesn't exist in parentId to make
    // the following for-loops work as expected.
    // In the current design, we use `-1` because a valid parent id is >= 0.
    int previousParentId = -1;

    std::vector<size_t> acceptedLengths;
    std::vector<size_t> acceptedIndices{0};  // The first token in `treeCandidates` is a must-accept token.

    if (temperature == 0) {
        timerVerifying.start();
        // Find the tokens that have the maximum logit for each position specified by `parentId`.
        std::vector<TokenType> goldenTokens;
        TokenType goldenToken;
        for (const auto tokenId : parentId) {
            if (tokenId != previousParentId) {
                goldenToken = argmaxFrom16bitLogits(llamaModelOpt.modelOutputType,
                                                    logits + tokenId * vocabSize, vocabSize);
                previousParentId = tokenId;
            }
            goldenTokens.push_back(goldenToken);
        }

        // Calculate the accepted length of each candidate
        for (const auto& candidate : retrieveIndices) {
            size_t acceptedLength = 0;
            for (const auto tokenId : candidate) {
                if (tokenId == -1) {
                    // `-1` is to align with the current design of `retrieveIndices` defined in
                    // `medusa_config.h`, which means no further draft token in this draft sequence.
                    break;
                }
                if (treeCandidates[tokenId] == goldenTokens[tokenId - 1]) {
                    // "tokenId - 1" because starting index mismatch btw goldenTokens and retrieveIndices.
                    acceptedLength += 1;
                } else {
                    break;
                }
            }
            acceptedLengths.push_back(acceptedLength);
        }

        size_t bestCandidate = std::distance(
            acceptedLengths.begin(),
            std::max_element(acceptedLengths.begin(), acceptedLengths.end())
        );
        size_t acceptedLength = acceptedLengths[bestCandidate];
        for (size_t i = 0; i < acceptedLength; i++) {
            acceptedIndices.push_back(retrieveIndices[bestCandidate][i]);
        }
        verifyingElapsed = timerVerifying.elapsed() * 1000;
    } else {
        // Calculate the threshold based on the entropy of posterior probability.
        std::vector<float> posteriorProb, candidateProb, thresholds;
        float threshold;
        const double scale = modelOutputQuantScale / temperature;
        size_t treeCandidateId = 1;  // The first token in `treeCandidates` is a must-accept token.
        for (const auto tokenId : parentId) {
            if (tokenId != previousParentId) {
                timerDequantLogits.start();
                std::vector<float> posteriorProbTemp(logits + tokenId * vocabSize,
                                                     logits + tokenId * vocabSize + vocabSize);
                std::transform(posteriorProbTemp.begin(),
                               posteriorProbTemp.end(),
                               posteriorProbTemp.begin(),
                               [&scale](float element) { return element *= scale; });
                dequantLogitElapsed += timerDequantLogits.elapsed() * 1000;
                timerSoftmax.start();
                const auto maxProb = *std::max_element(posteriorProbTemp.begin(),
                                                       posteriorProbTemp.end());
                convertToSoftmax(posteriorProbTemp.data(), vocabSize, maxProb, 1);
                softmaxElapsed += timerSoftmax.elapsed() * 1000;
                posteriorProb = posteriorProbTemp;

                float posteriorEntropy = 0;
                for (const auto prob : posteriorProb) {
                    if (prob > 0)
                        posteriorEntropy -= prob * log(prob);
                }
                threshold = std::min(posteriorThreshold, posteriorAlpha * exp(-posteriorEntropy));

                previousParentId = tokenId;
            }
            candidateProb.push_back(posteriorProb[treeCandidates[treeCandidateId]]);
            thresholds.push_back(threshold);
            treeCandidateId++;
        }

        // Calculate the accepted length of each candidate
        timerVerifying.start();
        for (const auto& candidate : retrieveIndices) {
            size_t acceptedLength = 0;
            for (const auto tokenId : candidate) {
                if (tokenId == -1 || candidateProb[tokenId - 1] <= thresholds[tokenId - 1])
                    break;
                acceptedLength += 1;
            }
            acceptedLengths.push_back(acceptedLength);
        }

        auto maxAcceptedLength = *std::max_element(acceptedLengths.begin(), acceptedLengths.end());
        // If there are more than one candidate sequences have the max accepted length, select the
        // one with maximum likelihood.
        if (maxAcceptedLength > 0) {
            std::vector<size_t> bestCandidates;
            for (size_t candidateId = 0; candidateId < retrieveIndices.size(); candidateId++) {
                if (acceptedLengths[candidateId] == maxAcceptedLength) {
                    bestCandidates.push_back(candidateId);
                }
            }

            std::vector<double> likelihoods;
            for (const auto candidateId : bestCandidates) {
                double likelihood = 0;
                for (size_t retrieveId = 0; retrieveId < maxAcceptedLength; retrieveId++) {
                    likelihood += log(candidateProb[retrieveIndices[candidateId][retrieveId] - 1]);
                    // `-1` because index mismatch btw candidateProb and retrieveIndices.
                }
                likelihoods.push_back(likelihood);
            }
            const auto maxLikelihoodIdx = std::distance(
                likelihoods.begin(), std::max_element(likelihoods.begin(), likelihoods.end()));
            size_t bestCandidate = bestCandidates[maxLikelihoodIdx];

            for (size_t i = 0; i < maxAcceptedLength; i++) {
                acceptedIndices.push_back(retrieveIndices[bestCandidate][i]);
            }
        }
        verifyingElapsed = timerVerifying.elapsed() * 1000;
    }
    return acceptedIndices;
}

std::vector<TokenType> llm_medusa_gen_response(void* llamaRuntime, const TokenizerUPtr& tokenizer,
                                               const size_t maxResponse,
                                               const TokenType firstInputToken,
                                               void* lastHiddenState, const float temperature,
                                               std::string& fullResponse, double& genTokPerSec) {
    const MedusaConfig* medusaConfig = getMedusaConfig(llamaModelOpt.numMedusaHeads,
                                                       llamaModelOpt.genTokenBatchSize);
    const auto logitsType = llamaModelOpt.modelOutputType;
    const size_t hiddenStatesSize = neuron_llama_get_per_token_hidden_states_size(llamaRuntime);
    const size_t logitsSize = neuron_llama_get_per_token_logits_size(llamaRuntime);

    const size_t maxTokenLength = llamaModelOpt.maxTokenLength;
    auto curTokenIndex = neuron_llama_get_token_index(llamaRuntime);
    const auto& sequenceLength = curTokenIndex; // The number of tokens the model has seen.

    // DEBUG message
    std::vector<size_t> acceptedLengths;
    double treeCandiTotalLatency = 0, medusaHeadTotalLatency = 0, topKTotalLatency = 0, baseModelTotalLatency = 0;
    double verifyTotalLatency = 0, retrieveTotalLatency = 0, rollbackTotalLatency = 0;
    double dequantLogitsTotalLatency = 0, softmaxTotalLatency = 0, verifyingTotalLatency = 0;
    Timer timerTreeCandi, timerBaseModel, timerVerify, timerRetrieve, timerRollback;

    double elapsed = 0, genTotalTime = 0;
    genTokPerSec = 0;
    size_t genTokCount = 0;
    size_t genInferenceStep = 0;

    std::string response;
    UTF8CharResolver utf8Resolver;
    TokenType outputToken = firstInputToken;

    std::vector<TokenType> generatedTokens = {firstInputToken};

    auto isStopToken = [](const auto token) {
        const auto& stopTokenSet = llamaRuntimeOpt.specialTokens.stopToken;
        return stopTokenSet.find(token) != stopTokenSet.end();
    };

    Timer timer;
    timer.start();
    neuron_llama_set_medusa_tree_attn(llamaRuntime, medusaConfig->mask, medusaConfig->positions);
    while (genTokCount < maxResponse && sequenceLength < maxTokenLength) {
        SET_DUMP_INDEX(inferenceStep++);

        // Warn cache overflow
        if (sequenceLength >= llamaModelOpt.cacheSize) {
            LOG(WARN) << "The max context length (" << llamaModelOpt.cacheSize << ") has already "
                         "been reached, about to overflow the cache.";
        }

        // Medusa heads generate draft tokens and prepare `tree_candidates`.
        double medusaHeadElapsed, topKElapsed;
        timerTreeCandi.start();
        auto treeCandidates = llm_medusa_gen_tree_candidates(
            llamaRuntime,
            tokenizer->vocabSize(),
            outputToken,
            lastHiddenState,
            medusaConfig->medusaHeadTopK,
            medusaHeadElapsed,
            topKElapsed
        );
        treeCandiTotalLatency += timerTreeCandi.elapsed() * 1000;

        DCHECK_EQ(treeCandidates.size(), llamaModelOpt.genTokenBatchSize)
            << "Mismatch between Medusa Tree Config and genTokenBatchSize.";

        // Base model (chunks + lm-head) run on the `tree_candidates`.
        // Return (1) the logits and (2) all hidden states of the last decoder layer.
        timerBaseModel.start();
        auto [logits, hiddenStates] =
            neuron_llama_inference_once_return_hidden(llamaRuntime, treeCandidates, false);
        auto baseModelElapsed = timerBaseModel.elapsed() * 1000;

        // Verify based on logits and tree_candidates.
        double dequantLogitElapsed, softmaxElapsed, verifyingElapsed;
        timerVerify.start();
        auto acceptedIndices = llm_medusa_verify(
            reinterpret_cast<const int16_t*>(logits), treeCandidates,
            llamaModelOpt.modelOutputQuantScale, tokenizer->vocabSize(), medusaConfig->parentId,
            medusaConfig->retrieveIndices, dequantLogitElapsed, softmaxElapsed, verifyingElapsed,
            temperature
        );
        auto verifyLatency = timerVerify.elapsed() * 1000;

        // Retrieve the hidden state for 1-t medusa head with the index `acceptedIndices.back()`.
        timerRetrieve.start();
        lastHiddenState = reinterpret_cast<char*>(hiddenStates)
                        + hiddenStatesSize * acceptedIndices.back();
        // Retrieve the logits which is used to prepare the 1st token in the "tree candidates".
        const auto lastLogits = reinterpret_cast<char*>(logits) + logitsSize * acceptedIndices.back();
        outputToken = argmaxFrom16bitLogits(logitsType, lastLogits, tokenizer->vocabSize());
        auto retrieveLatency = timerRetrieve.elapsed() * 1000;

        // Cache manipulation and handle the offset of ring buffer.
        timerRollback.start();
        neuron_medusa_rollback(llamaRuntime, acceptedIndices);
        auto rollbackLatency = timerRollback.elapsed() * 1000;

        // Save all accepted tokens in this decoding step.
        std::vector<TokenType> acceptedTokens;
        for (const auto tokenId : acceptedIndices) {
            acceptedTokens.push_back(treeCandidates[tokenId]);
        }

        genTokCount += acceptedTokens.size();
        curTokenIndex += acceptedTokens.size();
        genInferenceStep++;

        elapsed = timer.reset();
        genTotalTime += elapsed;
        LOG(DEBUG) << "Single loop time taken: " << elapsed * 1000 << " ms";

        // Print all accepted tokens in this decoding step.
        for (const auto acceptedToken : acceptedTokens) {
            generatedTokens.push_back(acceptedToken);
            if (isStopToken(acceptedToken)) {
                outputToken = acceptedToken;
                break;
            }
            const std::string tokStr = tokenizer->detokenize(acceptedToken);

            LOG(DEBUG) << "[Gen Inference Step " << genInferenceStep
                       << "] Output token " << acceptedToken
                       << ": \"" << tokStr << "\"";

            const bool is_tok_str_resolved = utf8Resolver.addBytes(tokStr);
            if (is_tok_str_resolved) {
                response = utf8Resolver.getResolvedStr();
                std::cout << response << std::flush;
                fullResponse += response;
            }
            DUMP(RESPONSE).fromValue("sampled_token", acceptedToken);
            DUMP(RESPONSE).fromString("sampled_text", tokStr);
            DUMP(RESPONSE).fromString("full_response", fullResponse);
        }

        // Performance information.
        acceptedLengths.push_back(acceptedTokens.size());
        medusaHeadTotalLatency += medusaHeadElapsed;
        topKTotalLatency += topKElapsed;
        baseModelTotalLatency += baseModelElapsed;
        verifyTotalLatency += verifyLatency;
        dequantLogitsTotalLatency += dequantLogitElapsed;
        softmaxTotalLatency += softmaxElapsed;
        verifyingTotalLatency += verifyingElapsed;
        retrieveTotalLatency += retrieveLatency;
        rollbackTotalLatency += rollbackLatency;

        acceptedTokens.clear();

        // Stop when output is EOS
        if (isStopToken(outputToken)) {
            std::cout << "</eos>";
            break;
        }
    }
    std::cout << "</end>" << std::endl;
    genTokPerSec = double(genTokCount) / genTotalTime;
    std::cout << "\n[Full Response]\n" << fullResponse << std::endl;

    // DEBUG message
    LOG(INFO) << "\n================ Accepted Lengths ====================";
    double avgAcceptanceLength = static_cast<double>(
        std::reduce(acceptedLengths.begin(), acceptedLengths.end())) / acceptedLengths.size();
    std::cout << "Average: " << avgAcceptanceLength << std::endl;

    LOG(INFO) << "\n================ Latency Breakdown ===================";
    LOG(INFO) << "Gen Total Time:             " << 1000 * genTotalTime / genInferenceStep;
    LOG(INFO) << "Generate Tree Candidates:   " << treeCandiTotalLatency / genInferenceStep;
    LOG(INFO) << "   Medusa Heads:            " << medusaHeadTotalLatency / genInferenceStep;
    LOG(INFO) << "   TopK from Medusa Heads:  " << topKTotalLatency / genInferenceStep;
    LOG(INFO) << "Base Model (+ LM Head):     " << baseModelTotalLatency / genInferenceStep;
    LOG(INFO) << "Verify:                     " << verifyTotalLatency / genInferenceStep;
    LOG(INFO) << "   DequantLogits:           " << dequantLogitsTotalLatency / genInferenceStep;
    LOG(INFO) << "   Softmax:                 " << softmaxTotalLatency / genInferenceStep;
    LOG(INFO) << "   Verifying:               " << verifyingTotalLatency / genInferenceStep;
    LOG(INFO) << "Retrieve (logits & hidden): " << retrieveTotalLatency / genInferenceStep;
    LOG(INFO) << "Rollback Cache:             " << rollbackTotalLatency / genInferenceStep;
    std::cout << std::endl;

    return generatedTokens;
}

std::tuple<double, double>
llm_llama_inference(void* llamaRuntime, const std::string& inputString,
                    const size_t maxResponse = 50, const float temperature = 0,
                    const bool parsePromptTokens = false) {
    // Prepare tokenizer
    const auto& tokenizer = prepare_tokenizer();

    LOG(INFO) << "Vocab size: " << tokenizer->vocabSize();

    // Convert string to tokens
    auto [inputPrompt, inputTokens] =
        get_prompt_and_tokens(inputString, tokenizer, parsePromptTokens);
    DUMP(PROMPT).fromVector("prompt_tokens", inputTokens);
    DUMP(PROMPT).fromString("prompt_text", inputPrompt);

    std::cout << "\n[Prompt]\n" << inputPrompt << '\n' << std::endl;

    // Input prompt caching
    size_t numModelInputToken = llamaModelOpt.promptTokenBatchSize;
    double promptTokPerSec;
    // Get both token and the last hidden states.
    // The last hidden states will be used as the input of Medusa Heads.
    const auto [outputToken, lastHiddenState] =
        llm_llama_digest_prompt(llamaRuntime, tokenizer, inputTokens, numModelInputToken,
                                promptTokPerSec);
    // Swap to gen mode model
    const size_t genTokenBatchSize = llamaModelOpt.genTokenBatchSize;
    if (numModelInputToken > genTokenBatchSize) {
        llm_llama_swap_model(llamaRuntime, genTokenBatchSize);
        numModelInputToken = genTokenBatchSize;
    }

    std::string fullResponse;

    // Generation process
    std::cout << "\nResponse [Max Length = " << maxResponse << "]:" << std::endl;
    std::string tokStr = tokenizer->detokenize(outputToken);
    LOG(DEBUG) << "First output token " << outputToken << ": \"" << tokStr << "\"";

    double genTokPerSec;
    const auto outputTokens = llm_medusa_gen_response(llamaRuntime, tokenizer, maxResponse,
                                                      outputToken, lastHiddenState, temperature,
                                                      fullResponse, genTokPerSec);

    // Show the output tokens if the input is also tokens
    if (parsePromptTokens) {
        std::cout << "\nGenerated Tokens: " << outputTokens << std::endl;
    }
    std::cout << "\n[Latency]" << std::endl;
    std::cout << "      Prompt Mode: " << promptTokPerSec << " tok/s" << std::endl;
    std::cout << "  Generative Mode: " << genTokPerSec << " tok/s" << std::endl;
    return {promptTokPerSec, genTokPerSec};
}

void llm_llama_reset(void* llamaRuntime) {
    neuron_llama_reset(llamaRuntime);
}

void llm_llama_release(void* llamaRuntime) {
    neuron_llama_release(llamaRuntime);
}

int main(int argc, char* argv[]) {
    std::vector<std::string> yamlConfigPaths;
    size_t maxResponse = 200;
    float temperature = 0;
    bool parsePromptTokens = false; // Read prompt as a string of tokens
    bool onePromptPerLine = false; // Treat each line in prompt text as a single prompt. Will replace literal "\n" with new line char '\n'.
    std::string preformatterName = "";
    std::vector<std::string> promptPaths; // Paths containing the prompt text
    std::vector<std::string> prompts;
    // std::string prompt = "Once upon a time,";
    const std::string defaultPrompt = "Tell me about alpacas";

    // Process command line.
    //  -m or --max to set the max response.
    //  -p or --prompt to set the input prompt.
    //  -i or --input-file to set the path to the text containing the input prompt.
    //  --read-tokens to read the input prompt as a string of tokens.
    //  --one-prompt-per-line to treat each line in prompt file as one prompt. The literal "\n" is treated as new line.
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
        } else if (fs::path(curArg).extension() == ".yaml") {
            LOG(INFO) << "Using yaml config file: " << curArg;
            yamlConfigPaths.push_back(curArg);
        } else if (matchArgument(curArg, "--read-tokens", "-t")) {
            parsePromptTokens = true;
        } else if (matchArgument(curArg, "--one-prompt-per-line")) {
            onePromptPerLine = true;
        } else if (matchArgument(curArg, "--preformatter")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            preformatterName = argv[++i];
        } else if (matchArgument(curArg, "--temperature")) {
            ENSURE_NEXT_ARG_EXISTS(i)
            temperature = std::atof(argv[++i]);
            LOG(INFO) << "Temperature setting: " << temperature;
        } else {
            LOG(INFO) << "Unrecognized argument: " << curArg;
        }
    }

    prompts = readPromptFiles(promptPaths, onePromptPerLine);

    if (prompts.empty())
        prompts.push_back(defaultPrompt);  // Use the default example.

    if (yamlConfigPaths.empty()) {
        LOG(ERROR) << "No yaml config file provided.";
    }

    const size_t numPrompt = prompts.size();
    void* llamaRuntime;
    for (const auto& yamlConfigPath : yamlConfigPaths) {
        double allPromptTokPerSec = 0, allGenTokPerSec = 0;
        std::cout << "\n>>>>>>>>>>> Current yaml config: " << yamlConfigPath << " <<<<<<<<<<<"
                  << std::endl;
        llm_llama_init(&llamaRuntime, yamlConfigPath);
        for (size_t i = 0; i < numPrompt; i++) {
            std::cout << "=========== Processing the " << i << "-th input. ==========="
                      << std::endl;
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
            auto [promptTokPerSec, genTokPerSec] =
                llm_llama_inference(llamaRuntime, prompt, maxResponse, temperature,
                                    parsePromptTokens);
            allPromptTokPerSec += promptTokPerSec;
            allGenTokPerSec += genTokPerSec;
            llm_llama_reset(llamaRuntime);
            llm_llama_swap_model(llamaRuntime, llamaModelOpt.promptTokenBatchSize);

            LOG(INFO) << "Phone is sleeping now ... (5 seconds)";
            std::this_thread::sleep_for(std::chrono::seconds(5));
        }
        llm_llama_release(llamaRuntime);
        std::cout << "\n[Average Performance among the given " << numPrompt << " prompts]\n";
        std::cout << "      Prompt Mode: " << allPromptTokPerSec / numPrompt << " tok/s\n";
        std::cout << "  Generative Mode: " << allGenTokPerSec / numPrompt << " tok/s\n";
    }
}