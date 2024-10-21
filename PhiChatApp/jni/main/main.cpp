#include "llm_llama.h"
#include "utils/utils.h"
#include "common/timer.h"
#include "common/dump.h"
#include "common/logging.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/tokenizer_factory.h"

#include <string>
#include <vector>

#include <sstream>
#include <iostream>
#include <filesystem>

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

TokenType llm_llama_digest_prompt(void* llamaRuntime, const TokenizerUPtr& tokenizer,
                                  const std::vector<TokenType>& inputTokens,
                                  size_t& numModelInputToken, double& promptTokPerSec) {
    const auto logitsType = llamaModelOpt.modelOutputType;
    void* lastLogits;
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
    auto outputToken = argmaxFrom16bitLogits(logitsType, lastLogits, tokenizer->vocabSize());
    return outputToken;
}

TokenType llm_llama_autoregressive_per_step(void* llamaRuntime, const TokenizerUPtr& tokenizer,
                                            const TokenType inputToken) {
    const auto logitsType = llamaModelOpt.modelOutputType;
    void* lastLogits;

    // Run inference to get the logits in INT16
    lastLogits = neuron_llama_inference_once(llamaRuntime, {inputToken});

    // Compute argmax on the logits
    auto outputToken = argmaxFrom16bitLogits(logitsType, lastLogits, tokenizer->vocabSize());

    return outputToken;
}

std::vector<TokenType> llm_llama_gen_response(void* llamaRuntime, const TokenizerUPtr& tokenizer,
                                              const size_t maxResponse,
                                              const TokenType firstInputToken,
                                              std::string& fullResponse, double& genTokPerSec) {
    const size_t maxTokenLength = llamaModelOpt.maxTokenLength;
    auto curTokenIndex = neuron_llama_get_token_index(llamaRuntime);
    const auto& sequenceLength = curTokenIndex; // The number of tokens the model has seen.

    double elapsed = 0, genTotalTime = 0;
    genTokPerSec = 0;
    size_t genTokCount = 0;

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
    while (genTokCount < maxResponse && sequenceLength < maxTokenLength) {
        SET_DUMP_INDEX(inferenceStep++);

        // Warn cache overflow
        if (sequenceLength == llamaModelOpt.cacheSize) {
            LOG(WARN) << "The max context length (" << llamaModelOpt.cacheSize << ") has already "
                         "been reached, about to overflow the cache.";
        }
        outputToken = llm_llama_autoregressive_per_step(llamaRuntime, tokenizer, outputToken);
        generatedTokens.push_back(outputToken);
        genTokCount++;
        curTokenIndex++;

        elapsed = timer.reset();
        genTotalTime += elapsed;
        LOG(DEBUG) << "Single loop time taken: " << elapsed * 1000 << " ms";

        // Stop when output is a stop token (default to EoS if not set)
        if (isStopToken(outputToken)) {
            std::cout << "</eos>";
            break;
        }
        // Convert token id from argmax to string (bytes)
        const std::string tokStr = tokenizer->detokenize(outputToken);

        LOG(DEBUG) << "[Response " << genTokCount << "] Output token " << outputToken
                   << ": \"" << tokStr << "\"";

        const bool isTokStrResolved = utf8Resolver.addBytes(tokStr);
        if (isTokStrResolved) {
            response = utf8Resolver.getResolvedStr();
            std::cout << response << std::flush;
            fullResponse += response;
        }
        DUMP(RESPONSE).fromValue("sampled_token", outputToken);
        DUMP(RESPONSE).fromString("sampled_text", tokStr);
        DUMP(RESPONSE).fromString("full_response", fullResponse);
    }
    std::cout << "</end>" << std::endl;
    genTokPerSec = double(genTokCount) / genTotalTime;
    std::cout << "\n[Full Response]\n" << fullResponse << std::endl;
    return generatedTokens;
}

std::tuple<double, double>
llm_llama_inference(void* llamaRuntime, const std::string& inputString,
                    const TokenizerUPtr& tokenizer, const size_t maxResponse = 50,
                    const bool parsePromptTokens = false) {
    // Convert string to tokens
    auto [inputPrompt, inputTokens] =
        get_prompt_and_tokens(inputString, tokenizer, parsePromptTokens);
    DUMP(PROMPT).fromVector("prompt_tokens", inputTokens);
    DUMP(PROMPT).fromString("prompt_text", inputPrompt);

    std::cout << "\n[Prompt]\n" << inputPrompt << '\n' << std::endl;

    // Input prompt caching
    size_t numModelInputToken = llamaModelOpt.promptTokenBatchSize;
    double promptTokPerSec;
    auto outputToken = llm_llama_digest_prompt(llamaRuntime, tokenizer, inputTokens,
                                               numModelInputToken, promptTokPerSec);
    // Swap to gen mode model
    if (numModelInputToken > 1) {
        llm_llama_swap_model(llamaRuntime, 1);
        numModelInputToken = 1;
    }

    std::string fullResponse;

    // Generation process
    std::cout << "\nResponse [Max Length = " << maxResponse << "]:" << std::endl;
    std::string tokStr = tokenizer->detokenize(outputToken);
    std::cout << tokStr << std::flush;
    fullResponse += tokStr;
    LOG(DEBUG) << "First output token " << outputToken << ": \"" << tokStr << "\"";
    DUMP(RESPONSE).fromValue("sampled_token", outputToken);
    DUMP(RESPONSE).fromString("sampled_text", tokStr);
    DUMP(RESPONSE).fromString("full_response", fullResponse);

    double genTokPerSec;
    const auto outputTokens =
        llm_llama_gen_response(llamaRuntime, tokenizer, maxResponse, outputToken, fullResponse,
                               genTokPerSec);

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
    bool parsePromptTokens = false; // Read prompt as a string of tokens
    bool onePromptPerLine = false; // Treat each line in prompt text as a single prompt. Will replace literal "\n" with new line char '\n'.
    std::string preformatterName = "";
    std::vector<std::string> promptPaths; // Paths containing the prompt text
    std::vector<std::string> prompts;

    // Sample 64 tokens input.
    const std::string defaultPrompt = "Q: What is the difference between Intel and PPC? What is the"
                                      " hardware and software differences between Intel and PPC "
                                      "Macs? A: When it comes to Apple hardware, the differences "
                                      "between the last generation of PowerPC and the first "
                                      "generation of Intel were fairly minor as far as the end "
                                      "user experience goes.";

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
        // Get current config from yaml
        llm_llama_init(&llamaRuntime, yamlConfigPath);

        // Create tokenizer for the current config. Its lifetime is until the end of this scope.
        const auto tokenizer = prepare_tokenizer();
        LOG(INFO) << "Vocab size: " << tokenizer->vocabSize();

        // Start inferencing on the prompts
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
                llm_llama_inference(llamaRuntime, prompt, tokenizer, maxResponse,
                                    parsePromptTokens);
            allPromptTokPerSec += promptTokPerSec;
            allGenTokPerSec += genTokPerSec;

            // Reset cache for the next prompt
            llm_llama_reset(llamaRuntime);
            llm_llama_swap_model(llamaRuntime, llamaModelOpt.promptTokenBatchSize);
        }
        llm_llama_release(llamaRuntime);
        std::cout << "\n[Average Performance among the given " << numPrompt << " prompts]\n";
        std::cout << "      Prompt Mode: " << allPromptTokPerSec / numPrompt << " tok/s\n";
        std::cout << "  Generative Mode: " << allGenTokPerSec / numPrompt << " tok/s\n";
    }
}