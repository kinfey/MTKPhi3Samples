#include "llm_llama.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"
#include "tokenizer/tokenizer.h"

#include "runtime/neuron_runtime.h"

#include "executor/executor_factory.h"

#include "common/dump.h"
#include "common/logging.h"

#include <thread>

#define LLM_API __attribute__((visibility("default")))

#ifdef USE_USDK_BACKEND
static constexpr bool kUseUsdkBackend = true;
#else
static constexpr bool kUseUsdkBackend = false;
#endif

#ifdef DISABLE_MULTITHREAD_MODEL_LOAD
static constexpr bool kUseMultiThreadedLoad = false;
#else
static constexpr bool kUseMultiThreadedLoad = true;
#endif

using LlamaDlaExecutor = GetExecutorClass(Llama);
using LlamaMedusaDlaExecutor = GetExecutorClass(LlamaMedusa);
using LmHeadDlaExecutor = GetExecutorClass(Neuron);
using MedusaHeadsDlaExecutor = GetExecutorClass(Neuron);

using TokenType = Tokenizer::TokenType;

struct LlamaRuntime {
    std::vector<Executor*> dlaExecutors;
    Executor* dlaLmHeadExecutor;
    Executor* dlaMedusaHeadsExecutor;
    TokenEmbeddingLut* tokenEmbLut;
    RotaryEmbeddingMasterLut* rotEmbMasterLut;
    LlamaRuntimeOptions options;
};

bool LLM_API neuron_llama_init(void** runtime, const LlamaModelOptions& modelOptions,
                               const LlamaRuntimeOptions& runtimeOptions) {

    if constexpr (kUseUsdkBackend) {
        LOG(DEBUG) << "Using NeuronUsdk (NeuronAdapter)";
    } else {
        LOG(DEBUG) << "Using Neuron Runtime";
        if (!init_neuron_runtime_library()) {
            LOG(ERROR) << "Failed to initialize runtime library.";
            *runtime = nullptr;
            return false;
        }
    }
    if (!init_dmabuf_library()) {
        LOG(ERROR) << "Failed to initialize dmabuf library.";
        *runtime = nullptr;
        return false;
    }

    const auto& dlaChunkPaths = (runtimeOptions.dlaPromptPaths.size())
                                ? runtimeOptions.dlaPromptPaths : runtimeOptions.dlaGenPaths;
    const auto numChunk = dlaChunkPaths.size();
    const auto& numModelInputTokens = modelOptions.promptTokenBatchSize;

    // External cache loading shared weights loading
    const auto numSharedWeightsPath = runtimeOptions.sharedWeightsPaths.size();
    const auto numCachePath = runtimeOptions.cachePaths.size();
    if ((numCachePath > 0 && numChunk != numCachePath) ||
        (numSharedWeightsPath > 0 && numChunk != numSharedWeightsPath)) {
        // Mismatch chunk count
        LOG(ERROR) << "Mismatch chunk count!";
        *runtime = nullptr;
        return false;
    }

    // Per-chunk path getter helpers
    auto getCachePath = [&](const size_t chunkIdx) -> std::string {
        return (numCachePath > 0) ? runtimeOptions.cachePaths[chunkIdx] : "";
    };
    auto getSharedWeightsPath = [&](const size_t chunkIdx) -> std::string {
        return (numSharedWeightsPath > 0) ? runtimeOptions.sharedWeightsPaths[chunkIdx] : "";
    };
    auto getLoraWeightsPathMap = [&](const size_t chunkIdx) {
        std::unordered_map<LoraKey, std::string> loraWeightsPathMap;
        if (!runtimeOptions.loraWeightsPaths.empty()) {
            for (const auto& [loraKey, loraChunkPaths] : runtimeOptions.loraWeightsPaths) {
                CHECK_EQ(loraChunkPaths.size(), numChunk)
                    << "Invalid LoRA input weights chunk count for '" << loraKey << "'";
                loraWeightsPathMap[loraKey] = loraChunkPaths[chunkIdx];
            }
        }
        return loraWeightsPathMap;
    };

    // Get number of caches
    const size_t numCache = 2 * modelOptions.numLayer / numChunk; // Split cache
    CHECK_EQ(modelOptions.numLayer % numChunk, 0)
        << "Requires each DLA chunk to contain equal number of layers.";
    LOG(DEBUG) << "Number of cache per dla: " << numCache;

    // Create llama runtime
    LlamaRuntime* llamaRuntime = new LlamaRuntime;
    llamaRuntime->options = runtimeOptions;

    // Initialize and prepare rotary embedding master lookup-table
    const size_t rotEmbDim = modelOptions.hiddenSize / modelOptions.numHead;
    llamaRuntime->rotEmbMasterLut = new RotaryEmbeddingMasterLut(
        modelOptions.rotEmbType, modelOptions.maxTokenLength, rotEmbDim, modelOptions.rotEmbBase);
    llamaRuntime->rotEmbMasterLut->generate();

    constexpr size_t numRotEmbInputs = 1;

    const ExecutorType llamaExecType = (modelOptions.numMedusaHeads > 0)
                                       ? ExecutorType::LlamaMedusa : ExecutorType::Llama;
    ExecutorFactory llamaExecFactory(llamaExecType);

    for (int chunkIdx = 0; chunkIdx < numChunk; ++chunkIdx) {
        std::vector<LLMRuntimeInfo> runtimeInfos;
        auto addRuntimeInfo = [&](const auto& dlaPaths, const size_t batchSize) {
            if (dlaPaths.size() > 0) {
                DCHECK_GT(dlaPaths.size(), chunkIdx);
                runtimeInfos.push_back({dlaPaths[chunkIdx], batchSize, modelOptions.cacheSize});
            }
        };
        addRuntimeInfo(runtimeOptions.dlaPromptPaths, numModelInputTokens);
        addRuntimeInfo(runtimeOptions.dlaGenPaths, modelOptions.genTokenBatchSize);

        const auto& sharedWeightsPath = getSharedWeightsPath(chunkIdx);
        LOG(DEBUG) << "Loading DLA " << chunkIdx;

        auto dlaExec = llamaExecFactory.create(
            runtimeInfos, sharedWeightsPath, modelOptions.maxTokenLength,
            modelOptions.cacheSize, numCache, modelOptions.cacheType, modelOptions.maskType,
            llamaRuntime->rotEmbMasterLut, numRotEmbInputs, getLoraWeightsPathMap(chunkIdx),
            runtimeOptions.initWithLoraKey, runtimeOptions.loraInputCount, getCachePath(chunkIdx),
            runtimeOptions.startTokenIndex
        );
        llamaRuntime->dlaExecutors.push_back(dlaExec);
    }

    ExecutorFactory neuronExecFactory(ExecutorType::Neuron);

    if (!runtimeOptions.dlaLmHeadPath.empty()) {
        LOG(DEBUG) << "Loading and initializing Executor for LM Head.";
        const std::vector<std::string> modelPaths = {runtimeOptions.dlaLmHeadPath};
        llamaRuntime->dlaLmHeadExecutor = neuronExecFactory.create(modelPaths);
        llamaRuntime->dlaLmHeadExecutor->initialize();
        llamaRuntime->dlaLmHeadExecutor->registerRuntimeIO();
    } else {
        LOG(DEBUG) << "No LM Head is specified in config, assign dlaLmHeadExecutor to nullptr.";
        llamaRuntime->dlaLmHeadExecutor = nullptr;
    }

    if (!runtimeOptions.dlaMedusaHeadsPath.empty()) {
        LOG(DEBUG) << "Loading and initializing Executor for Medusa Heads.";
        const std::vector<std::string> modelPaths = {runtimeOptions.dlaMedusaHeadsPath};
        llamaRuntime->dlaMedusaHeadsExecutor = neuronExecFactory.create(modelPaths);
        llamaRuntime->dlaMedusaHeadsExecutor->initialize();
        llamaRuntime->dlaMedusaHeadsExecutor->registerRuntimeIO();
    } else {
        LOG(DEBUG) << "No Medusa Heads is specified in config, assign dlaMedusaHeadsExecutor to "
                   << "nullptr.";
        llamaRuntime->dlaMedusaHeadsExecutor = nullptr;
    }

    // Use multi-threading to speedup model loading
    std::vector<std::thread> threadPool;

    auto initExecutor = [&](const auto dlaExec) {
        if constexpr (kUseMultiThreadedLoad)
            threadPool.emplace_back(&Executor::initialize, dlaExec);
        else
            dlaExec->initialize();
    };

    auto initTokenEmbLut = [&]{
        // NOTE: Token embedding lookup-table type must match the model input type
        llamaRuntime->tokenEmbLut = new TokenEmbeddingLut(
            runtimeOptions.tokenEmbPath, modelOptions.modelInputType, modelOptions.hiddenSize);
        LOG(DEBUG) << "Initialized input token embedding lookup table.";
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
        // Initialize after reserving the input buffer so that the buffer allocator doesn't need to
        // allocate for inputs that are using an existing buffer created elsewhere.
        auto dlaExec = llamaRuntime->dlaExecutors[chunkIdx];
        LOG(DEBUG) << "Initializing DLA " << chunkIdx;
        if (chunkIdx > 0)
            dlaExec->reserveInputBuffer(); // Prevent allocation of buffer for input 0
        initExecutor(dlaExec);
    }
    threadPool.emplace_back(initTokenEmbLut);

    // Wait for model to finish loading
    for (auto& thread : threadPool) {
        thread.join();
    }
    LOG(DEBUG) << "Done initializing DLAs";

    // Chain the IO between the runtime chunks:
    // InputToken -> [EmbeddingLut -> DlaChunk1 -> DlaChunk2 -> ... -> DlaChunkN]-> Output
    auto getPrevChunkOutput = [&](const int chunkIdx) -> const IOBuffer& {
        DCHECK_GE(chunkIdx, 1);
        return llamaRuntime->dlaExecutors[chunkIdx - 1]->getOutput();
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
        // Initialize after setModelInput so that the buffer allocator doesn't need to allocate for
        // inputs that are using an existing buffer.
        auto dlaExec = llamaRuntime->dlaExecutors[chunkIdx];
        if (chunkIdx > 0)
            dlaExec->setModelInput(getPrevChunkOutput(chunkIdx));
        dlaExec->updateModelIO(); // Ensure IO sizes are correct, esp when using prev chunk buffer
        dlaExec->registerRuntimeIO(); // Attach allocated buffers to model IO
    }
    // Link first chunk emb input to token emb lut output
    const auto& tokenEmbInput = llamaRuntime->dlaExecutors.front()->getInput();
    llamaRuntime->tokenEmbLut->setOutput(tokenEmbInput.buffer, tokenEmbInput.sizeBytes);

    LOG(DEBUG) << "Done model chunks IO chaining";

    *runtime = llamaRuntime;
    return true;
}

void LLM_API neuron_llama_swap_model(void* runtime, const size_t batchSize) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto numDlaChunk = llamaRuntime->dlaExecutors.size();

    // Use multi-threading to speedup model swapping (if necessary)
    std::vector<std::thread> threadPool;

    auto swapModel = [&](const auto chunkIdx) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors[chunkIdx]);
        if (!llamaDlaExec->hotSwapModel(batchSize))
            LOG(ERROR) << "Hot swapping failed on chunk " << chunkIdx;
    };

    const bool isSharedWeightsUsed = !llamaRuntime->options.sharedWeightsPaths.empty();
    for (size_t chunkIdx = 0; chunkIdx < numDlaChunk; chunkIdx++) {
        if (!kUseMultiThreadedLoad || isSharedWeightsUsed)
            swapModel(chunkIdx); // Multi-threading will slightly slow down with weights sharing
        else
            threadPool.emplace_back(swapModel, chunkIdx);
    }

    // Wait for model swapping threads to finish
    for (auto& thread : threadPool) {
        thread.join();
    }
}

void LLM_API neuron_llama_release(void* runtime) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        dlaExec->release();
        delete dlaExec;
    };
    llamaRuntime->dlaExecutors.clear();
    delete llamaRuntime->dlaLmHeadExecutor;
    delete llamaRuntime->dlaMedusaHeadsExecutor;
    delete llamaRuntime->tokenEmbLut;
    delete llamaRuntime->rotEmbMasterLut;
    delete llamaRuntime;
}

void LLM_API neuron_llama_set_medusa_tree_attn(void* runtime,
                                               const std::vector<std::vector<int>>& mask,
                                               const std::vector<size_t>& positions) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        static_cast<LlamaMedusaDlaExecutor*>(dlaExec)->setMedusaTreeAttn(mask, positions);
    }
}

void* LLM_API neuron_llama_inference_once(void* runtime, const std::vector<TokenType>& inputTokens,
                                          const bool lastLogits) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.front());
    const auto modelNumInputToken = firstExecutor->getModelNumInputToken();
    const auto currentTokenIndex = firstExecutor->getTokenIndex();

    // Error checking
    if (inputTokens.size() > modelNumInputToken) {
        LOG(FATAL) << "The required input token length (" << inputTokens.size() << ") "
                   << "exceeds what the model can take in (" << modelNumInputToken << ")";
    }

    // Handle padding
    auto curInputTokens = inputTokens;
    const size_t padSize = modelNumInputToken - inputTokens.size();
    constexpr TokenType padToken = 0; // By right any token should work.

    // Use left-padding if possible as it has lower overhead than right-padding.
    // Right-padding involves cache shifting (for non-ring buffer) which incurs additional overhead.
    const bool isLeftPadAllowed = (currentTokenIndex == 0);
    if (padSize > 0) {
        if (isLeftPadAllowed) {
            // Pad left since the cache is fresh new.
            curInputTokens.insert(curInputTokens.begin(), padSize, padToken);
            LOG(DEBUG) << "Padding left by " << padSize;
        } else {
            // Pad right since left side of cache is occupied either by loaded cache or previous
            // inference pass.
            curInputTokens.insert(curInputTokens.end(), padSize, padToken);
            LOG(DEBUG) << "Padding right by " << padSize;
        }
    }
    CHECK_EQ(modelNumInputToken, curInputTokens.size());

    auto setPadding = [&](const auto& llamaExecutor) {
        if (isLeftPadAllowed)
            llamaExecutor->setLeftPadding(padSize);
        else
            llamaExecutor->setRightPadding(padSize);
    };

    static size_t inferenceStep = 0;
    SET_DUMP_INDEX(inferenceStep++);

    llamaRuntime->tokenEmbLut->lookupEmbedding(curInputTokens);
    LOG(DEBUG) << "Emb Lut output buf[0] = "
               << reinterpret_cast<const int16_t*>(firstExecutor->getInputBuffer())[0];

    size_t chunkIdx = 0;
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        SET_DUMP_CHUNK_INDEX(chunkIdx);

        // Set padding if needed
        setPadding(llamaDlaExec);

        // Update auxiliary inputs to model
        llamaDlaExec->updatePosEmbAndMask(modelNumInputToken);

        // Run inference based on the inputs assigned by previous model chunk and rot emb & mask
        llamaDlaExec->runInference();

        // Advance token index by the actual number that the model input requires.
        llamaDlaExec->advanceTokenIndex();

        // Perform any necessary adjustments when padding is used
        llamaDlaExec->paddingPostprocess();

        // Dump chunk output
        const auto chunkOutputBuffer = llamaDlaExec->getOutputBuffer();
        const auto chunkOutputSize = llamaDlaExec->getModelOutputSizeBytes();
        DUMP(CHUNK_OUT).fromBinary("output", chunkOutputBuffer, chunkOutputSize);

        // Dump chunk cache outputs
        if (SHOULD_DUMP(CACHE)) {
            std::vector<char*> cacheBuffers;
            size_t sizePerCache;
            llamaDlaExec->getCacheBuffersWithSize(cacheBuffers, sizePerCache);
            for (size_t i = 0; i < cacheBuffers.size(); i++) {
                DUMP(CACHE).fromBinary("cache_" + std::to_string(i), cacheBuffers[i], sizePerCache);
            }
        }
        chunkIdx++;
    }

    auto getLogitsBuffer = [=](const auto executor, const auto tokenSize, const auto rightPadSize) {
        auto logitsBuffer = reinterpret_cast<char*>(executor->getOutputBuffer());
        size_t logitsOffset = 0;
        if (lastLogits && tokenSize > 1) {
            const auto logitsSize = executor->getModelOutputSizeBytes();
            logitsOffset = (logitsSize / tokenSize) * (tokenSize - 1 - rightPadSize);
            DCHECK_LE(logitsOffset, logitsSize);
        }
        return logitsBuffer + logitsOffset;
    };

    // Clipped subtraction between unsigned values
    auto max0Subtract = [](const auto lhs, const auto rhs) -> decltype(lhs) {
        if (lhs < rhs)
            return 0;
        return lhs - rhs;
    };

    const auto finalExecutor = llamaRuntime->dlaExecutors.back();
    const auto lmHeadExecutor = llamaRuntime->dlaLmHeadExecutor;
    const size_t rightPadSize = !isLeftPadAllowed * padSize;

    if (!lmHeadExecutor) {
        // No separated LM head, return the logits from the final executor directly.
        return getLogitsBuffer(finalExecutor, modelNumInputToken, rightPadSize);
    }

    // Execute the LM head on the hidden state generated from the last chunk of decoder layers.
    const auto hiddenStateSize = finalExecutor->getModelOutputSizeBytes();
    const auto lmHeadInputSize = lmHeadExecutor->getModelInputSizeBytes();
    const auto perTokenHiddenStateSize = hiddenStateSize / modelNumInputToken;
    const auto lmHeadTokenSize = lmHeadInputSize / perTokenHiddenStateSize;

    const size_t tokenOffset = max0Subtract(modelNumInputToken, rightPadSize + lmHeadTokenSize);
    const size_t lmHeadPadSize = max0Subtract(lmHeadTokenSize, modelNumInputToken - rightPadSize);

    const size_t hiddenStateOffset = perTokenHiddenStateSize * tokenOffset;
    DCHECK_LE(hiddenStateOffset, hiddenStateSize);
    DCHECK_LE(hiddenStateSize - hiddenStateOffset, lmHeadInputSize);
    auto hiddenStateBuffer = reinterpret_cast<char*>(finalExecutor->getOutputBuffer());
    lmHeadExecutor->runInference(hiddenStateBuffer + hiddenStateOffset, lmHeadInputSize);

    // Return logits from LM head output
    if (!lastLogits || modelNumInputToken == 1) {
        // If the logits of all the input tokens are expected, the token size of LM-Head chunk
        // must align with the token size of the currently used chunks (`modelNumInputToken`).
        DCHECK_EQ(modelNumInputToken, lmHeadTokenSize);
    }
    return getLogitsBuffer(lmHeadExecutor, lmHeadTokenSize, lmHeadPadSize);
}

std::tuple<void*, void*>  // logits, last_hidden_states
LLM_API neuron_llama_inference_once_return_hidden(void* runtime,
                                                  const std::vector<TokenType>& inputTokens,
                                                  const bool lastLogits) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.front());
    const auto modelNumInputToken = firstExecutor->getModelNumInputToken();
    const auto currentTokenIndex = firstExecutor->getTokenIndex();

    // Error checking
    if (llamaRuntime->dlaLmHeadExecutor == nullptr) {
        LOG(FATAL) << "Separated LM Head is necessary for getting the last hidden states.";
    }
    if (inputTokens.size() > modelNumInputToken) {
        LOG(FATAL) << "The required input token length (" << inputTokens.size() << ") "
                   << "exceeds what the model can take in (" << modelNumInputToken << ")";
    }

    // Handle padding
    std::vector<TokenType> curInputTokens = inputTokens;
    const size_t padSize = modelNumInputToken - inputTokens.size();
    constexpr TokenType padToken = 0; // By right any token should work.

    // Use left-padding if possible as it has lower overhead than right-padding.
    // Right-padding involves cache shifting (for non-ring buffer) which incurs additional overhead.
    const bool isLeftPadAllowed = (currentTokenIndex == 0);
    if (padSize > 0) {
        if (isLeftPadAllowed) {
            // Pad left since the cache is fresh new.
            curInputTokens.insert(curInputTokens.begin(), padSize, padToken);
            LOG(DEBUG) << "Padding left by " << padSize;
        } else {
            // Pad right since left side of cache is occupied either by loaded cache or previous
            // inference pass.
            curInputTokens.insert(curInputTokens.end(), padSize, padToken);
            LOG(DEBUG) << "Padding right by " << padSize;
        }
    }
    CHECK_EQ(modelNumInputToken, curInputTokens.size());

    auto setPadding = [&](const auto& llamaExecutor) {
        if (isLeftPadAllowed)
            llamaExecutor->setLeftPadding(padSize);
        else
            llamaExecutor->setRightPadding(padSize);
    };

    static size_t inferenceStep = 0;
    SET_DUMP_INDEX(inferenceStep++);

    llamaRuntime->tokenEmbLut->lookupEmbedding(curInputTokens);
    LOG(DEBUG) << "Emb Lut output buf[0] = "
               << reinterpret_cast<const int16_t*>(firstExecutor->getInputBuffer())[0];

    size_t chunkIdx = 0;
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        SET_DUMP_CHUNK_INDEX(chunkIdx);

        // Set padding if needed
        setPadding(llamaDlaExec);

        // Update auxiliary inputs to model
        llamaDlaExec->updatePosEmbAndMask(modelNumInputToken);

        // Run inference based on the inputs assigned by previous model chunk and rot emb & mask
        llamaDlaExec->runInference();

        // Advance token index by the actual number that the model input requires.
        llamaDlaExec->advanceTokenIndex();

        // Perform any necessary adjustments when padding is used
        llamaDlaExec->paddingPostprocess();

        // Dump chunk output
        const auto chunkOutputBuffer = llamaDlaExec->getOutputBuffer();
        const auto chunkOutputSize = llamaDlaExec->getModelOutputSizeBytes();
        DUMP(CHUNK_OUT).fromBinary("output", chunkOutputBuffer, chunkOutputSize);

        // Dump chunk cache outputs
        if (SHOULD_DUMP(CACHE)) {
            std::vector<char*> cacheBuffers;
            size_t sizePerCache;
            llamaDlaExec->getCacheBuffersWithSize(cacheBuffers, sizePerCache);
            for (size_t i = 0; i < cacheBuffers.size(); i++) {
                DUMP(CACHE).fromBinary("cache_" + std::to_string(i), cacheBuffers[i], sizePerCache);
            }
        }
        chunkIdx++;
    }

    // Clipped subtraction between unsigned values
    auto max0Subtract = [](const auto lhs, const auto rhs) -> decltype(lhs) {
        if (lhs < rhs)
            return 0;
        return lhs - rhs;
    };

    const auto finalExecutor = llamaRuntime->dlaExecutors.back();
    const auto lmHeadExecutor = llamaRuntime->dlaLmHeadExecutor;
    const auto hiddenStateSize = finalExecutor->getModelOutputSizeBytes();
    const size_t rightPadSize = !isLeftPadAllowed * padSize;

    // Execute the LM head on the hidden state generated from the last chunk of decoder layers.
    const auto lmHeadInputSize = lmHeadExecutor->getModelInputSizeBytes();
    const auto perTokenHiddenStateSize = hiddenStateSize / modelNumInputToken;
    const auto lmHeadTokenSize = lmHeadInputSize / perTokenHiddenStateSize;

    const size_t tokenOffset = max0Subtract(modelNumInputToken, rightPadSize + lmHeadTokenSize);
    const size_t lmHeadPadSize = max0Subtract(lmHeadTokenSize, modelNumInputToken - rightPadSize);

    const size_t hiddenStateOffset = perTokenHiddenStateSize * tokenOffset;
    DCHECK_LE(hiddenStateOffset, hiddenStateSize);
    DCHECK_LE(hiddenStateSize - hiddenStateOffset, lmHeadInputSize);
    auto hiddenStateBuffer = reinterpret_cast<char*>(finalExecutor->getOutputBuffer());
    lmHeadExecutor->runInference(hiddenStateBuffer + hiddenStateOffset, lmHeadInputSize);

    // Return logits from LM head output
    size_t logitsOffset = 0;
    auto logitsBuffer = reinterpret_cast<char*>(lmHeadExecutor->getOutputBuffer());
    if (lastLogits && modelNumInputToken > 1) {
        const auto logitsSize = lmHeadExecutor->getModelOutputSizeBytes();
        logitsOffset = (logitsSize / lmHeadTokenSize) * (lmHeadTokenSize - 1 - lmHeadPadSize);
        DCHECK_LE(logitsOffset, logitsSize);
    } else {
        // If the logits of all the input tokens are expected, the token size of LM-Head chunk
        // should align with the token size of the currently used chunks (`modelNumInputToken`).
        DCHECK_EQ(modelNumInputToken, lmHeadTokenSize);
    }
    return {logitsBuffer + logitsOffset, hiddenStateBuffer};
}

// Return medusa logits
void* LLM_API neuron_medusa_heads_inference_once(void* runtime, void* hiddenState) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    // Error checking
    if (llamaRuntime->dlaMedusaHeadsExecutor == nullptr) {
        LOG(FATAL) << "Medusa Heads is necessary for Medusa inference.";
    }

    const auto medusaExecutor = static_cast<MedusaHeadsDlaExecutor*>(
        llamaRuntime->dlaMedusaHeadsExecutor
    );

    medusaExecutor->runInference(
        reinterpret_cast<char*>(hiddenState), medusaExecutor->getModelInputSizeBytes()
    );
    auto logitsBuffer = medusaExecutor->getOutputBuffer();

    return reinterpret_cast<char*>(logitsBuffer);
}

void LLM_API neuron_llama_apply_lora(void* runtime, const LoraKey& loraKey) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        llamaDlaExec->applyLoraWeights(loraKey);
    }
}

void LLM_API neuron_llama_apply_lora_from_buffer(void* runtime,
                                                 const std::vector<char*>& loraWeightBuffers,
                                                 const std::vector<size_t>& sizes) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);

    const auto& loraInputCount = llamaRuntime->options.loraInputCount; // Per chunk
    const auto& chunkCount = llamaRuntime->dlaExecutors.size();

    // Verify arguments
    CHECK_EQ(loraWeightBuffers.size(), sizes.size());
    CHECK_EQ(chunkCount * loraInputCount, loraWeightBuffers.size())
        << "The provided number of LoRA weights buffers does not match the total number of "
           "LoRA inputs";

    auto getSubsetForChunk = [&](const size_t chunkIdx, const auto& vec) {
        const size_t start = chunkIdx * loraInputCount;
        const size_t end = start + loraInputCount;
        return std::vector(vec.begin() + start, vec.begin() + end);
    };

    // Chunk the LoRA weight buffers and feed into each DLA chunk.
    for (size_t chunkIdx = 0; chunkIdx < chunkCount; chunkIdx++) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors[chunkIdx]);
        const auto& loraWeightForChunk = getSubsetForChunk(chunkIdx, loraWeightBuffers);
        const auto& sizesForChunk = getSubsetForChunk(chunkIdx, sizes);
        llamaDlaExec->applyLoraWeights(loraWeightForChunk, sizesForChunk);
    }
}

void LLM_API neuron_llama_remove_lora(void* runtime) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        llamaDlaExec->removeLoraWeights();
    }
}

void LLM_API neuron_llama_get_caches(void* runtime, std::vector<std::vector<char*>>& caches,
                                     size_t& byteSizePerCache) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        std::vector<char*> chunkCaches;
        llamaDlaExec->getCacheBuffersWithSize(chunkCaches, byteSizePerCache);
        caches.emplace_back(chunkCaches);
    }
}

void LLM_API neuron_llama_reset(void* runtime, const bool resetCache) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        if (resetCache) {
            // Reset cache and token index, resetTokenIndex() will be called
            llamaDlaExec->initCache();
        } else {
            // Reset token index without resetting cache
            llamaDlaExec->resetTokenIndex();
        }
    }
}

size_t LLM_API neuron_llama_get_per_token_logits_size(void* runtime) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto finalExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.back());
    const auto modelNumInputToken = finalExecutor->getModelNumInputToken();
    if (llamaRuntime->dlaLmHeadExecutor == nullptr) {
        const auto logitsSize = finalExecutor->getModelOutputSizeBytes();
        return logitsSize / modelNumInputToken;
    } else {
        const auto perTokenHiddenStateSize = finalExecutor->getModelOutputSizeBytes()
                                           / modelNumInputToken;
        const auto lmHeadExecutor = static_cast<LmHeadDlaExecutor*>(llamaRuntime->dlaLmHeadExecutor);
        const auto lmHeadTokenBatchSize = lmHeadExecutor->getModelInputSizeBytes()
                                        / perTokenHiddenStateSize;
        const auto logitsSize = lmHeadExecutor->getModelOutputSizeBytes();
        return logitsSize / lmHeadTokenBatchSize;
    }
}

size_t LLM_API neuron_llama_get_per_token_hidden_states_size(void* runtime) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    // Error checking
    if (llamaRuntime->dlaLmHeadExecutor == nullptr) {
        LOG(FATAL) << "Separated LM Head is necessary for calculating the size of hidden states.";
    }
    const auto finalExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.back());
    const auto modelNumInputToken = finalExecutor->getModelNumInputToken();
    const auto hiddenStateSize = finalExecutor->getModelOutputSizeBytes();
    return hiddenStateSize / modelNumInputToken;
}

size_t LLM_API neuron_llama_get_token_index(void* runtime) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.front());
    return firstExecutor->getTokenIndex();
}

void LLM_API neuron_llama_rollback(void* runtime, const size_t rollbackCount) {
    if (rollbackCount == 0)
        return;
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    const auto finalExecutor = static_cast<LlamaDlaExecutor*>(llamaRuntime->dlaExecutors.back());
    const auto& modelNumInputToken = finalExecutor->getModelNumInputToken();
    for (auto& dlaExec : llamaRuntime->dlaExecutors) {
        // align tokenindex and rollback cache
        auto llamaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        llamaDlaExec->alignInputTokens(modelNumInputToken - rollbackCount);
    }
}

void LLM_API neuron_medusa_rollback(void* runtime, const std::vector<size_t>& acceptedIndices) {
    auto llamaRuntime = reinterpret_cast<LlamaRuntime*>(runtime);
    for (auto& dlaExec : llamaRuntime->dlaExecutors) {
        auto llamaMedusaDlaExec = static_cast<LlamaMedusaDlaExecutor*>(dlaExec);
        llamaMedusaDlaExec->rollbackTreeCache(acceptedIndices);
        llamaMedusaDlaExec->alignInputTokens(acceptedIndices.size());
    }
}
