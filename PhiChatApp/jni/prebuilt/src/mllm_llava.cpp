#include "mllm_llava.h"

#include "executor/executor_factory.h"
#include "embedding_producer.h"
#include "image_transform.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/token_embedding.h"
#include "runtime/neuron_runtime.h"
#include "tokenizer/tokenizer.h"
#include "common/dump.h"
#include "common/logging.h"
#include "common/timer.h"

#include <thread>
#include <string>
#include <string_view>
#include <algorithm>

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

#ifdef ALLOW_MLLM_LEFT_PADDING
static constexpr bool kAllowLeftPadding = true;
#else
static constexpr bool kAllowLeftPadding = false;
#endif

using LlamaDlaExecutor = GetExecutorClass(Llama);
using ClipEmbDlaExecutor = GetExecutorClass(Neuron);
using PatchEmbTfliteExecutor = GetExecutorClass(TFLite);

using TokenType = Tokenizer::TokenType;

struct LlavaRuntime {
    std::vector<Executor*> dlaExecutors;
    RotaryEmbeddingMasterLut* rotEmbMasterLut;
    PatchEmbTfliteExecutor* clipPatchEmbExecutor;
    ClipEmbDlaExecutor* clipExecutor;
    TokenEmbeddingLut* tokenEmbLut;
    LlavaRuntimeOptions options;
};

inline std::vector<std::pair<size_t, size_t>>
subtoken_delimit(const std::vector<TokenType>& inputTokens, const TokenType delimiter,
                 const bool preserveDelimiter = true) {
    std::vector<std::pair<size_t, size_t>> result; // Intervals

    auto appendResult = [&result](const auto start, const auto end) {
        if (start != end)
            result.push_back({start, end});
    };

    auto findDelimIdx = [&](const size_t startIndex) {
        const auto firstIt = inputTokens.begin();
        const auto lastIt = inputTokens.end();
        return std::find(firstIt + startIndex, lastIt, delimiter) - firstIt;
    };

    size_t start = 0;
    size_t delimIdx = findDelimIdx(0);

    while (delimIdx < inputTokens.size()) {
        appendResult(start, delimIdx);
        if (preserveDelimiter) {
            appendResult(delimIdx, delimIdx + 1);
        }
        start = delimIdx + 1;
        delimIdx = findDelimIdx(start);
    }
    appendResult(start, delimIdx);
    return result;
}

bool LLM_API neuron_llava_init(void** runtime, const LlamaModelOptions& modelOptions,
                               const LlavaRuntimeOptions& runtimeOptions) {
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
                                ? runtimeOptions.dlaPromptPaths
                                : runtimeOptions.dlaGenPaths;
    const auto numChunk = dlaChunkPaths.size();
    const auto& numModelInputTokens = modelOptions.promptTokenBatchSize;

    // External cache loading shared weights loading
    const auto numSharedWeightsPath = runtimeOptions.sharedWeightsPaths.size();
    const auto numCachePath = runtimeOptions.cachePaths.size();
    if ((numCachePath > 0 && numChunk != numCachePath)
        || (numSharedWeightsPath > 0 && numChunk != numSharedWeightsPath)) {
        // Mismatch chunk count
        LOG(ERROR) << "Mismatch chunk count!";
        *runtime = nullptr;
        return false;
    }
    auto getCachePath = [&](const size_t chunkIdx) -> std::string {
        return (numCachePath > 0) ? runtimeOptions.cachePaths[chunkIdx] : "";
    };
    auto getSharedWeightsPath = [&](const size_t chunkIdx) -> std::string {
        return (numSharedWeightsPath > 0) ? runtimeOptions.sharedWeightsPaths[chunkIdx] : "";
    };

    const size_t numCache = 2 * modelOptions.numLayer / numChunk; // Split cache
    CHECK_EQ(modelOptions.numLayer % numChunk, 0) << "Requires each DLA chunk to contain equal "
                                                  << "number of layers.";
    LOG(DEBUG) << "Number of cache per dla: " << numCache;
    // Create llama runtime
    LlavaRuntime* llavaRuntime = new LlavaRuntime;
    llavaRuntime->options = runtimeOptions;

    // Initialize and prepare rotary embedding master lookup-table
    const size_t rotEmbDim = modelOptions.hiddenSize / modelOptions.numHead;
    llavaRuntime->rotEmbMasterLut = new RotaryEmbeddingMasterLut(
        modelOptions.rotEmbType, modelOptions.maxTokenLength, rotEmbDim);
    llavaRuntime->rotEmbMasterLut->generate();

    constexpr size_t numRotEmbInputs = 1;

    for (size_t chunkIdx = 0; chunkIdx < numChunk; ++chunkIdx) {
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
        auto dlaExec = new LlamaDlaExecutor(
            runtimeInfos, sharedWeightsPath, modelOptions.maxTokenLength, modelOptions.cacheSize,
            numCache, modelOptions.cacheType, modelOptions.maskType, llavaRuntime->rotEmbMasterLut,
            numRotEmbInputs, {}, "", 0, getCachePath(chunkIdx), runtimeOptions.startTokenIndex);
        llavaRuntime->dlaExecutors.push_back(dlaExec);
    }

    // Use multi-threading to speedup model loading
    std::vector<std::thread> threadPool;

    auto initExecutor = [&](const auto dlaExec) {
        if constexpr (kUseMultiThreadedLoad)
            threadPool.emplace_back(&Executor::initialize, dlaExec);
        else
            dlaExec->initialize();
    };

    auto initTokenEmbLut = [&] {
        // NOTE: Token embedding lookup-table type must match the model input type
        llavaRuntime->tokenEmbLut = new TokenEmbeddingLut(
            runtimeOptions.tokenEmbPath, modelOptions.modelInputType, modelOptions.hiddenSize);
        LOG(DEBUG) << "Initialized input token embedding lookup table.";
    };

    auto initClipExecutors = [&] {
        const auto& patchEmbPath = runtimeOptions.patchEmbPath;
        llavaRuntime->clipPatchEmbExecutor = new PatchEmbTfliteExecutor(patchEmbPath);
        llavaRuntime->clipPatchEmbExecutor->initialize();

        // Load CLIP Model
        LOG(DEBUG) << "Loading CLIP DLA: " << runtimeOptions.clipPath;
        const auto& clipPath = std::vector{runtimeOptions.clipPath};
        llavaRuntime->clipExecutor = new ClipEmbDlaExecutor(clipPath);

        llavaRuntime->clipExecutor->setModelInput(llavaRuntime->clipPatchEmbExecutor->getOutput());
        llavaRuntime->clipExecutor->initialize();
        llavaRuntime->clipExecutor->registerRuntimeIO();

        LOG(DEBUG) << "Initialized CLIP DLA";
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
        // Initialize after reserving the input buffer so that the buffer allocator doesn't need to
        // allocate for inputs that are using an existing buffer created elsewhere.
        auto dlaExec = llavaRuntime->dlaExecutors[chunkIdx];
        LOG(DEBUG) << "Initializing DLA " << chunkIdx;
        if (chunkIdx > 0)
            dlaExec->reserveInputBuffer(); // Prevent allocation of buffer for input 0
        initExecutor(dlaExec);
    }
    threadPool.emplace_back(initTokenEmbLut);
    threadPool.emplace_back(initClipExecutors);

    // Wait for model to finish loading
    for (auto& thread : threadPool) {
        thread.join();
    }
    LOG(DEBUG) << "Done initializing DLAs";

    // Chain the IO between the runtime chunks:
    // InputToken -> [EmbeddingLut -> DlaChunk1 -> DlaChunk2 -> ... -> DlaChunkN]-> Output
    auto getPrevChunkOutput = [&](const int chunkIdx) -> const IOBuffer& {
        DCHECK_GE(chunkIdx, 1);
        return llavaRuntime->dlaExecutors[chunkIdx - 1]->getOutput();
    };

    for (size_t chunkIdx = 0; chunkIdx < numChunk; chunkIdx++) {
        // Initialize after setModelInput so that the buffer allocator doesn't need to allocate for
        // inputs that are using an existing buffer.
        auto dlaExec = llavaRuntime->dlaExecutors[chunkIdx];
        if (chunkIdx > 0)
            dlaExec->setModelInput(getPrevChunkOutput(chunkIdx));
        dlaExec->updateModelIO(); // Ensure IO sizes are correct, esp when using prev chunk buffer
        dlaExec->registerRuntimeIO(); // Attach allocated buffers to model IO
    }
    // Link first chunk emb input to token emb lut output
    const auto& tokenEmbInput = llavaRuntime->dlaExecutors.front()->getInput();
    llavaRuntime->tokenEmbLut->setOutput(tokenEmbInput.buffer, tokenEmbInput.sizeBytes);

    LOG(DEBUG) << "Done model chunks IO chaining";

    *runtime = llavaRuntime;
    return true;
}

void* LLM_API neuron_llava_inference_once(void* runtime, const size_t leftPadSize,
                                          const size_t rightPadSize, const void* inputEmb,
                                          const bool lastLogits) {
    DCHECK(leftPadSize == 0 || rightPadSize == 0)
        << "Invalid padding: Both both left and right padding are set.";

    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);

    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llavaRuntime->dlaExecutors.front());
    const auto modelNumInputToken = firstExecutor->getModelNumInputToken();

    if (inputEmb != nullptr) {
        // Manually provided input embedding
        const auto inputEmbSize = firstExecutor->getModelInputSizeBytes();
        firstExecutor->setModelInput(inputEmb, inputEmbSize);
        firstExecutor->registerRuntimeIO();
    }

    static size_t inferenceStep = 0;
    SET_DUMP_INDEX(inferenceStep++);

    size_t chunkIdx = 0;
    for (auto& dlaExec : llavaRuntime->dlaExecutors) {
        auto llavaDlaExec = static_cast<LlamaDlaExecutor*>(dlaExec);
        SET_DUMP_CHUNK_INDEX(chunkIdx);

        // Set padding if needed
        if (leftPadSize > 0)
            llavaDlaExec->setLeftPadding(leftPadSize);
        else if (rightPadSize > 0)
            llavaDlaExec->setRightPadding(rightPadSize);

        // Update auxiliary inputs to model
        llavaDlaExec->updatePosEmbAndMask(modelNumInputToken);

        // Run inference based on the inputs assigned by previous model chunk and rot emb & mask
        llavaDlaExec->runInference();

        // Advance token index by the actual number that the model input requires.
        llavaDlaExec->advanceTokenIndex();

        // Perform any necessary adjustments when padding is used
        llavaDlaExec->paddingPostprocess();

        // Dump chunk output
        const auto chunkOutputBuffer = llavaDlaExec->getOutputBuffer();
        const auto chunkOutputSize = llavaDlaExec->getModelOutputSizeBytes();
        DUMP(CHUNK_OUT).fromBinary("output", chunkOutputBuffer, chunkOutputSize);

        // Dump chunk cache outputs
        if (SHOULD_DUMP(CACHE)) {
            std::vector<char*> cacheBuffers;
            size_t sizePerCache;
            llavaDlaExec->getCacheBuffersWithSize(cacheBuffers, sizePerCache);
            for (size_t i = 0; i < cacheBuffers.size(); i++) {
                DUMP(CACHE).fromBinary("cache_" + std::to_string(i), cacheBuffers[i], sizePerCache);
            }
        }
        chunkIdx++;
    }

    // Return logits
    const auto finalExecutor = static_cast<LlamaDlaExecutor*>(llavaRuntime->dlaExecutors.back());
    auto logitsBuffer = finalExecutor->getOutputBuffer();
    size_t offset = 0;
    if (lastLogits && modelNumInputToken > 1) {
        const auto logitsSize = finalExecutor->getModelOutputSizeBytes();
        offset = (logitsSize / modelNumInputToken) * (modelNumInputToken - 1 - rightPadSize);
        DCHECK_LE(offset, logitsSize);
    }
    return reinterpret_cast<char*>(logitsBuffer) + offset;
}

void LLM_API neuron_llava_release(void* runtime) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);
    for (auto dlaExec : llavaRuntime->dlaExecutors) {
        dlaExec->release();
        delete dlaExec;
    };
    llavaRuntime->dlaExecutors.clear();
    delete llavaRuntime->tokenEmbLut;
    delete llavaRuntime->rotEmbMasterLut;
    delete llavaRuntime;

    // Delete CLIP Part
    delete llavaRuntime->clipExecutor;
    delete llavaRuntime->clipPatchEmbExecutor;
}

void* LLM_API neuron_llava_consume_prompt(void* runtime, const std::vector<TokenType>& tokens,
                                          const std::vector<std::string>& imagePaths,
                                          size_t* numPromptToken, bool lastLogits) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);

    // Get target consumer buffer
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llavaRuntime->dlaExecutors.front());
    const auto targetBuffer = firstExecutor->getInputBuffer();
    const auto targetSize = firstExecutor->getModelInputSizeBytes();

    // Prepare information for embedding producers
    const auto imageTokenSize = llavaRuntime->options.imageTokenSize;
    const auto singleEmbSize = llavaRuntime->tokenEmbLut->getEmbSizeBytes();

    auto isImageToken = [&tokens](const auto start, const auto end) {
        return (end - start == 1) && (tokens[start] == kImagePlaceholderToken);
    };

    auto loadImgEmb = [runtime](const std::string& imagePath) {
        int imageSizeBytes = 0;
        const auto image = clip_preprocess(imagePath, imageSizeBytes, kImgSize, kCropSize, kScale);
        return neuron_llava_get_clip_embedding(runtime, image.data, imageSizeBytes);
    };

    // Initialize the embedding producers
    const auto subtokenIntervals = subtoken_delimit(tokens, kImagePlaceholderToken, true);
    const auto numPromptSections = subtokenIntervals.size();

    std::vector<std::unique_ptr<EmbeddingProducer>> embProducerQueue;
    embProducerQueue.reserve(numPromptSections);

    *numPromptToken = 0; // Reset

    size_t imageIdx = 0;
    for (const auto& [start, end] : subtokenIntervals) {
        std::unique_ptr<EmbeddingProducer> curEmbProducer;
        if (isImageToken(start, end)) { // Image token
            CHECK_LT(imageIdx, imagePaths.size())
                << "Detected more image tokens than the number of given images.";
            curEmbProducer = std::make_unique<ImageEmbeddingProducer>(
                imagePaths[imageIdx++], imageTokenSize, loadImgEmb, singleEmbSize);
            *numPromptToken += imageTokenSize;
        } else { // Text token
            const auto subTokens = std::vector(tokens.begin() + start, tokens.begin() + end);
            curEmbProducer = std::make_unique<TextEmbeddingProducer>(
                subTokens, llavaRuntime->tokenEmbLut, singleEmbSize);
            *numPromptToken += subTokens.size();
        }
        DCHECK(!curEmbProducer->isEmpty());
        curEmbProducer->setConsumer(targetBuffer, targetSize);
        embProducerQueue.emplace_back(std::move(curEmbProducer));
    }
    const auto& imageTokenCount = imageIdx; // For readability in logging
    CHECK_EQ(imageTokenCount, imagePaths.size())
        << "The number of image tokens in the prompt does not match then number of given images.";

    // Begin consuming the prompt chunk by chunk
    auto curEmbProdIt = embProducerQueue.begin();
    auto hasProducer = [&]() { return curEmbProdIt != embProducerQueue.end(); };
    void* logitsBuffer = nullptr;
    const auto modelTokenSize = firstExecutor->getModelNumInputToken();
    const auto padSize = modelTokenSize - (*numPromptToken % modelTokenSize);

    auto getLeftPadding = [&] {
        if (kAllowLeftPadding && firstExecutor->getTokenIndex() == 0)
            return padSize;
        return 0UL;
    };

    while (hasProducer()) {
        // Fill modelTokenSize number of embeddings, or break if no embedding left to consume
        const auto leftPadSize = getLeftPadding();
        size_t demandRemain = modelTokenSize - leftPadSize;
        while (demandRemain > 0 && hasProducer()) {
            const auto numProduced = (*curEmbProdIt)->produceEmbedding(demandRemain);
            DCHECK_LE(numProduced, demandRemain);
            demandRemain -= numProduced;
            if ((*curEmbProdIt)->isEmpty()) {
                ++curEmbProdIt; // Move to the next producer
            }
        }
        const auto rightPadSize = demandRemain;
        logitsBuffer =
            neuron_llava_inference_once(runtime, leftPadSize, rightPadSize, nullptr, lastLogits);
    }
    return logitsBuffer;
}

size_t LLM_API neuron_llava_get_token_index(void* runtime) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llavaRuntime->dlaExecutors.front());
    return firstExecutor->getTokenIndex();
}

void* LLM_API neuron_llava_get_text_embedding(void* runtime,
                                              const std::vector<TokenType>& inputTokens,
                                              void* inputTextEmbCopy) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);
    const auto firstExecutor = static_cast<LlamaDlaExecutor*>(llavaRuntime->dlaExecutors.front());
    const auto modelNumInputToken = firstExecutor->getModelNumInputToken();

    // Error checking
    if (inputTokens.size() > modelNumInputToken) {
        LOG(FATAL) << "The required input token length (" << inputTokens.size() << ") "
                   << "exceeds what the model can take in (" << modelNumInputToken << ")";
    }
    llavaRuntime->tokenEmbLut->lookupEmbedding(inputTokens);

    const auto inputEmbBuffer = firstExecutor->getInputBuffer();
    const auto perTokenEmbSizeBytes = firstExecutor->getModelInputSizeBytes() / modelNumInputToken;
    const auto textEmbSizeBytes = perTokenEmbSizeBytes * inputTokens.size();

    if (inputTextEmbCopy != nullptr) {
        std::memcpy(inputTextEmbCopy, inputEmbBuffer, textEmbSizeBytes);
        return inputTextEmbCopy;
    }
    return inputEmbBuffer;
}

void* LLM_API neuron_llava_get_clip_embedding(void* runtime, void* imageBuffer,
                                              const size_t imageBufferSize) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);

    // assume image is already preprocessed
    Timer patchEmbTimer, clipDLATimer, quantTimer;
    patchEmbTimer.start();
    llavaRuntime->clipPatchEmbExecutor->runInference(imageBuffer, imageBufferSize);
    double patchEmbTimeTaken = patchEmbTimer.reset();
    LOG(INFO) << "Patch embedding takes: " << patchEmbTimeTaken << "s";

    clipDLATimer.start();
    llavaRuntime->clipExecutor->runInference();
    double clipDLATimeTaken = clipDLATimer.reset();
    LOG(INFO) << "Done CLIP dla inference in: " << clipDLATimeTaken << "s";
    const auto clipEmbBuffer = llavaRuntime->clipExecutor->getOutputBuffer();

    return clipEmbBuffer;
}

size_t LLM_API neuron_llava_get_input_emb_size_bytes(void* runtime) {
    auto llavaRuntime = reinterpret_cast<LlavaRuntime*>(runtime);
    return llavaRuntime->dlaExecutors.front()->getModelInputSizeBytes();
}