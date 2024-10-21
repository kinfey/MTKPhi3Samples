#pragma once

#include "executor/neuron_executor.h"
#include "executor/neuron_usdk_executor.h"

#include "llm_types.h"
#include "llm_llama.h"
#include "llm_helper/include/rotary_embedding.h"
#include "llm_helper/include/mask_builder.h"
#include "llm_helper/include/lora_weights_loader.h"

#include <array>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

struct LLMRuntimeInfo {
    std::string modelPath;
    size_t batchSize = 1; // E.g. prompt model is 32, whereas generative mode is 1
    size_t cacheSize = 0; // For FMS-based dynamic shape cache
};

using LoraWeightsPathMap = std::unordered_map<LoraKey, std::string>;


#ifdef USE_USDK_BACKEND
using ExecutorBackend = NeuronUsdkExecutor;
#else
using ExecutorBackend = NeuronExecutor;
#endif

class LlamaExecutor : public ExecutorBackend {
public:
    using ShapeValueType = std::remove_extent_t<decltype(RuntimeAPIDimensions::dimensions)>;
    using ShapeType = std::array<ShapeValueType, kDimensionSize>;

    // Dimension where the cache length canbe found from the input cache shape
    static constexpr size_t kCacheLengthDim = 2;

private:
    static constexpr size_t kUnusedSize = 0;

public:
    explicit LlamaExecutor(const std::vector<LLMRuntimeInfo>& runtimeInfos,
                           const std::string& sharedWeightsPath,
                           const size_t maxTokenLength,
                           const size_t cacheLength,
                           const size_t cacheCount,
                           const LLMType cacheType,
                           const LLMType maskType,
                           // Rotary Embedding Lut
                           const RotaryEmbeddingMasterLut* rotEmbMasterLut,
                           const size_t rotEmbInputCount,
                           // Lora
                           const LoraWeightsPathMap& loraWeightsPathMap,
                           const LoraKey& initWithLoraKey,
                           const size_t loraInputCount,
                           // Init cache paths
                           const std::string& initCachePath,
                           const size_t initTokenIndex = 0,
                           // Inputs
                           const size_t maskInputIndex = 1,
                           const size_t rotEmbInputIndex = 2)
        : ExecutorBackend(getModelPaths(runtimeInfos), sharedWeightsPath),
          kRuntimeInfos(runtimeInfos),
          // Llama specific options
          kMaxTokenLength(maxTokenLength),
          kCacheLength(cacheLength),
          kCacheCount(cacheCount),
          kCacheTypeSize(getLLMTypeSize(cacheType)),
          kMaskType(maskType),
          kMaskTypeSize(getLLMTypeSize(kMaskType)),
          kInitTokenIndex(initTokenIndex),
          kInitCachePath(initCachePath),
          mRotEmbMasterLut(rotEmbMasterLut),
          kRotEmbInputCount(rotEmbInputCount),
          // Lora Input Weights, infer the number of Lora inputs from bin header if not specified.
          kLoraWeightsPathMap(loraWeightsPathMap),
          kDefaultLoraKey(initWithLoraKey),
          kLoraInputCount(loraInputCount ? loraInputCount : getLoraInputCount(loraWeightsPathMap)),
          // Llama specific IO indexes
          kMaskInputIndex(maskInputIndex),
          kRotEmbInputIndexes(getIndexRange(rotEmbInputIndex, rotEmbInputCount)),
          kCacheInputIndexes(getIndexRange(rotEmbInputIndex + rotEmbInputCount, cacheCount)),
          kCacheOutputIndexes(getIndexRange(1, cacheCount)),
          kLoraWeightsInputIndexes(getIndexRange(kCacheInputIndexes.back() + 1, kLoraInputCount)) {}

    ~LlamaExecutor() {}

    // Initialization
    virtual void initialize() override;
    virtual void preInitBufferProcess() override;

    void setNumIOs();

    // Hot-swap to model with batchSize and cacheSize if available.
    // Returns true if swap successfully, false if otherwise.
    bool hotSwapModel(const size_t batchSize, const size_t cacheSize = kUnusedSize);

    // Caches
    virtual void initCache();
    // Get cache buffers
    void getCacheBuffersWithSize(std::vector<char*>& cacheBuffers, size_t& byteSizePerCache);

    // Token index
    virtual void resetTokenIndex();
    void setTokenIndex(const size_t index); // NOTE: Need to modify cache if token index was not 0
    void advanceTokenIndex();
    size_t getTokenIndex() const;

    // Align the model state (cache & token index) with the current input. Used for >1t model.
    // Returns the number of tokens being shifted/rolledback
    int alignInputTokens(const size_t numInputToken);

    void updatePosEmbAndMask(const size_t numInputToken = 1);

    // Padding
    void setLeftPadding(const size_t leftPadSize);
    void setRightPadding(const size_t rightPadSize);
    void paddingPostprocess(); // General padding postprocessing and will call L/R specific routine

    // Get expected input token count from the model
    const size_t getModelNumInputToken() const { return mModelNumInputToken; }
    // Get expected input token count excluding padded tokens from the model
    const size_t getValidModelNumInputToken() const { return mModelNumInputToken - getPadSize(); }

    // LoRA-as-inputs
    // Apply Lora based on predefined Lora Key. Empty key will remove Lora and use base weights only
    void applyLoraWeights(const LoraKey& loraKey = "");

    // Apply Lora based on provided Lora weights, will override/bypass any predefined Lora keys.
    void applyLoraWeights(const std::vector<char*>& loraWeights, const std::vector<size_t>& sizes);

    // Remove Lora and use base weights only
    void removeLoraWeights();

protected:
    const size_t getMaskInputIdx()                 const { return kMaskInputIndex; }
    const std::vector<size_t> getRotEmbInputIdxs() const { return kRotEmbInputIndexes; }
    const std::vector<size_t> getCacheInputIdxs()  const { return kCacheInputIndexes; }
    const std::vector<size_t> getCacheOutputIdxs() const { return kCacheOutputIndexes; }

    size_t getPadSize() const { return mCurrentPadSize; }
    size_t getLeftPadding() const;
    size_t getRightPadding() const;

    // Cache post-processing specific to left/right padding
    virtual void leftPaddingCachePostprocess();
    virtual void rightPaddingCachePostprocess();

    virtual void rollbackCache(const size_t tokenCount);

    virtual std::vector<char*> getCacheBuffers();

    // Helper functions
    size_t getCacheNumRows(const size_t index) const;

    size_t getCacheStrideSize() const;

private:
    size_t getExpectedNumInputs() const {
        return 2 + kRotEmbInputCount + kCacheCount + kLoraInputCount + this->isSharedWeightsUsed();
    }
    size_t getExpectedNumOutputs() const { return 1 + kCacheCount; }

    void initMaskBuilder();

    virtual void setPosEmbed(const size_t tokenIndex);

    virtual void linkCacheIOs();

    // Build the mapping from model info (batch size, cache size) to runtime index
    void buildRuntimeIdxMap();

    // Select the model with largest batch size with smallest cache size
    void setDefaultModel();

    static std::vector<size_t> getIndexRange(const size_t startIndex, const size_t count) {
        std::vector<size_t> indexes(count);
        size_t counter = startIndex;
        for (auto& idx : indexes) {
            idx = counter++;
        }
        return indexes;
    }

    static std::vector<std::string> getModelPaths(const std::vector<LLMRuntimeInfo>& runtimeInfos) {
        std::vector<std::string> modelPaths;
        for (const auto& runtimeInfo : runtimeInfos) {
            modelPaths.push_back(runtimeInfo.modelPath);
        }
        return modelPaths;
    }

    static size_t getLoraInputCount(const LoraWeightsPathMap& loraWeightsPathMap) {
        std::unordered_set<size_t> loraInputsCountSet;
        for (const auto& [loraKey, path] : loraWeightsPathMap) {
            const auto numLoraInputs = LoraWeightsLoader(path).getNumLoraInputs();
            LOG(DEBUG) << " Lora weights '" << loraKey << "' has " << numLoraInputs << " inputs.";
            loraInputsCountSet.insert(numLoraInputs);
        }
        if (loraInputsCountSet.size() > 1) {
            LOG(ERROR) << "Unsupported: Different Lora weight input count found across Lora "
                       << "weights bin files.";
        }
        return loraInputsCountSet.empty() ? 0 : *loraInputsCountSet.cbegin();
    }

protected:
    // The number of input tokens the the fixed-shape model takes
    size_t mModelNumInputToken = 1;

    const std::vector<LLMRuntimeInfo> kRuntimeInfos;

    // Map [batchSize][cacheSize] -> runtime index
    std::unordered_map<int, std::unordered_map<int, size_t>> mRuntimeIdxMap;

    // Cache
    std::vector<ShapeType> mCacheShapes;
    const size_t kMaxTokenLength;
    const size_t kCacheLength;
    const size_t kCacheCount;
    const size_t kCacheTypeSize; // bytes

    // Mask
    size_t mMaskLength;
    const LLMType kMaskType;
    const size_t kMaskTypeSize; // bytes

    enum class PaddingMode { LEFT, RIGHT };

    // Padding
    size_t mCurrentPadSize = 0;
    PaddingMode mPaddingMode = PaddingMode::RIGHT;

    const size_t kInitTokenIndex = 0;

    const std::string kInitCachePath;

    // Master lookup table for rotary embedding
    const RotaryEmbeddingMasterLut* mRotEmbMasterLut;
    const size_t kRotEmbInputCount;

    // Mask builder
    std::unique_ptr<MaskBuilder> mMaskBuilder;

    size_t mCurrentTokenIndex = 0; // Default init from 0, also can be numSeenToken

    // Will be set to false during init and after model swap
    bool mIsMaskUpdatable = false;

    // LoRA-as-inputs
    const LoraWeightsPathMap kLoraWeightsPathMap;
    const size_t kLoraInputCount = 0;
    const LoraKey kDefaultLoraKey;
    LoraKey mCurrentLoraKey;

    // IO Indexes
    const size_t kMaskInputIndex;
    const std::vector<size_t> kRotEmbInputIndexes;
    const std::vector<size_t> kCacheInputIndexes;
    const std::vector<size_t> kCacheOutputIndexes;
    const std::vector<size_t> kLoraWeightsInputIndexes;
};
