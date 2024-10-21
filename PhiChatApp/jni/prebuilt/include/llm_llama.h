#pragma once

#include "llm_types.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

typedef struct LlamaModelOptions {
    // Sizes
    size_t promptTokenBatchSize = 1;
    size_t genTokenBatchSize    = 1;
    size_t cacheSize            = 512;
    size_t hiddenSize           = 4096;
    size_t numHead              = 32;
    size_t numLayer             = 32;
    size_t maxTokenLength       = 2048;
    size_t numMedusaHeads       = 0;

    // Rotary Embedding
    float rotEmbBase            = 10000.0;

    // Types
    LLMType modelInputType  = LLMType::INT16;
    LLMType modelOutputType = LLMType::INT16;
    LLMType cacheType       = LLMType::INT16;
    LLMType maskType        = LLMType::INT16;
    LLMType rotEmbType      = LLMType::INT16;

    // Quantization
    float embOutputQuantScale   = 0; // default 0 means not used
    float modelOutputQuantScale = 1;
} LlamaModelOptions;

using LoraKey = std::string;
using ChunkPaths = std::vector<std::string>;
using TokenSet = std::unordered_set<Tokenizer::TokenType>;

typedef struct LlamaRuntimeOptions {
    struct SpecialTokens {
        Tokenizer::TokenType bosId = 1; // Beginning of Sentence Token Id
        Tokenizer::TokenType eosId = 2; // End of Sentence Token Id
        bool addBos = false;            // Whether BoS token will be prepended during tokenization
        TokenSet stopToken;             // Inference stops once the model generates a stop token
    } specialTokens;
    std::string tokenizerRegex; // Optional
    std::vector<std::string> tokenizerPath; // Either a directory or file path(s)
    std::string tokenEmbPath;
    ChunkPaths dlaPromptPaths;
    ChunkPaths dlaGenPaths;
    std::string dlaLmHeadPath;
    std::string dlaMedusaHeadsPath;
    int startTokenIndex = 0;
    ChunkPaths cachePaths; // Each file is a concatenation of all caches in a chunk.
    ChunkPaths sharedWeightsPaths;

    LoraKey initWithLoraKey;
    size_t loraInputCount = 0; // Per DLA chunk
    std::unordered_map<LoraKey, ChunkPaths> loraWeightsPaths;
} LlamaRuntimeOptions;

bool neuron_llama_init(void** runtime, const LlamaModelOptions& modelOptions,
                       const LlamaRuntimeOptions& runtimeOptions);

void neuron_llama_release(void* runtime);

void neuron_llama_set_medusa_tree_attn(void* runtime, const std::vector<std::vector<int>>& mask,
                                       const std::vector<size_t>& positions);

void* neuron_llama_inference_once(void* runtime,
                                  const std::vector<Tokenizer::TokenType>& inputTokens,
                                  const bool lastLogits = true);

std::tuple<void*, void*>
neuron_llama_inference_once_return_hidden(void* runtime,
                                          const std::vector<Tokenizer::TokenType>& inputTokens,
                                          const bool lastLogits = true);

void* neuron_medusa_heads_inference_once(void* runtime, void* hiddenState);

void neuron_llama_swap_model(void* runtime, const size_t batchSize = 1);

void neuron_llama_apply_lora(void* runtime, const LoraKey& loraKey);

void neuron_llama_apply_lora_from_buffer(void* runtime, const std::vector<char*>& loraWeightBuffers,
                                         const std::vector<size_t>& sizes);

void neuron_llama_remove_lora(void* runtime);

void neuron_llama_get_caches(void* runtime, std::vector<std::vector<char*>>& caches,
                             size_t& byteSizePerCache);

void neuron_llama_reset(void* runtime, const bool resetCache = true);

size_t neuron_llama_get_per_token_logits_size(void* runtime);
size_t neuron_llama_get_per_token_hidden_states_size(void* runtime);

size_t neuron_llama_get_token_index(void* runtime);

void neuron_llama_rollback(void* runtime, const size_t rollbackCount);
void neuron_medusa_rollback(void* runtime, const std::vector<size_t>& acceptedIndices);
