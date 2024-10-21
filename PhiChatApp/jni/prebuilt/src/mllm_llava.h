#pragma once

#include "llm_llama.h"

#include <iostream>
#include <string>
#include <vector>

/* LLaVA HyperParameter Settings */
constexpr int32_t kImagePlaceholderToken = -200; // Placeholder image token

typedef struct LlavaRuntimeOptions : public LlamaRuntimeOptions {
    std::string clipPath;
    std::string patchEmbPath;
    size_t imageTokenSize = 576;
} LlavaRuntimeOptions;

bool neuron_llava_init(void** runtime, const LlamaModelOptions& modelOptions,
                       const LlavaRuntimeOptions& runtimeOptions);

void* neuron_llava_inference_once(void* runtime, const size_t leftPadSize = 0,
                                  const size_t rightPadSize = 0, const void* inputEmb = nullptr,
                                  bool lastLogits = true);

void* neuron_llava_consume_prompt(void* runtime, const std::vector<Tokenizer::TokenType>& tokens,
                                  const std::vector<std::string>& imagePaths,
                                  size_t* numPromptToken, bool lastLogits = true);

size_t neuron_llava_get_token_index(void* runtime);

void* neuron_llava_get_text_embedding(void* runtime,
                                      const std::vector<Tokenizer::TokenType>& inputTokens,
                                      void* inputTextEmbCopy = nullptr);

void* neuron_llava_get_clip_embedding(void* runtime, void* imageBuffer,
                                      const size_t imageBufferSize);

void neuron_llava_release(void* runtime);

size_t neuron_llava_get_input_emb_size_bytes(void* runtime);