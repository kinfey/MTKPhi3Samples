#include "llm_helper/include/token_embedding.h"

#include "llm_types.h"
#include "common/logging.h"
#include "tokenizer/tokenizer.h"

#include <string>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

TokenEmbeddingLut::TokenEmbeddingLut(const std::string& tokenEmbLutPath,
                                     const LLMType tokenEmbLutType, const size_t hiddenSize)
    : kTokenEmbLutType(tokenEmbLutType),
      kTokenEmbLutTypeSize(getLLMTypeSize(tokenEmbLutType)),
      kHiddenSize(hiddenSize),
      kLutRowSizeBytes(kHiddenSize * kTokenEmbLutTypeSize) {
    std::ifstream file(tokenEmbLutPath, std::ios::binary);
    if (!file) {
        LOG(FATAL) << "Token embedding lookup table file not found: " << tokenEmbLutPath;
    }
    LOG(DEBUG) << "Loading token embedding lookup table: " << tokenEmbLutPath;

    const size_t lutFileSize = fs::file_size(tokenEmbLutPath);

    mVocabSize = lutFileSize / hiddenSize / kTokenEmbLutTypeSize;
    LOG(DEBUG) << "TokenEmbeddingLut: Vocab size = " << mVocabSize;

    mLutBuffer = std::make_unique<char[]>(lutFileSize);

    file.read(mLutBuffer.get(), lutFileSize);
    CHECK_EQ(file.gcount(), lutFileSize);
}

void TokenEmbeddingLut::setOutput(void* buffer, const size_t size) {
    CHECK(buffer != nullptr);
    CHECK_GT(size, 0);
    mOutputBuffer = reinterpret_cast<char*>(buffer);
    mOutputBufferSize = size;
}

void TokenEmbeddingLut::lookupEmbedding(const std::vector<Tokenizer::TokenType>& tokens)  const {
    lookupEmbedding(tokens, mOutputBuffer, mOutputBufferSize);
}

void TokenEmbeddingLut::lookupEmbedding(const std::vector<Tokenizer::TokenType>& tokens,
                                        void* buffer, const size_t size) const {
    const auto numTokens = tokens.size();
    const size_t requiredOutputSize = numTokens * kHiddenSize * kTokenEmbLutTypeSize;
    if (size < requiredOutputSize) {
        LOG(ERROR) << "Token embedding buffer size (" << size << ") "
                   << "is insufficient to hold embedding for " << numTokens
                   << " tokens (requires " << requiredOutputSize << ")";
        return;
    }
    if (buffer == nullptr || size == 0) {
        LOG(ERROR) << "TokenEmbeddingLut: Output is not yet set for embedding lookup.";
        return;
    }

    const auto lutBuffer = mLutBuffer.get();
    auto outputBuffer = reinterpret_cast<char*>(buffer);
    size_t outputOffset = 0;
    for (const auto token : tokens) {
        // Copy one row from lookup table per token
        CHECK_LT(token, mVocabSize) << "Token id exceeds embedding lookup table range.";
        const auto& rowIdx = token;
        const size_t lutOffset = rowIdx * kLutRowSizeBytes;
        std::memcpy(outputBuffer + outputOffset, lutBuffer + lutOffset, kLutRowSizeBytes);
        outputOffset += kLutRowSizeBytes;
    }
}