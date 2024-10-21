#include "llm_helper/include/lora_weights_loader.h"
#include "llm_helper/include/utils.h"
#include "common/logging.h"

#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

LoraWeightsLoader::LoraWeightsLoader(const std::string& path)
    : mFile(std::ifstream(path, std::ios::binary)), kFilePath(path) {
    if (!mFile)
        LOG(ERROR) << "Failed to load Lora weights file: " << path;
}

size_t LoraWeightsLoader::getNumLoraInputs() {
    if (!mFile)
        return 0;
    return loadHeader().numLoraInputs;
}

LoraWeightsHeader LoraWeightsLoader::loadHeader() {
    if (!mFile)
        LOG(ERROR) << "Lora weights not loaded.";

    // Load header
    LoraWeightsHeader header;
    mFile.seekg(0, std::ios_base::beg); // Seek to the top
    mFile.read((char*)&header, sizeof(LoraWeightsHeader));

    // Check version
    if (header.version > LORA_BIN_VERSION) {
        LOG(ERROR) << "Unsupported Lora bin version: " << header.version << ". "
                   << "Supported version is <= " << LORA_BIN_VERSION;
    }
    return header;
}

std::vector<LoraWeightsLoader::SizeType> LoraWeightsLoader::loadSizes() {
    // Get number of Lora input
    const auto& header = loadHeader();
    const auto numLoraInputs = header.numLoraInputs;

    constexpr auto headerSize = sizeof(LoraWeightsHeader);
    mFile.seekg(headerSize, std::ios_base::beg); // Seek past header

    // Load sizes
    std::vector<SizeType> sizes(numLoraInputs);
    mFile.read((char*)sizes.data(), sizeof(SizeType) * numLoraInputs);
    return sizes;
}

// Load Lora weights from file to targetBuffers
void LoraWeightsLoader::loadLoraWeights(const std::vector<void*>& targetBuffers,
                                        const std::vector<size_t>& targetSizes) {
    if (!mFile) {
        LOG(ERROR) << "Lora weights not loaded.";
        return;
    }

    // Check number of Lora inputs
    const auto& loraInputSizes = loadSizes();
    const auto numLoraInputs = loraInputSizes.size();
    const auto numBuffers = targetBuffers.size();
    DCHECK_EQ(numBuffers, targetSizes.size());
    CHECK_EQ(numLoraInputs, numBuffers)
        << "Mismatch number of Lora inputs: Expected " << numBuffers << " but have "
        << numLoraInputs;

    // Check Lora weights size
    constexpr auto headerSize = sizeof(LoraWeightsHeader);
    const auto sizesSectionSize = sizeof(SizeType) * numLoraInputs;
    const auto loraWeightSectionOffset = headerSize + sizesSectionSize;
    const auto loraWeightsFileSize = fs::file_size(kFilePath);
    CHECK_GE(loraWeightsFileSize, loraWeightSectionOffset);

    // Size based on actual file size
    const auto totalAvailSize = loraWeightsFileSize - loraWeightSectionOffset;
    // Size based on the sizes section of the bin file
    const auto totalExpectedSize = reduce_sum(loraInputSizes);
    // Size required according to the argument
    const auto totalRequiredSize = reduce_sum(targetSizes);
    CHECK_EQ(totalExpectedSize, totalAvailSize)
        << "Mismatch of Lora weights sizes available in the actual file (" << totalAvailSize << ") "
        << "and sizes described in the bin (" << totalExpectedSize << ").";
    CHECK_EQ(totalRequiredSize, totalAvailSize)
        << "Mismatch between Lora input buffer total size (" << totalRequiredSize << ") "
        << "and actual Lora weights size (" << totalAvailSize << ")";

    // Read Lora weights to target buffers
    mFile.seekg(loraWeightSectionOffset, std::ios_base::beg); // Seek to the start of lora weights section
    for (size_t i = 0; i < numBuffers; i++) {
        auto loraInputBuffer = reinterpret_cast<char*>(targetBuffers[i]);
        const auto expectedSize = loraInputSizes[i];
        const auto requiredSize = targetSizes[i];
        CHECK_EQ(expectedSize, requiredSize)
            << "Lora input " << i << ": Expected to read " << expectedSize << " but require "
            << requiredSize << " instead.";
        LOG(DEBUG) << "Reading " << i << "-th Lora weights of size " << requiredSize;
        mFile.read(loraInputBuffer, requiredSize);
        CHECK_EQ(mFile.gcount(), requiredSize) << "Failed reading " << i << "-th Lora weights.";
    }
}
