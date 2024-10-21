#include "executor/llama_medusa_executor.h"

#include "common/logging.h"

void LlamaMedusaExecutor::setPosEmbed(const size_t tokenIndex) {
    // Cut the array from master
    if (tokenIndex >= kMaxTokenLength) {
        LOG(FATAL) << "Attempting to set rotaty embedding using index exceeding the supported "
                      "max token length (" << kMaxTokenLength << ")";
    }
    START_TIMER
    const auto& rotEmbInputIdxes = getRotEmbInputIdxs();
    DCHECK_EQ(rotEmbInputIdxes.size(), kRotEmbInputCount);

    auto getRotEmbInputs = [&]() {
        std::vector<void*> rotEmbInputs(kRotEmbInputCount);
        for (size_t i = 0; i < kRotEmbInputCount; i++)
            rotEmbInputs[i] = this->getInputBuffer(rotEmbInputIdxes[i]);
        return rotEmbInputs;
    };

    const bool isMedusaTreeAttn = !mMedusaTreePositions.empty();

    if (isMedusaTreeAttn) {
        CHECK_EQ(mMedusaTreePositions.size(), mModelNumInputToken)
            << "Medusa tree attention is not set.";
        DCHECK_EQ(getLeftPadding(), 0);
        DCHECK_EQ(getRightPadding(), 0);
        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), tokenIndex, mMedusaTreePositions);
    } else {
        mRotEmbMasterLut->setEmbed(getRotEmbInputs(), tokenIndex, mModelNumInputToken,
                                   getLeftPadding(), getRightPadding());
    }
    LOG_LATENCY
    LOG_DONE
}

void LlamaMedusaExecutor::resetTokenIndex() {
    LlamaExecutorBase::resetTokenIndex();
    resetMedusaTreeAttn();
}

void LlamaMedusaExecutor::setMedusaTreeAttn(const std::vector<std::vector<int>>& mask,
                                            const std::vector<size_t>& positions) {
    mMedusaTreePositions = positions;
    mMaskBuilder->setMedusaTreeMask(mask);
}

void LlamaMedusaExecutor::resetMedusaTreeAttn() {
    mMedusaTreePositions.clear();
    mMaskBuilder->resetMedusaTreeMask();
}

void LlamaMedusaExecutor::rollbackTreeCache(const std::vector<size_t>& acceptedIndices) {
    size_t firstNonSkipIdx = 0;
    for (const size_t tokenIdx : acceptedIndices) {
        if (tokenIdx == firstNonSkipIdx) {
            firstNonSkipIdx++;
        } else {
            break;
        }
    }
    if (firstNonSkipIdx == acceptedIndices.size()) {
        return; // do nothing
    }
    START_TIMER

    // View cache buffer of shape [..., kCacheLength, ...] as:
    //   [numRows, (kCacheLength, strideSizeBytes)]
    //    <----->  <----------------------------->
    //      row                   col

    const size_t strideSizeBytes = getCacheStrideSize();
    const size_t rowSize = kCacheLength * strideSizeBytes;

    auto cacheBuffers = getCacheBuffers();

    size_t cacheCounter = 0;
    for (auto cacheBuffer : cacheBuffers) {
        const size_t numRows = getCacheNumRows(cacheCounter++);
        for (size_t rowIdx = 0; rowIdx < numRows; rowIdx++) {
            auto cacheBufRow = cacheBuffer + rowIdx * rowSize; // Pointer pointing to start of row
            size_t dstTokenIdx = kCacheLength - mModelNumInputToken + firstNonSkipIdx;
            for (size_t i = firstNonSkipIdx; i < acceptedIndices.size(); i++) {
                size_t tokenIdx = acceptedIndices[i];
                const size_t dstOffset = dstTokenIdx * strideSizeBytes;
                const size_t srcOffset = (kCacheLength - mModelNumInputToken + tokenIdx)
                                       * strideSizeBytes;
                std::memcpy(cacheBufRow + dstOffset, cacheBufRow + srcOffset, strideSizeBytes);
                dstTokenIdx += 1;
            }
        }
    }
    LOG_LATENCY
    LOG_DONE
}
