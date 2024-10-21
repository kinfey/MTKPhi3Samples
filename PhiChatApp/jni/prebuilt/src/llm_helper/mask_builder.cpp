#include "common/logging.h"
#include "llm_helper/include/mask_builder.h"

// Define mask values for different types
template <typename T>
struct MaskVal;

#define __DECL_MASK__(TYPE, TRUE_VAL, FALSE_VAL) \
template <>                                      \
struct MaskVal<TYPE> {                           \
    static constexpr TYPE kTrue = TRUE_VAL;      \
    static constexpr TYPE kFalse = FALSE_VAL;    \
};

__DECL_MASK__(bool, true, false)
__DECL_MASK__(int16_t, 0, -32768)
__DECL_MASK__(__fp16,  0, -100)
__DECL_MASK__(float,   0, -100)
#undef __DECL_MASK__

MaskBuilder::MaskBuilder(void* maskBuffer, const size_t maskSizeBytes, const LLMType maskType,
                         const size_t cacheLength)
    : mMaskBuffer(maskBuffer), mMaskSizeBytes(maskSizeBytes), kMaskType(maskType),
      kMaskTypeSize(getLLMTypeSize(maskType)), kCacheLength(cacheLength) {}

MaskBuilder::~MaskBuilder() {}

void MaskBuilder::updateMaskSize(const size_t sizeBytes) {
    mMaskSizeBytes = sizeBytes;
}

void MaskBuilder::markMaskDirty() {
    mIsMaskUpdatable = false;
}

template <typename MaskType>
void MaskBuilder::buildMask(const size_t tokenBatchSize, const size_t numSeenToken) {
    constexpr auto maskTrue = MaskVal<MaskType>::kTrue;
    constexpr auto maskFalse = MaskVal<MaskType>::kFalse;
    const size_t maskLength = kCacheLength + tokenBatchSize;

    // The mask is a combination (concat) of input cache mask and attention mask
    const size_t startTrueIdx = kCacheLength - std::min(kCacheLength, numSeenToken);

    const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

    const size_t expectedMaskSizeBytes = tokenBatchSize * maskLength * kMaskTypeSize;
    // Use '<' instead of '!=' because mMaskSizeBytes may be padded by compiler to fit HW
    if (mMaskSizeBytes < expectedMaskSizeBytes) {
        LOG(WARN) << "Model input mask size (" << mMaskSizeBytes << ") < mask size to be built "
                     "(" << expectedMaskSizeBytes << "). Please ensure your model options are set "
                     "correctly.";
    }
    const bool isMedusaTreeAttn = !mMedusaTreeMask.empty();

    if (isMedusaTreeAttn) {
        DCHECK_EQ(mLeftPadLength, 0)
            << "For medusa inference, tree-candidate length must align with genTokenBatchSize.";
        DCHECK_EQ(mRightPadLength, 0)
            << "For medusa inference, tree-candidate length must align with genTokenBatchSize.";
    }

    // There are tokenBatchSize number of rows
    for (size_t inTokIdx = 0; inTokIdx < tokenBatchSize; inTokIdx++) {
        const auto& rowIdx = inTokIdx; // For clarity
        auto curMaskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer) + rowIdx * rowSize;
        size_t i = 0; // Buffer write index

        // Set the (rectangle) input cache mask
        while (i < startTrueIdx) curMaskBuffer[i++] = maskFalse;
        while (i < kCacheLength) curMaskBuffer[i++] = maskTrue;

        if (!isMedusaTreeAttn) {
            // Set the (triangle) attention mask
            const size_t attnTrueCount = inTokIdx + 1;
            for (size_t counter = 0; counter < attnTrueCount; counter++) {
                curMaskBuffer[i++] = maskTrue;
            }
            // Fill the remaining with False
            while (i < maskLength) curMaskBuffer[i++] = maskFalse;
        } else {
            // Medusa mask
            for (const auto medusaMaskVal : mMedusaTreeMask[rowIdx]) {
                if (medusaMaskVal == 1)
                    curMaskBuffer[i++] = maskTrue;
                else
                    curMaskBuffer[i++] = maskFalse;
            }
            DCHECK_EQ(i, maskLength);
        }
    }

    // Modify mask for padding if needed. Mask is not updatable if modified for padding.
    mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(tokenBatchSize);
}

template <typename MaskType>
void MaskBuilder::updateMask(const size_t tokenBatchSize, const size_t numSeenToken,
                             const size_t length) {
    if (!mIsMaskUpdatable) {
        buildMask<MaskType>(tokenBatchSize, numSeenToken);
        return;
    }

    // The mask is a combination (concat) of input cache mask and attention mask
    auto maskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);

    const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

    // Only modify the left rectangle part
    const size_t startTrueOffset = kCacheLength - std::min(kCacheLength, numSeenToken);
    for (size_t inTokIdx = 0; inTokIdx < tokenBatchSize; inTokIdx++) {
        const auto& rowIdx = inTokIdx; // For clarity
        auto curMaskBuffer = maskBuffer + rowIdx * rowSize + startTrueOffset;
        const size_t trueCount = std::min(length, numSeenToken); // Can only True for seen token
        std::fill(curMaskBuffer, curMaskBuffer + trueCount, MaskVal<MaskType>::kTrue);
    }
    // Modify mask for padding if needed. Mask is not updatable if modified for padding.
    mIsMaskUpdatable = !adjustMaskForPadding<MaskType>(tokenBatchSize);
}

void MaskBuilder::buildMask(const size_t tokenBatchSize, const size_t numSeenToken) {
    switch (kMaskType) {
        case LLMType::INT16:
            buildMask<int16_t>(tokenBatchSize, numSeenToken);
            return;
        case LLMType::FP16:
            buildMask<__fp16>(tokenBatchSize, numSeenToken);
            return;
        case LLMType::FP32:
            buildMask<float>(tokenBatchSize, numSeenToken);
            return;
        default:
            break;
    }
    LOG(FATAL) << "Attempting to build mask with type " << getLLMTypeName(kMaskType) << ". "
               << "Supported types are INT16, FP16, FP32.";
}

void MaskBuilder::updateMask(const size_t tokenBatchSize, const size_t numSeenToken,
                             const size_t length) {
    switch (kMaskType) {
        case LLMType::INT16:
            updateMask<int16_t>(tokenBatchSize, numSeenToken, length);
            return;
        case LLMType::FP16:
            updateMask<__fp16>(tokenBatchSize, numSeenToken, length);
            return;
        case LLMType::FP32:
            updateMask<float>(tokenBatchSize, numSeenToken, length);
            return;
        default:
            break;
    }
    LOG(FATAL) << "Attempting to update with an unsupported mask type. "
               << "Supported types are INT16, FP16, FP32.";
}

void MaskBuilder::notifyLeftPadding(const size_t padLength) {
    CHECK_EQ(mRightPadLength, 0) << "Attempting to set left pad after right pad has been set.";
    if (mLeftPadLength > 0) {
        LOG(WARN) << "Calling notifyLeftPadding() multiple times before building/updating mask.";
    }
    mLeftPadLength = padLength;
}

void MaskBuilder::notifyRightPadding(const size_t padLength) {
    CHECK_EQ(mLeftPadLength, 0) << "Attempting to set right pad after left pad has been set.";
    if (mRightPadLength > 0) {
        LOG(WARN) << "Calling notifyRightPadding() multiple times before building/updating mask.";
    }
    mRightPadLength = padLength;
}

template <typename MaskType>
bool MaskBuilder::adjustMaskForPadding(const size_t tokenBatchSize) {
    if (mLeftPadLength + mRightPadLength == 0) {
        return false; // No need to modify mask since no padding
    }
    DCHECK(mLeftPadLength == 0 || mRightPadLength == 0)
        << "Only allow setting either left or right pad";
    constexpr auto maskFalse = MaskVal<MaskType>::kFalse;
    const size_t maskLength = kCacheLength + tokenBatchSize;

    // The mask is a combination (concat) of input cache mask and attention mask
    auto maskBuffer = reinterpret_cast<MaskType*>(mMaskBuffer);

    const size_t rowSize = mMaskSizeBytes / tokenBatchSize / kMaskTypeSize;

    if (mLeftPadLength > 0) {
        // Mask the padded rows
        for (size_t inTokIdx = 0; inTokIdx < mLeftPadLength; inTokIdx++) {
            auto curMaskBuffer = maskBuffer + inTokIdx * rowSize;
            std::fill(curMaskBuffer, curMaskBuffer + maskLength, maskFalse);
        }
        // Mask the padded attention region
        for (size_t inTokIdx = mLeftPadLength; inTokIdx < tokenBatchSize; inTokIdx++) {
            auto curMaskBuffer = maskBuffer + inTokIdx * rowSize + kCacheLength;
            // Anything from inTokIdx + 1 onwards is already False, so can skip them.
            const size_t maskPadCount = std::min(mLeftPadLength, inTokIdx + 1);
            std::fill(curMaskBuffer, curMaskBuffer + maskPadCount, maskFalse);
        }
        mLeftPadLength = 0; // Reset pad length
    } else if (mRightPadLength > 0) {
        // Mask the padded rows
        const auto startIdx = tokenBatchSize - mRightPadLength;
        for (size_t inTokIdx = startIdx; inTokIdx < tokenBatchSize; inTokIdx++) {
            auto curMaskBuffer = maskBuffer + inTokIdx * rowSize;
            std::fill(curMaskBuffer, curMaskBuffer + maskLength, maskFalse);
        }
        mRightPadLength = 0; // Reset pad length
    }
    return true; // Mask is modified for padding
}

void MaskBuilder::setMedusaTreeMask(const std::vector<std::vector<int>>& mask) {
    mMedusaTreeMask = mask;
}

void MaskBuilder::resetMedusaTreeMask() {
    mMedusaTreeMask.clear();
}