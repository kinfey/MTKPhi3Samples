#pragma once

#include <vector>

class RingBufferCacheContext;

class RingBufferCacheManager {
public:
    using ShapeType = std::vector<size_t>;

public:
    explicit RingBufferCacheManager();

    ~RingBufferCacheManager();

    void initialize(const std::vector<ShapeType>& cacheShapes, const size_t cacheConcatDim,
                    const size_t cacheTypeSizeBytes, const size_t initTokenIndex,
                    const size_t maxTokenLength);

    void setIoCacheBuffers(const std::vector<void*>& inputCacheRingBuffers,
                           const std::vector<void*>& outputCacheBuffers);

public:
    // Query ring buffer overhead size after initialization
    size_t getOverheadSizeBytes() const;

    // Query the current ring buffer offset in bytes
    size_t getRingOffset() const;

    // Reset ring buffer offset to 0 without modifying the cache buffers
    void resetRingOffset();

    // Advance ring buffer offset by token count
    void advanceRingOffset(const size_t tokenCount);

    // Append output cache to input cache ring bfufer
    void appendInOutCaches(const size_t tokenBatchSize, const size_t leftPadLength,
                           const size_t rightPadLength, const bool isCacheEmpty);

    // Ring buffer reset to start from top again
    void resetRingBuffer();

    // Returns true if rollback ring buffer is successful, false if otherwise.
    bool rollback(const size_t tokenCount);

private:
    void ensureInit() const;

private:
    RingBufferCacheContext* mCtx = nullptr;

    bool mIsInitialized = false;
};
