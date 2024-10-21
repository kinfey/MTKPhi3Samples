#pragma once

#include "runtime/api/dmabuff/BufferAllocatorWrapper.h"
#include "runtime/api/neuron/NeuronAdapter.h"

struct IOBuffer {
    void* buffer = nullptr;
    int fd = -1;
    size_t sizeBytes = 0;
    size_t usedSizeBytes = 0;
    NeuronMemory* neuronMemory = nullptr;

    // Helper functions
    operator bool() const {
        return isAllocated();
    }
    bool operator!() const {
        return !isAllocated();
    }
    bool isAllocated() const {
        return buffer != nullptr && sizeBytes != 0;
    }
};

class Allocator {
public:
    virtual ~Allocator() = default;
    virtual bool allocateMemory(IOBuffer& ioBuffer) = 0;
    virtual bool releaseMemory(IOBuffer& ioBuffer) = 0;
};

class DmaBufferAllocator : public Allocator {
public:
    explicit DmaBufferAllocator();
    virtual ~DmaBufferAllocator();

    virtual bool allocateMemory(IOBuffer& ioBuffer) override;
    virtual bool releaseMemory(IOBuffer& ioBuffer) override;

private:
    BufferAllocator* mDmabufHeapAllocator = nullptr;
};