#include "executor/allocator.h"
#include "common/logging.h"

#include <sys/mman.h>
#include <unistd.h>

DmaBufferAllocator::DmaBufferAllocator() {
    mDmabufHeapAllocator = fnCreateDmabufHeapBufferAllocator();
}

DmaBufferAllocator::~DmaBufferAllocator() {
    fnFreeDmabufHeapBufferAllocator(mDmabufHeapAllocator);
    mDmabufHeapAllocator = nullptr;
}

bool DmaBufferAllocator::allocateMemory(IOBuffer& ioBuffer) {
    if (mDmabufHeapAllocator == nullptr) {
        LOG(ERROR) << "Trying to allocate DMA Buffer without initializing dmabuf library.";
        return false;
    }
    const auto& size = ioBuffer.sizeBytes;
    int fd = fnDmabufHeapAlloc(mDmabufHeapAllocator, "mtk_mm-uncached", size, 0, 0);
    void* buffer_addr = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (buffer_addr == MAP_FAILED) {
        LOG(ERROR) << "(neuron) mmap failed!";
        return false;
    }
    ioBuffer.fd = fd;
    ioBuffer.buffer = buffer_addr;
    return true;
}

bool DmaBufferAllocator::releaseMemory(IOBuffer& ioBuffer) {
    if (mDmabufHeapAllocator == nullptr) {
        LOG(ERROR) << "Trying to release DMA Buffer without initializing dmabuf library.";
        return false;
    }
    if ((void*)::munmap(ioBuffer.buffer, ioBuffer.sizeBytes) == MAP_FAILED) {
        LOG(ERROR) << "(neuron) munmap failed!";
        return false;
    }
    if (close(ioBuffer.fd) != 0) {
        return false;
    }
    ioBuffer.buffer = nullptr;
    ioBuffer.sizeBytes = 0;
    ioBuffer.fd = -1;
    return true;
}
