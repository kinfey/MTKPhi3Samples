#ifndef DMABUFHEAP_DEF_H_
#define DMABUFHEAP_DEF_H_
#include <linux/dma-buf.h>
static const char kDmabufSystemHeapName[] = "system";
static const char kDmabufSystemUncachedHeapName[] = "system-uncached";
typedef enum {
    kSyncRead = DMA_BUF_SYNC_READ,
    kSyncWrite = DMA_BUF_SYNC_WRITE,
    kSyncReadWrite = DMA_BUF_SYNC_RW,
} SyncType;
#endif
