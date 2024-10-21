#pragma once
#include "dmabufheap-defs.h"
#include <stdbool.h>
#include <sys/types.h>


typedef struct BufferAllocator BufferAllocator;

typedef BufferAllocator* (*FnCreateDmabufHeapBufferAllocator)();
typedef void (*FnFreeDmabufHeapBufferAllocator)(BufferAllocator* buffer_allocator);
typedef int (*FnDmabufHeapAlloc)(BufferAllocator* buffer_allocator, const char* heap_name, size_t len,
                    unsigned int heap_flags, size_t legacy_align);


// static FnFreeDmabufHeapBufferAllocator fnFreeDmabufHeapBufferAllocator;
// static FnDmabufHeapAlloc fnDmabufHeapAlloc;
// static FnCreateDmabufHeapBufferAllocator fnCreateDmabufHeapBufferAllocator;

extern FnFreeDmabufHeapBufferAllocator fnFreeDmabufHeapBufferAllocator;
extern FnDmabufHeapAlloc fnDmabufHeapAlloc;
extern FnCreateDmabufHeapBufferAllocator fnCreateDmabufHeapBufferAllocator;


