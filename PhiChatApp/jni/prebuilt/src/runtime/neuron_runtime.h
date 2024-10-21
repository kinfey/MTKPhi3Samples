#pragma once

#include "runtime/api/neuron/RuntimeAPI.h"
#include "runtime/api/dmabuff/BufferAllocatorWrapper.h"

#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <dlfcn.h>

// typedef to the functions pointer signatures.
typedef int (*FnNeuronRuntime_create)(const EnvOptions* options, void** runtime);
typedef int (*FnNeuronRuntime_create_with_options)(const char* c_options, const EnvOptions* options, void** runtime);
typedef int (*FnNeuronRuntime_loadNetworkFromFile)(void* runtime, const char* pathToDlaFile);
typedef int (*FnNeuronRuntime_setInput)(void* runtime, uint64_t handle, const void* buffer, size_t length, BufferAttribute attr);
typedef int (*FnNeuronRuntime_setOutput)(void* runtime, uint64_t handle, void* buffer, size_t length, BufferAttribute attr);
typedef int (*FnNeuronRuntime_setOffsetedInput)(void* runtime, uint64_t handle, const void* buffer, size_t length, BufferAttribute attribute, size_t offset);
typedef int (*FnNeuronRuntime_setQoSOption)(void* runtime, const QoSOptions* qosOption);
typedef int (*FnNeuronRuntime_getInputSize)(void* runtime, uint64_t handle, size_t* size);
typedef int (*FnNeuronRuntime_getOutputSize)(void* runtime, uint64_t handle, size_t* size);
typedef int (*FnNeuronRuntime_getInputPaddedSize)(void* runtime, uint64_t handle, size_t* size);
typedef int (*FnNeuronRuntime_getOutputPaddedSize)(void* runtime, uint64_t handle, size_t* size);
typedef int (*FnNeuronRuntime_getInputPaddedDimensions)(void* runtime, uint64_t handle, RuntimeAPIDimensions *dims);
typedef int (*FnNeuronRuntime_getOutputPaddedDimensions)(void* runtime, uint64_t handle, RuntimeAPIDimensions* dims);
typedef int (*FnNeuronRuntime_getInputNumber)(void* runtime, size_t* size);
typedef int (*FnNeuronRuntime_getOutputNumber)(void* runtime, size_t* size);
typedef int (*FnNeuronRuntime_getProfiledQoSData)(void* runtime, ProfiledQoSData** profiledQoSData, uint8_t* execBoostValue);
typedef int (*FnNeuronRuntime_inference)(void* runtime);
typedef void (*FnNeuronRuntime_release)(void* runtime);
typedef int (*FnNeuronRuntime_getVersion)(NeuronVersion* version);

extern FnNeuronRuntime_create fnNeuronRuntime_create;
extern FnNeuronRuntime_create_with_options fnNeuronRuntime_create_with_options;
extern FnNeuronRuntime_loadNetworkFromFile fnNeuronRuntime_loadNetworkFromFile;
extern FnNeuronRuntime_setInput fnNeuronRuntime_setInput;
extern FnNeuronRuntime_setOutput fnNeuronRuntime_setOutput;
extern FnNeuronRuntime_setOffsetedInput fnNeuronRuntime_setOffsetedInput;
extern FnNeuronRuntime_setQoSOption fnNeuronRuntime_setQoSOption;
extern FnNeuronRuntime_getInputSize fnNeuronRuntime_getInputSize;
extern FnNeuronRuntime_getOutputSize fnNeuronRuntime_getOutputSize;
extern FnNeuronRuntime_getInputPaddedSize fnNeuronRuntime_getInputPaddedSize;
extern FnNeuronRuntime_getOutputPaddedSize fnNeuronRuntime_getOutputPaddedSize;
extern FnNeuronRuntime_getInputPaddedDimensions fnNeuronRuntime_getInputPaddedDimensions;
extern FnNeuronRuntime_getOutputPaddedDimensions fnNeuronRuntime_getOutputPaddedDimensions;
extern FnNeuronRuntime_getInputNumber fnNeuronRuntime_getInputNumber;
extern FnNeuronRuntime_getOutputNumber fnNeuronRuntime_getOutputNumber;
extern FnNeuronRuntime_getProfiledQoSData fnNeuronRuntime_getProfiledQoSData;
extern FnNeuronRuntime_inference fnNeuronRuntime_inference;
extern FnNeuronRuntime_release fnNeuronRuntime_release;
extern FnNeuronRuntime_getVersion fnNeuronRuntime_getVersion;

// extern FnFreeDmabufHeapBufferAllocator fnFreeDmabufHeapBufferAllocator;
// extern FnDmabufHeapAlloc fnDmabufHeapAlloc;
// extern FnCreateDmabufHeapBufferAllocator fnCreateDmabufHeapBufferAllocator;

extern bool neuron_runtime_loaded;

bool init_neuron_runtime_library();
bool init_dmabuf_library();
