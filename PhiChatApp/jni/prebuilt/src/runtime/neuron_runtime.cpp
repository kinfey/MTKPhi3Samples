#include "runtime/neuron_runtime.h"
#include "common/logging.h"

#define LOG_TAG_LOAD "neuron_runtime_library_load_func"
#define LOG_TAG_INIT "neuron_runtime_library_init"

FnNeuronRuntime_create fnNeuronRuntime_create;
FnNeuronRuntime_create_with_options fnNeuronRuntime_create_with_options;
FnNeuronRuntime_loadNetworkFromFile fnNeuronRuntime_loadNetworkFromFile;
FnNeuronRuntime_setInput fnNeuronRuntime_setInput;
FnNeuronRuntime_setOutput fnNeuronRuntime_setOutput;
FnNeuronRuntime_setOffsetedInput fnNeuronRuntime_setOffsetedInput;
FnNeuronRuntime_setQoSOption fnNeuronRuntime_setQoSOption;
FnNeuronRuntime_getInputSize fnNeuronRuntime_getInputSize;
FnNeuronRuntime_getOutputSize fnNeuronRuntime_getOutputSize;
FnNeuronRuntime_getInputPaddedSize fnNeuronRuntime_getInputPaddedSize;
FnNeuronRuntime_getOutputPaddedSize fnNeuronRuntime_getOutputPaddedSize;
FnNeuronRuntime_getInputPaddedDimensions fnNeuronRuntime_getInputPaddedDimensions;
FnNeuronRuntime_getOutputPaddedDimensions fnNeuronRuntime_getOutputPaddedDimensions;
FnNeuronRuntime_getInputNumber fnNeuronRuntime_getInputNumber;
FnNeuronRuntime_getOutputNumber fnNeuronRuntime_getOutputNumber;
FnNeuronRuntime_getProfiledQoSData fnNeuronRuntime_getProfiledQoSData;
FnNeuronRuntime_inference fnNeuronRuntime_inference;
FnNeuronRuntime_release fnNeuronRuntime_release;
FnNeuronRuntime_getVersion fnNeuronRuntime_getVersion;

FnFreeDmabufHeapBufferAllocator fnFreeDmabufHeapBufferAllocator;
FnDmabufHeapAlloc fnDmabufHeapAlloc;
FnCreateDmabufHeapBufferAllocator fnCreateDmabufHeapBufferAllocator;

bool neuron_runtime_loaded = false;
bool dmabuf_library_loaded = false;

inline void* load_func(void * handle, const char * func_name) {
    //Load the function specified by func_name, and exit if the loading is failed.
    void * func_ptr = dlsym(handle, func_name);

    if (func_name == nullptr) {
        LOG(ERROR, LOG_TAG_LOAD) << "Fail to find function: " << func_name;
    }
    else {
        LOG(DEBUG, LOG_TAG_LOAD) << "Found function: " << func_name;
    }
    return func_ptr;
}

bool init_neuron_runtime_library() {
    if (neuron_runtime_loaded) {
        LOG(DEBUG, LOG_TAG_INIT) << "Skip loading neuron runtime again.";
        return true;
    }
    LOG(DEBUG, LOG_TAG_INIT) << "Begin loading neuron runtime.";

    bool status = true;

    // Load neuron runtime
    LOG(DEBUG, LOG_TAG_INIT) << "dlopen neuron_runtime";
    void* rt_handle = dlopen("libneuron_runtime.so", RTLD_LAZY);
    if (rt_handle == nullptr) {
        LOG(ERROR, LOG_TAG_INIT) << "Failed to load neuron";
        status = false;
    } else {
        LOG(DEBUG, LOG_TAG_INIT) << "Load neuron OK";
    }
    #define LOAD(name)                                                      \
        fn##name = reinterpret_cast<Fn##name>(load_func(rt_handle, #name)); \
        if (fn##name == nullptr) {                                          \
            status = false;                                                 \
        }
    LOAD(NeuronRuntime_create)
    LOAD(NeuronRuntime_create_with_options)
    LOAD(NeuronRuntime_loadNetworkFromFile)
    LOAD(NeuronRuntime_setInput)
    LOAD(NeuronRuntime_setOutput)
    LOAD(NeuronRuntime_setOffsetedInput)
    LOAD(NeuronRuntime_setQoSOption)
    LOAD(NeuronRuntime_getInputSize)
    LOAD(NeuronRuntime_getOutputSize)
    LOAD(NeuronRuntime_getInputPaddedSize)
    LOAD(NeuronRuntime_getOutputPaddedSize)
    LOAD(NeuronRuntime_getInputPaddedDimensions)
    LOAD(NeuronRuntime_getOutputPaddedDimensions)
    LOAD(NeuronRuntime_getInputNumber)
    LOAD(NeuronRuntime_getOutputNumber)
    LOAD(NeuronRuntime_getProfiledQoSData)
    LOAD(NeuronRuntime_inference)
    LOAD(NeuronRuntime_release)
    LOAD(NeuronRuntime_getVersion)
    #undef LOAD

    neuron_runtime_loaded = true;

    return status;
}

bool init_dmabuf_library() {
    if (dmabuf_library_loaded) {
        LOG(DEBUG, LOG_TAG_INIT) << "Skip loading dmabuf library again.";
        return true;
    }
    LOG(DEBUG, LOG_TAG_INIT) << "Begin loading dmabuf library.";

    bool status = true;
    // Load libdmabuf
    LOG(DEBUG, LOG_TAG_INIT) << "dlopen dmabufflib";
    void* dma_handle = dlopen("libdmabufheap.so", RTLD_LAZY);
    if (dma_handle == nullptr) {
        LOG(ERROR, LOG_TAG_INIT) << "Failed to load dmabuf";
        status = false;
    } else {
        LOG(DEBUG, LOG_TAG_INIT) << "Load dmabuf OK";
    }
    #define LOAD(name)                                                       \
        fn##name = reinterpret_cast<Fn##name>(load_func(dma_handle, #name)); \
        if (fn##name == nullptr) {                                           \
            status = false;                                                  \
        }
    LOAD(FreeDmabufHeapBufferAllocator)
    LOAD(DmabufHeapAlloc)
    LOAD(CreateDmabufHeapBufferAllocator)
    #undef LOAD

    dmabuf_library_loaded = true;

    return status;
}