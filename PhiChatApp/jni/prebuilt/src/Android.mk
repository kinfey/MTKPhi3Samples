ROOT_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_C_INCLUDES += $(ROOT_PATH)

# First include prebuilt libcommon
include $(CLEAR_VARS)
LOCAL_MODULE := common
LOCAL_PATH := $(ROOT_PATH)
LOCAL_SRC_FILES := ../libcommon.so
LOCAL_EXPORT_C_INCLUDES := $(ROOT_PATH)
include $(PREBUILT_SHARED_LIBRARY)

# Then build the necessary components
RUNTIME_ROOT := $(ROOT_PATH)/runtime
include $(RUNTIME_ROOT)/Android.mk

EXECUTOR_ROOT := $(ROOT_PATH)/executor
include $(EXECUTOR_ROOT)/Android.mk

LLM_HELPER_ROOT := $(ROOT_PATH)/llm_helper
include $(LLM_HELPER_ROOT)/Android.mk

# Finally build the .so
include $(CLEAR_VARS)
LOCAL_PATH := $(ROOT_PATH)
LOCAL_MODULE := llm_llama
LOCAL_SRC_FILES := llm_llama.cpp
LOCAL_STATIC_LIBRARIES += runtime executor llm_helper
LOCAL_SHARED_LIBRARIES += common
LOCAL_LDLIBS := -llog
LOCAL_C_INCLUDES += $(ROOT_PATH)
include $(BUILD_SHARED_LIBRARY)