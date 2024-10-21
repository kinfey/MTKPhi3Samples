LOCAL_PATH := $(call my-dir)
USER_LOCAL_C_INCLUDES := $(LOCAL_C_INCLUDES)

include $(CLEAR_VARS)
LOCAL_MODULE := main_llava
LOCAL_SRC_FILES := main_llava.cpp
LOCAL_STATIC_LIBRARIES += utils
LOCAL_SHARED_LIBRARIES += llm_prebuilt
LOCAL_SHARED_LIBRARIES += llava_prebuilt yaml_cpp common tokenizer
LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)

LOCAL_LDLIBS := -llog

include $(BUILD_EXECUTABLE)

LOCAL_C_INCLUDES := $(USER_LOCAL_C_INCLUDES)