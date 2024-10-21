ROOT_PATH := $(call my-dir)

include $(CLEAR_VARS)
OPENCV_INSTALL_MODULES:=on

LOCAL_C_INCLUDES += $(ROOT_PATH)

LLM_PREBUILT_ROOT := $(ROOT_PATH)/prebuilt
include $(LLM_PREBUILT_ROOT)/Android_llava.mk

UTILS_ROOT := $(ROOT_PATH)/utils
include $(UTILS_ROOT)/Android.mk

MAIN_ROOT := $(ROOT_PATH)/main
include $(MAIN_ROOT)/Android_llava.mk