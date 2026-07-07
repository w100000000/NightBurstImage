# Makefile — ELF2 Pipeline ASI585MC 交叉编译
#
# 编译前设置:
#   export SDK=~/burst_image_project/tools/compile_tools/aarch64-buildroot-linux-gnu_sdk-buildroot
#   export RKNN_API=~/burst_image_project/SDK/ELF2-linux-source/external/rknpu2/runtime/Linux/librknn_api
#   export NCNN_ROOT=~/ncnn
#   export ASI_SDK=~/ASI_linux_mac_SDK_V1.41
#
# 编译:
#   make

CC  := $(SDK)/bin/aarch64-buildroot-linux-gnu-gcc
CXX := $(SDK)/bin/aarch64-buildroot-linux-gnu-g++

CFLAGS   := -O2 -Wall -pthread
CXXFLAGS := -O2 -Wall -pthread -std=c++11

INCLUDES := -I. \
            -I$(RKNN_API)/include \
            -I$(NCNN_ROOT)/src \
            -I$(NCNN_ROOT)/build/src \
            -I$(ASI_SDK)/include \
            -I$(SDK)/aarch64-buildroot-linux-gnu/sysroot/usr/include/libdrm

LIBS := -L$(RKNN_API)/aarch64 -lrknnrt \
        -L$(NCNN_ROOT)/build/src -lncnn \
        -L$(ASI_SDK)/lib/armv8 -lASICamera2 \
        -L$(HOME)/libusb_arm64/lib -lusb-1.0 \
        -ldrm \
        -lm -lpthread

LPR_DIR := ../lpr
OBJS := main.o ring_buf.o asi_capture.o npu_worker.o fb_display.o lpr_worker.o

TARGET := pipeline

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS)
	@echo "=== Built pipeline (ASI585MC) ==="
	@echo "Deploy: scp pipeline $(ASI_SDK)/lib/armv8/libASICamera2.so.1.41 root@192.168.0.232:/tmp/"

asi_capture.o: asi_capture.c pipeline_types.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

main.o: main.c pipeline_types.h
	$(CC)  $(CFLAGS)   $(INCLUDES) -c $< -o $@

ring_buf.o: ring_buf.c pipeline_types.h
	$(CC)  $(CFLAGS)   $(INCLUDES) -c $< -o $@

npu_worker.o: npu_worker.c pipeline_types.h
	$(CC)  $(CFLAGS)   $(INCLUDES) -c $< -o $@

fb_display.o: fb_display.c pipeline_types.h
	$(CC)  $(CFLAGS)   $(INCLUDES) -c $< -o $@

lpr_worker.o: $(LPR_DIR)/lpr_worker.c pipeline_types.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(LPR_DIR)/lpr_worker.c -o $@

clean:
	rm -f $(OBJS) $(TARGET)

# ── asi_snap / asi_stream: 相机测试工具 (只依赖 ASI SDK) ──
ASI_ONLY_INC  := -I$(ASI_SDK)/include
ASI_ONLY_LIBS := -L$(ASI_SDK)/lib/armv8 -lASICamera2 \
                 -L$(HOME)/libusb_arm64/lib -lusb-1.0 -lm -lpthread

asi_snap: ../tools/asi_snap.c
	$(CXX) $(CXXFLAGS) $(ASI_ONLY_INC) -o $@ $< $(ASI_ONLY_LIBS)
	@echo "=== Built asi_snap ==="

asi_stream: ../tools/asi_stream.c
	$(CXX) $(CXXFLAGS) $(ASI_ONLY_INC) -o $@ $< $(ASI_ONLY_LIBS)
	@echo "=== Built asi_stream ==="

lpr_server: ../tools/lpr_server.c
	$(CXX) $(CXXFLAGS) -o $@ $<
	@echo "=== Built lpr_server ==="

.PHONY: all clean
