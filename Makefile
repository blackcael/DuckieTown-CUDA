# ========= Project Settings =========
TARGET    := duckietown_img

BASE_DIR  := cuda_img_processing
SRC_DIR   := $(BASE_DIR)/src
INC_DIR   := $(BASE_DIR)/include
IMG_DIR   := $(BASE_DIR)/sample_images
BUILD_DIR := $(BASE_DIR)/build

# ========= Compiler =========
NVCC := nvcc

# ========= GPU Architecture =========
ARCH := -gencode arch=compute_86,code=sm_86

# ========= Flags =========
NVCCFLAGS := -std=c++17 -O2 $(ARCH) -I$(INC_DIR) -rdc=true \
             -Xcompiler -Wall,-Wextra

CFLAGS := -std=c11 -O2 -Wall -Wextra -I$(INC_DIR)

# ========= Source Files =========
CU_SOURCES  := $(SRC_DIR)/color_filter_kernel.cu $(SRC_DIR)/gaussian_blur_kernel.cu $(SRC_DIR)/sobel_filter_kernel.cu $(SRC_DIR)/NMS_kernel.cu $(SRC_DIR)/bitwiseAND_kernel.cu $(SRC_DIR)/dilate_kernel.cu $(SRC_DIR)/erode_kernel.cu $(SRC_DIR)/timing.cu $(SRC_DIR)/main.cu
C_SOURCES   := $(SRC_DIR)/image_utils.c

OBJ_CU      := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SOURCES))
OBJ_C       := $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(C_SOURCES))

OBJECTS     := $(OBJ_CU) $(OBJ_C)

# ========= Default Target =========
.PHONY: all
all: $(TARGET)

# ========= Build Rules =========
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# CUDA compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# C compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	gcc $(CFLAGS) -c $< -o $@

# Link everything with nvcc
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# ========= Run with Image Argument =========
# Usage:
#   make run           -> uses image0.jpg
#   make run IMG=3     -> uses image3.jpg
IMG ?= 0

.PHONY: run
run: $(TARGET)
	@rm -f $(BASE_DIR)/cuda_timing.log;
	@if [ "$(IMG)" -ge 0 ] && [ "$(IMG)" -le 4 ]; then \
	    echo "Running with $(IMG_DIR)/image$(IMG).jpg"; \
	    ./$(TARGET) "$(IMG_DIR)/image$(IMG).jpg" >> $(BASE_DIR)/cuda_timing.log; \
	else \
	    echo "ERROR: IMG must be between 0 and 4 (got $(IMG))"; \
	    exit 1; \
	fi

# ========= Run ALL images 0–4 =========
.PHONY: run_all
run_all: $(TARGET)
	@if [ "$(APPEND)" != "1" ]; then rm -f $(BASE_DIR)/cuda_timing.log; fi
	@for i in 0 1 2 3 4; do \
	    echo "=== Running image $$i ==="; \
	    ./$(TARGET) "$(IMG_DIR)/image$$i.jpg" >> $(BASE_DIR)/cuda_timing.log; \
	    echo ""; \
	done

# ========= Run ALL images 0–4 100 Times =========
.PHONY: run_100
run_100:
	@rm -f $(BASE_DIR)/cuda_timing.log;
	@for n in $(shell seq 1 100); do \
	    echo "=== Batch $$n ===" >> $(BASE_DIR)/cuda_timing.log; \
	    $(MAKE) APPEND=1 run_all; \
	    echo "" >> $(BASE_DIR)/cuda_timing.log; \
	done

# ========= Clean =========
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)
