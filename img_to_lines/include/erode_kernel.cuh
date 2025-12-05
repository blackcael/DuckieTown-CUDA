#ifndef ERODE_KERNEL_H
#define ERODE_KERNEL_H

// includes
#include <stdint.h>

// PARAMS
#define WHITE_DILATION_SIZE 5
#define WHITE_EROSION_SIZE 3

#define YELLOW_DILATION_SIZE 5
#define YELLOW_EROSION_SIZE 3

__global__ void erode_kernel(
    unsigned char* image_in, 
    int image_height, 
    int image_width,
    unsigned char* image_out
);

#endif