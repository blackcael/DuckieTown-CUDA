#ifndef NMS_KERNEL_H
#define NMS_KERNEL_H

// includes
#include <stdint.h>

#define CANNY_THRESH_HIGH 15000.0f
#define CANNY_THRESH_LOW 0

// PARAMS
__global__ void NMS_kernel(
    float* magnitude2_in,
    unsigned char* angle_in,
    int image_height, 
    int image_width,
    float* nms_mag_out,
    unsigned char* edge_out
);

#endif