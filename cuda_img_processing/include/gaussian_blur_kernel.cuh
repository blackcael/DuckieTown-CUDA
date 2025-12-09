#ifndef GAUSSIAN_BLUR_KERNEL_H
#define GAUSSIAN_BLUR_KERNEL_H

// includes
#include <stdint.h>

// PARAMS
#define BLUR_MASK_SIZE 3
#define BLUR_MASK_LENGTH BLUR_MASK_SIZE * BLUR_MASK_SIZE
#define GAUSSIAN_BLUR_ARRAY {1.0f,2.0f,1.0f,2.0f,4.0f,2.0f,1.0f,2.0f,1.0f}
#define GAUSSIAN_DENOMINATOR 16


__global__ void gaussian_blur_kernel(
    unsigned char* gray_scale_pixels_in, 
    int image_height, 
    int image_width,
    unsigned char* blurred_pixels_out
);

#endif