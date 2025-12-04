#ifndef SOBEL_FILTER_KERNEL_H
#define SOBEL_FILTER_KERNEL_H

// includes
#include <stdint.h>
#include <math.h>
// PARAMS
#define SOBEL_MASK_SIZE 3
#define SOBEL_MASK_LENGTH SOBEL_MASK_SIZE * SOBEL_MASK_SIZE
#define SOBEL_MASK_ARRAY_X {-1, 0, 1, -2, 0, 2, -1, 0, 1}
#define SOBEL_MASK_ARRAY_Y {1, 2, 1, 0, 0, 0, -1, -2, -1}

#define CANNY_THRESH_HIGH 10
#define CANNY_THRESH_LOW 0

#ifndef M_PI
#define M_PI 3.14159
#endif

__global__ void sobel_filter_kernel(
    unsigned char* blurred_pixels_in, 
    int image_height, 
    int image_width,
    float* mag2_ws,
    unsigned char* magnitude2_ws
);

#endif