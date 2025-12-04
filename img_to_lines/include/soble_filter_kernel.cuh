#ifndef SOBLE_FILTER_KERNEL_H
#define SOBLE_FILTER_KERNEL_H

// includes
#include <stdint.h>

// PARAMS


#define SOBLE_MASK_SIZE 3
#define SOBLE_MASK_LENGTH SOBLE_MASK_SIZE * SOBLE_MASK_SIZE
#define SOBLE_MASK_ARRAY_X {-1, 0, 1, -2, 0, 2, -1, 0, 1}
#define SOBLE_MASK_ARRAY_Y {1, 2, 1, 0, 0, 0, -1, -2, -1}

#define CANNY_THRESH_HIGH 100
#define CANNY_THRESH_LOW 0


__global__ void img_to_lines_kernel(
    char* pixel_array, 
    int image_height, 
    int image_width,
    float gaussian_denominator,
    char* filter_ws,
    float* blur_ws,
    float* mag2_ws,
    char* yellow_out,
    char* white_out
);

#endif