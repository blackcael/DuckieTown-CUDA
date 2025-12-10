#ifndef RBG_TO_GRAYSCALE_KERNEL_H
#define RBG_TO_GRAYSCALE_KERNEL_H

// includes
#include <stdint.h>

// PARAMS (not used in program, but used in the LUT)
#define NUM_CHANNELS 3
#define R_WEIGHT 5
#define G_WEIGHT 9
#define B_WEIGHT 2

__global__ void rgb_to_grayscale_kernel(
    unsigned char* rgb_in, 
    int image_height, 
    int image_width,
    unsigned char* gray_scale_out
);

#endif