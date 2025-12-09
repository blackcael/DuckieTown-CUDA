#ifndef BITWISEAND_KERNEL_H
#define BITWISEAND_KERNEL_H

// includes
#include <stdint.h>


// PARAMS
__global__ void bitwiseAND_kernel(
    unsigned char* yellow_pixels_in,
    unsigned char* white_pixels_in,
    unsigned char* edge_mask_in,
    int image_height,    
    int image_width,
    unsigned char* yellow_edge_out,
    unsigned char* white_edge_out
);

#endif