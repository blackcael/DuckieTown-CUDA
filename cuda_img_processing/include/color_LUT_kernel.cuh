#ifndef COLOR_LUT_KERNEL_H
#define COLOR_LUT_KERNEL_H

// includes
#include <stdint.h>

// PARAMS (not used in program, but used in the LUT)
#define YELLOW_UPPER_THRESH_H 44 
#define YELLOW_UPPER_THRESH_S 179 
#define YELLOW_UPPER_THRESH_V 254 

#define YELLOW_LOWER_THRESH_H 16 
#define YELLOW_LOWER_THRESH_S 51 
#define YELLOW_LOWER_THRESH_V 91 

#define WHITE_UPPER_THRESH_H 170 
#define WHTIE_UPPER_THRESH_S 150 
#define WHITE_UPPER_THRESH_V 255 

#define WHITE_LOWER_THRESH_H 70 
#define WHTIE_LOWER_THRESH_S 0 
#define WHITE_LOWER_THRESH_V 150 

__global__ void color_LUT_kernel(
    unsigned char* pixel_array, 
    int image_height, 
    int image_width,
    unsigned char* yellow_out,
    unsigned char* white_out,
);

#endif