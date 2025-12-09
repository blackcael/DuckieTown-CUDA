#ifndef COLOR_FILTER_KERNEL_H
#define COLOR_FILTER_KERNEL_H

// includes
#include <stdint.h>

// PARAMS
#define YELLOW_UPPER_THRESH_H 44 //28
#define YELLOW_UPPER_THRESH_S 179 //180
#define YELLOW_UPPER_THRESH_V 254 //255

#define YELLOW_LOWER_THRESH_H 16 //12
#define YELLOW_LOWER_THRESH_S 51 //50
#define YELLOW_LOWER_THRESH_V 91 //140

#define WHITE_UPPER_THRESH_H 170 //150
#define WHTIE_UPPER_THRESH_S 150 //100
#define WHITE_UPPER_THRESH_V 255 //255

#define WHITE_LOWER_THRESH_H 70 //65
#define WHTIE_LOWER_THRESH_S 0 //0 // 50 is too high
#define WHITE_LOWER_THRESH_V 150 //145

__global__ void color_filter_kernel(
    unsigned char* pixel_array, 
    int image_height, 
    int image_width,
    unsigned char* yellow_out,
    unsigned char* white_out,
    unsigned char* gray_scale_out
);

#endif