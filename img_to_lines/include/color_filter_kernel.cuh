#ifndef COLOR_FILTER_KERNEL_H
#define COLOR_FILTER_KERNEL_H

// includes
#include "image_utils.h"
#include <stdint.h>

// PARAMS
#define YELLOW_UPPER_THRESH_H 45 //28
#define YELLOW_UPPER_THRESH_S 180 //180
#define YELLOW_UPPER_THRESH_V 255 //255

#define YELLOW_LOWER_THRESH_H 15 //12
#define YELLOW_LOWER_THRESH_S 50 //50
#define YELLOW_LOWER_THRESH_V 90 //140

#define WHITE_UPPER_THRESH_H 170 //150
#define WHTIE_UPPER_THRESH_S 100 //100
#define WHITE_UPPER_THRESH_V 255 //255

#define WHITE_LOWER_THRESH_H 60 //65
#define WHTIE_LOWER_THRESH_S 0 //0
#define WHITE_LOWER_THRESH_V 130 //145

#define WHITE_DILATION_SIZE 5
#define WHITE_EROSION_SIZE 3

#define YELLOW_DILATION_SIZE 5
#define YELLOW_EROSION_SIZE 3

__global__ void color_filter_kernel(
    unsigned char* pixel_array, 
    int image_height, 
    int image_width,
    unsigned char* filter_ws,
    unsigned char* yellow_out,
    unsigned char* white_out,
    unsigned char* gray_scale_out
);

#endif