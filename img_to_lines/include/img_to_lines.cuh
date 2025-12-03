#ifndef IMG_TO_LINES_H
#define IMG_TO_LINES_H

// includes
#include "image_utils.h"
#include <cuda_runtime.h>
#include <stdint.h>

// PARAMS
#define MAX_NUM_OF_LINES_YELLOW 16
#define MAX_NUM_OF_LINES_WHITE 16

#define YELLOW_UPPER_THRESH_H 28
#define YELLOW_UPPER_THRESH_S 180
#define YELLOW_UPPER_THRESH_V 255

#define YELLOW_LOWER_THRESH_H 12
#define YELLOW_LOWER_THRESH_S 50
#define YELLOW_LOWER_THRESH_V 140

#define WHITE_UPPER_THRESH_H 150
#define WHTIE_UPPER_THRESH_S 100
#define WHITE_UPPER_THRESH_V 255

#define WHITE_LOWER_THRESH_H 65
#define WHTIE_LOWER_THRESH_S 0
#define WHITE_LOWER_THRESH_V 145

#define WHITE_DILATION_SIZE 5
#define WHITE_EROSION_SIZE 3

#define YELLOW_DILATION_SIZE 5
#define YELLOW_EROSION_SIZE 3

#define BLUR_MASK_SIZE 3
#define SOBLE_MASK_SIZE 3

#define CANNY_THRESH_HIGH 100
#define CANNY_THRESH_LOW 0

void img_to_lines_kernel(
    char* pixel_array, 
    uint8_t image_height, 
    uint8_t image_width,
    uint8_t* gaussian_kernel,
    uint8_t gaussian_denominator,
    char* filter_ws,
    float* blur_ws_x,
    float* blur_ws_y,
    float* mag2_ws,
    char* yellow_out,
    char* white_out
);

#endif