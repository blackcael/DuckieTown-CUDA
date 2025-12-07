#ifndef TIMING_H
#define TIMING_H
#include <stdio.h>

extern float ms_memcpyHD;
extern float ms_color;
extern float ms_erode ;
extern float ms_dilate;
extern float ms_blur;
extern float ms_sobel;
extern float ms_NMS;
extern float ms_bitwiseAND;
extern float ms_total_kernels;
extern float ms_memcpyDH;

void timing_init();

void timing_calc_total_kernels();

void timing_start();

void timing_stop(float* ms_event);

void timing_print_times();

#endif // TIMING_H