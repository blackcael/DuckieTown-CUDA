#ifndef TIMING_H
#define TIMING_H


float ms_color = 0.0f;
float ms_erode = 0.0f;
float ms_dilate = 0.0f;
float ms_blur = 0.0f;
float ms_sobel = 0.0f;
float ms_NMS = 0.0f;
float ms_bitwiseAND = 0.0f;
float ms_total_kernels = 0.0f;

void timing_calc_total_kernels();

void timing_start();

void timing_stop(float* float);

void timing_print_times();

#endif // TIMING_H