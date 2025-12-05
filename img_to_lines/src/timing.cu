#include "timing.cuh"
// ---- Profiling setup ----
cudaEvent_t ev_start, ev_stop;
cudaEventCreate(&ev_start);
cudaEventCreate(&ev_stop);

float ms_color = 0.0f;
float ms_erode = 0.0f;
float ms_dilate = 0.0f;
float ms_blur = 0.0f;
float ms_sobel = 0.0f;
float ms_NMS = 0.0f;
float ms_bitwiseAND = 0.0f;
float ms_total_kernels = 0.0f;

void timing_calc_total_kernels() {
    ms_total_kernels =
        ms_color + ms_erode + ms_dilate +
        ms_blur + ms_sobel + ms_NMS + ms_bitwiseAND;
}

void timing_start() {
    cudaEventRecord(ev_start);
}

void timing_stop(float* float) {
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(&ms_H2D, ev_start, ev_stop);
}

void timing_print_times() {
    printf("\n=== CUDA TIMING (per image) ===\n");
    printf("color_filter      : %8.3f ms\n", ms_color);
    printf("erode (both)      : %8.3f ms\n", ms_erode);
    printf("dilate (both)     : %8.3f ms\n", ms_dilate);
    printf("gaussian_blur     : %8.3f ms\n", ms_blur);
    printf("sobel             : %8.3f ms\n", ms_sobel);
    printf("NMS               : %8.3f ms\n", ms_NMS);
    printf("bitwiseAND        : %8.3f ms\n", ms_bitwiseAND);
    printf("D2H copies        : %8.3f ms\n", ms_D2H);
    printf("-------------------------------\n");
    printf("Total kernels     : %8.3f ms\n", ms_total_kernels);
}