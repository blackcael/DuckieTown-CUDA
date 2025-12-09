#include "timing.cuh"
// ---- Profiling setup ----
float ms_memcpyHD = 0.0f;
float ms_color = 0.0f;
float ms_erode = 0.0f;
float ms_dilate = 0.0f;
float ms_blur = 0.0f;
float ms_sobel = 0.0f;
float ms_NMS = 0.0f;
float ms_bitwiseAND = 0.0f;
float ms_total_kernels = 0.0f;
float ms_memcpyDH = 0.0f;


cudaEvent_t ev_start, ev_stop;

void timing_init(){
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
}

void timing_calc_total_kernels() {
    ms_total_kernels =
        ms_color + ms_erode + ms_dilate +
        ms_blur + ms_sobel + ms_NMS + ms_bitwiseAND;
}

void timing_start() {
    cudaEventRecord(ev_start);
}

void timing_stop(float* ms_event) {
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    cudaEventElapsedTime(ms_event, ev_start, ev_stop);
}

void timing_print_times() {
    timing_calc_total_kernels();
    printf("\n=== CUDA TIMING (per image) ===\n");
    printf("H2D copies        : %8.3f ms\n", ms_memcpyHD);
    printf("color_filter      : %8.3f ms\n", ms_color);
    printf("erode (both)      : %8.3f ms\n", ms_erode);
    printf("dilate (both)     : %8.3f ms\n", ms_dilate);
    printf("gaussian_blur     : %8.3f ms\n", ms_blur);
    printf("canny_edges       : %8.3f ms\n", ms_sobel + ms_NMS);
    printf("bitwiseAND        : %8.3f ms\n", ms_bitwiseAND);
    printf("D2H copies        : %8.3f ms\n", ms_memcpyDH);
    printf("-------------------------------\n");
    printf("Total kernels     : %8.3f ms\n", ms_total_kernels);
    printf("Total time        : %8.3f ms\n", ms_total_kernels + ms_memcpyDH + ms_memcpyHD);
}