#include "sobel_filter_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later

// KERNEL DECLARATION //
__global__ void NMS_kernel(
    float* magnitude2_in,
    unsigned char* angle_in,
    int image_height, 
    int image_width,
    float* nms_mag_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        float m = magnitude2_in[pixelIndex];
        char dir = angle_in[pixelIndex];
        float m1, m2;

        switch (dir) {
          case 0:  // horizontal gradient → compare left/right neighbors (same row)
            m1 = magnitude2_in[rowIndex * image_width + (colIndex -1)];
            m2 = magnitude2_in[rowIndex * image_width + (colIndex +1)];
            break;
          case 1:  // 45° 
            m1 = magnitude2_in[(rowIndex -1) * image_width + (colIndex +1)]; 
            m2 = magnitude2_in[(rowIndex +1) * image_width + (colIndex -1)];
            break;
          case 2:  // 90°
            m1 = magnitude2_in[(rowIndex -1) * image_width + colIndex];
            m2 = magnitude2_in[(rowIndex +1) * image_width + colIndex];
            break;
          case 3:  // 135°
            m1 = magnitude2_in[(rowIndex -1) * image_width + (colIndex -1)];
            m2 = magnitude2_in[(rowIndex +1) * image_width + (colIndex +1)];
            break;
        }

        if (m >= m1 && m >= m2){
            nms_mag_out[pixelIndex] = m;
        }else{
            nms_mag_out[pixelIndex] = 0.0f;
        }
    }
}    



// void cuda_sobel_filter_run(
//     unsigned char* blurred_pixels_in, 
//     int image_height, 
//     int image_width,
//     float* mag_output,
//     float* angle_output
// ){

// }