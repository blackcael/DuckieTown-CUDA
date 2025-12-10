#include "rgb_to_grayscale_kernel.cuh"


// PARAMS (not used in program, but used in the LUT)


__global__ void rgb_to_grayscale_kernel(
    unsigned char* rgb_in, 
    int image_height, 
    int image_width,
    unsigned char* gray_scale_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        unsigned char r = pixel_array[NUM_CHANNELS * pixelIndex + 0];
        unsigned char g = pixel_array[NUM_CHANNELS * pixelIndex + 1];
        unsigned char b = pixel_array[NUM_CHANNELS * pixelIndex + 2];
        gray_scale_out[pixelIndex] = r * R_WEIGHT + g * G_WEIGHT + b * B_WEIGHT;
    }
}

#endif