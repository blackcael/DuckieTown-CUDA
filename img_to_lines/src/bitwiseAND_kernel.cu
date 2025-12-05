#include "gaussian_blur_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later


// KERNEL DECLARATION //
__global__ void bitwiseAND_kernel(
    unsigned char* yellow_pixels_in,
    unsigned char* white_pixels_in,
    unsigned char* edge_mask_in,
    int image_height,    
    int image_width,
    unsigned char* yellow_edge_out,
    unsigned char* white_edge_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        yellow_edge_out[pixelIndex] = (yellow_pixels_in[pixelIndex] & edge_mask_in[pixelIndex]);
        white_edge_out[pixelIndex] = (white_pixels_in[pixelIndex] & edge_mask_in[pixelIndex]);
    }
    return;
}