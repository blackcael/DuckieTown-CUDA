#include "gaussian_blur_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later


// KERNEL DECLARATION //
__global__ void gaussian_blur_kernel(
    unsigned char* gray_scale_pixels_in, 
    int image_height, 
    int image_width,
    unsigned char* blurred_pixels_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;

        float gaussian_kernel[BLUR_MASK_LENGTH] = GAUSSIAN_BLUR_ARRAY;
        float gauss_sum = 0;
        for(int i = 0; i < BLUR_MASK_SIZE; i++){
            int erode_row = rowIndex - BLUR_MASK_SIZE/2 + i;
            for(int j = 0; j< BLUR_MASK_SIZE; j++){
                int erode_col = colIndex - BLUR_MASK_SIZE/2 + j;
                // check if index is valid
                if (erode_col >= 0 && 
                    erode_row >= 0 &&
                    erode_col <= image_width &&
                    erode_row <= image_height
                ){
                    gauss_sum += gaussian_kernel[i * BLUR_MASK_SIZE + j] * gray_scale_pixels_in[erode_row * image_width + erode_col];
                }else{
                    gauss_sum += gaussian_kernel[i * BLUR_MASK_SIZE + j] * gray_scale_pixels_in[pixelIndex];
                }
            }
        }
        blurred_pixels_out[pixelIndex] = (unsigned char)(gauss_sum / GAUSSIAN_DENOMINATOR);
        __syncthreads(); //(1)
    }else{
        __syncthreads(); //(1)
    }
    return;
}