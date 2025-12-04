#include "soble_filter_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later
__device__ __forceinline__
char convolve(
    char isColor,
    char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE],
    int dilation_size
){
    int start = (MAX_COLOR_MASK_SIZE - dilation_size) / 2;
    for (int i = start; i < start + dilation_size; i++) {
        for (int j = start; j < start + dilation_size; j++) {
            char v = maskArray[i][j];
            isColor |= v;
        }
    }
    return isColor;
}

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

        int8_t soble_kernel_x[SOBLE_MASK_LENGTH] = SOBLE_MASK_ARRAY_X;
        int8_t soble_kernel_y[SOBLE_MASK_LENGTH] = SOBLE_MASK_ARRAY_Y;
        float soble_sum_x = 0;
        float soble_sum_y = 0;
        float filter_val;
        for(int i = 0; i < SOBLE_MASK_SIZE; i++){
            int erode_row = rowIndex - SOBLE_MASK_SIZE/2 + i;
            for(int j = 0; j< SOBLE_MASK_SIZE; j++){
                int erode_col = colIndex - SOBLE_MASK_SIZE/2 + j;
                // check if index is valid
                if (erode_col >= 0 && 
                    erode_row >= 0 &&
                    erode_col <= image_width &&
                    erode_row <= image_height
                ){
                    filter_val = blur_ws[erode_row * image_width + erode_col];
                    soble_sum_x += soble_kernel_x[i * SOBLE_MASK_SIZE + j] * filter_val;
                    soble_sum_y += soble_kernel_y[i * SOBLE_MASK_SIZE + j] * filter_val;
                }else{
                    // If outside of boundaries, filterval = centerpoint, which is this pixels gray_f
                    soble_sum_x += soble_kernel_x[i * SOBLE_MASK_SIZE + j] * gray_f;
                    soble_sum_y += soble_kernel_y[i * SOBLE_MASK_SIZE + j] * gray_f;
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