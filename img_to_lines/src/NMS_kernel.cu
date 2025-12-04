#include "sobel_filter_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later

// KERNEL DECLARATION //
__global__ void NMS_kernel(
    float* magnitude2_in,
    float* magnitude2_in,
    int image_height, 
    int image_width,
    float* mag2_out,
    float* angle_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        float gray_f = (float)blurred_pixels_in[pixelIndex];

        int8_t sobel_kernel_x[SOBEL_MASK_LENGTH] = SOBEL_MASK_ARRAY_X;
        int8_t sobel_kernel_y[SOBEL_MASK_LENGTH] = SOBEL_MASK_ARRAY_Y;
        float sobel_sum_x = 0;
        float sobel_sum_y = 0;
        float filter_val;
        for(int i = 0; i < SOBEL_MASK_SIZE; i++){
            int erode_row = rowIndex - SOBEL_MASK_SIZE/2 + i;
            for(int j = 0; j< SOBEL_MASK_SIZE; j++){
                int erode_col = colIndex - SOBEL_MASK_SIZE/2 + j;
                // check if index is valid
                if (erode_col >= 0 && 
                    erode_row >= 0 &&
                    erode_col < image_width &&
                    erode_row < image_height
                ){
                    filter_val = (float)blurred_pixels_in[erode_row * image_width + erode_col];
                    sobel_sum_x += sobel_kernel_x[i * SOBEL_MASK_SIZE + j] * filter_val;
                    sobel_sum_y += sobel_kernel_y[i * SOBEL_MASK_SIZE + j] * filter_val;
                }else{
                    // If outside of boundaries, filterval = centerpoint, which is this pixels gray_f
                    sobel_sum_x += sobel_kernel_x[i * SOBEL_MASK_SIZE + j] * gray_f;
                    sobel_sum_y += sobel_kernel_y[i * SOBEL_MASK_SIZE + j] * gray_f;
                }
            }
        }
        //calc magnitude and angle
        mag2_out[pixelIndex] = sobel_sum_x * sobel_sum_x + sobel_sum_y * sobel_sum_y;
        angle_out[pixelIndex] = atan2f(sobel_sum_y, sobel_sum_x);
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