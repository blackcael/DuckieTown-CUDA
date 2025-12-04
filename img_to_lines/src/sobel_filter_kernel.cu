#include "sobel_filter_kernel.cuh"

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))



// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later

// KERNEL DECLARATION //
__global__ void sobel_filter_kernel(
    unsigned char* blurred_pixels_in, 
    int image_height, 
    int image_width,
    float* mag2_ws,
    unsigned char* magnitude2_ws
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
                    erode_col <= image_width &&
                    erode_row <= image_height
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
        //calc magnitude
        unsigned char magnitude2 = (unsigned char)(sobel_sum_x * sobel_sum_x + sobel_sum_y * sobel_sum_y);
        magnitude2_ws[pixelIndex] = magnitude2;
        float mag2 = sobel_sum_x * sobel_sum_x + sobel_sum_y * sobel_sum_y;
        mag2_ws[pixelIndex] = mag2;

        __syncthreads(); //(1)

        //calc and sort angles
        float angle = atan2f(sobel_sum_y, sobel_sum_x);
        if(angle < 0) angle += M_PI;
        int dir;
        if (angle < M_PI/8 || angle >=  M_PI * 7 / 8) dir = 0;
        else if (angle < M_PI * 3 / 8) dir = 1;
        else if (angle < M_PI * 5 / 8) dir = 2;
        else dir = 3;

        int x1, y1, x2, y2;
        switch (dir) {
            case 0: // 0°
                x1 = colIndex - 1; y1 = rowIndex;
                x2 = colIndex + 1; y2 = rowIndex;
                break;
            case 1: // 45°
                x1 = colIndex - 1; y1 = rowIndex + 1;
                x2 = colIndex + 1; y2 = rowIndex - 1;
                break;
            case 2: // 90°
                x1 = colIndex;     y1 = rowIndex - 1;
                x2 = colIndex;     y2 = rowIndex + 1;
                break;
            case 3: // 135°
            default:
                x1 = colIndex - 1; y1 = rowIndex - 1;
                x2 = colIndex + 1; y2 = rowIndex + 1;
                break;
        }

        // clamp to borders
        x1 = MAX2(0, MIN2(x1, image_width  - 1));
        y1 = MAX2(0, MIN2(y1, image_height - 1));
        x2 = MAX2(0, MIN2(x2, image_width  - 1));
        y2 = MAX2(0, MIN2(y2, image_height - 1));

        __syncthreads(); //(2)
        float m1 = mag2_ws[y1 * image_width + x1];
        float m2 = mag2_ws[y2 * image_width + x2];

        float edge_strength = (magnitude2 >= m1 & magnitude2 >= m2) ? magnitude2 : 0.0f;
        char edge_strength_discrete;
        if (edge_strength >= CANNY_THRESH_HIGH){
            edge_strength_discrete = 2;
        }else if (edge_strength >= CANNY_THRESH_LOW){
            edge_strength_discrete = 1;
        } else edge_strength_discrete = 0;

        magnitude2_ws[pixelIndex] = (edge_strength_discrete == 2) ? 0xFF : 0x00;
    }else{
        __syncthreads(); //(1)
        __syncthreads(); //(2)
    }
    return;
}