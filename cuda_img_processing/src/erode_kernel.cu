#include "erode_kernel.cuh"
#define NUM_CHANNELS 3

// Macros
#define MAX2(a,b) ((a) > (b) ? (a) : (b))
#define MIN2(a,b) ((a) < (b) ? (a) : (b))
#define MAX3(a,b,c)  (( (a) > (b) ? (a) : (b) ) > (c) ? \
                      ( (a) > (b) ? (a) : (b) ) : (c))
#define MIN3(a,b,c) ( ((a) < (b) ? (a) : (b)) < (c) ? \
                      ((a) < (b) ? (a) : (b)) : (c) )

#define MAX_EROSION_SIZE MAX2(YELLOW_EROSION_SIZE, WHITE_EROSION_SIZE)
#define MAX_DILATION_SIZE MAX2(YELLOW_DILATION_SIZE, WHITE_DILATION_SIZE)
#define MAX_COLOR_MASK_SIZE MAX2(MAX_EROSION_SIZE, MAX_DILATION_SIZE)

// IMPORTANT - For now I am doing everything in global memory, we can implement tiling later

__device__ __forceinline__
char color_filter_erode(
    char isColor,
    char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE],
    int erosion_size
){
    int start = (MAX_COLOR_MASK_SIZE - erosion_size) / 2;
    for (int i = start; i < start + erosion_size; i++) {
        for (int j = start; j < start + erosion_size; j++) {
            char v = maskArray[i][j];
            isColor &= v;
        }
    }
    return isColor;
}

__device__ __forceinline__
void color_filter_assemble_mask(
    char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE],
    unsigned char* filter_ws,
    int row_index,
    int col_index,
    int image_width,
    int image_height
){
    for(int i = 0; i < MAX_COLOR_MASK_SIZE; i++){
        int erode_row = row_index - MAX_COLOR_MASK_SIZE/2 + i;
        for(int j = 0; j< MAX_COLOR_MASK_SIZE; j++){
            int erode_col = col_index - MAX_COLOR_MASK_SIZE/2 + j;
            // check if index is valid
            if (erode_col >= 0 && 
                erode_row >= 0 &&
                erode_col < image_width &&
                erode_row < image_height
            ){
                maskArray[i][j] = filter_ws[erode_row * image_width + erode_col];
            }else{
                maskArray[i][j] = 0x00;
            }
        }
    }
}


// KERNEL DECLARATION //
__global__ void erode_kernel(
    unsigned char* image_in, 
    int image_height, 
    int image_width,
    unsigned char* image_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        // subGoal 1.3 Erode and Dilate - NOTE: COULD BENEFIT MASSIVELY BY TILING
        char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE];
        char pixelVal = image_in[pixelIndex];
        color_filter_assemble_mask(maskArray, image_in, rowIndex, colIndex, image_width, image_height);
        image_out[pixelIndex] = color_filter_erode(pixelVal, maskArray, YELLOW_EROSION_SIZE);
    }
    return;
}