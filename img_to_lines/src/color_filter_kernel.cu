

#include "color_filter_kernel.cuh"
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

/*
img_to_lines()
Arguments:
- Pointer to array that was jpg image
- Pointers to convolution masks
- Pointer to output array / data structure


// At this point:
// - image width = 640 pixels
// - image height = 480 pixels
// - pixels is (width * height * 3 bytes)
// - Layout is row-major, interleaved RGB:
//   pixel (x, y) starts at index (y * width + x) * 3
//   R = pixels[idx + 0], G = pixels[idx + 1], B = pixels[idx + 2]

*/ 

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
char color_filter_dilate(
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
                erode_col <= image_width &&
                erode_row <= image_height
            ){
                maskArray[i][j] = filter_ws[erode_row * image_width + erode_col];
            }else{
                maskArray[i][j] = 0x00;
            }
        }
    }
}


// KERNEL DECLARATION //
__global__ void color_filter_kernel(
    unsigned char* pixel_array, 
    int image_height, 
    int image_width,
    unsigned char* filter_ws,
    unsigned char* yellow_out,
    unsigned char* white_out,
    unsigned char* gray_scale_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;
        unsigned char r = pixel_array[NUM_CHANNELS * pixelIndex + 0];
        unsigned char g = pixel_array[NUM_CHANNELS * pixelIndex + 1];
        unsigned char b = pixel_array[NUM_CHANNELS * pixelIndex + 2];

        // ### Goal 1.0 Filter Color Masks
        // subGoal 1.1 Convert to HSV
        // Normalize to [0,1]
        float r_0 = r / 255.0f;
        float g_0 = g / 255.0f;
        float b_0 = b / 255.0f; 

        // Use MAX3/MIN3 macros with floats
        float c_max   = MAX3(r_0, g_0, b_0);
        float c_min   = MIN3(r_0, g_0, b_0);
        float c_delta = c_max - c_min;  

        float H, S, V;  

        // Value
        V = c_max;  

        // Saturation
        if (c_max == 0.0f) {
            S = 0.0f;
        } else {
            S = c_delta / c_max;
        } 
        // Hue
        if (c_delta == 0.0f) {
            H = 0.0f;
        } else if (c_max == r_0) {
            H = 60.0f * ((g_0 - b_0) / c_delta);
            if (H < 0.0f)
                H += 360.0f;
        } else if (c_max == g_0) {
            H = 60.0f * (((b_0 - r_0) / c_delta) + 2.0f);
        } else { // c_max == b_0
            H = 60.0f * (((r_0 - g_0) / c_delta) + 4.0f);
        }
        // cast back to chars (faster!)
        unsigned char H_255 = (char)(H / 360.0f * 255.0f + 0.5f);
        unsigned char S_255 = (char)(S * 255.0f + 0.5f);
        unsigned char V_255 = (char)(V * 255.0f + 0.5f);

        // subGoal 1.2 Filter for Yellow and WHITE
        char isYellow, isWhite;
        if(H_255 < YELLOW_UPPER_THRESH_H &&
           H_255 > YELLOW_LOWER_THRESH_H &&
           S_255 < YELLOW_UPPER_THRESH_S &&
           S_255 > YELLOW_LOWER_THRESH_S &&
           V_255 < YELLOW_UPPER_THRESH_V &&
           V_255 > YELLOW_LOWER_THRESH_V
        ){
            isYellow = 0xFF;
        }else{
            isYellow = 0x00;
        }
        if(H_255 < WHITE_UPPER_THRESH_H &&
           H_255 > WHITE_LOWER_THRESH_H &&
           S_255 < WHTIE_UPPER_THRESH_S &&
           S_255 > WHTIE_LOWER_THRESH_S &&
           V_255 < WHITE_UPPER_THRESH_V &&
           V_255 > WHITE_LOWER_THRESH_V
        ){
            isWhite = 0xFF;
        }else{
            isWhite = 0x00;
        }
        // debug loop
        // yellow_out[pixelIndex] = isYellow;
        // white_out[pixelIndex] = isWhite;
        // return;

        filter_ws[pixelIndex] = isYellow;
    
        // subGoal 1.3 Erode and Dilate - NOTE: COULD BENEFIT MASSIVELY BY TILING
        char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE];

        // Erode Yellow
        __syncthreads(); //(1)
        color_filter_assemble_mask(maskArray, filter_ws, rowIndex, colIndex, image_width, image_height);
        isYellow = color_filter_erode(isYellow, maskArray, YELLOW_EROSION_SIZE);
        // Dilate Yellow
        __syncthreads(); //(2)
        color_filter_assemble_mask(maskArray, filter_ws, rowIndex, colIndex, image_width, image_height);
        isYellow = color_filter_dilate(isYellow, maskArray, YELLOW_DILATION_SIZE);


        // Erode White
        __syncthreads(); //(3)
        color_filter_assemble_mask(maskArray, filter_ws, rowIndex, colIndex, image_width, image_height);
        isWhite = color_filter_erode(isWhite, maskArray, WHITE_EROSION_SIZE);
        // Dilate White
        __syncthreads(); //(4)
        color_filter_assemble_mask(maskArray, filter_ws, rowIndex, colIndex, image_width, image_height);
        isWhite = color_filter_erode(isWhite, maskArray, WHITE_DILATION_SIZE);

        // // subGoal 2.1 Convert to GrayScale
        gray_scale_out[pixelIndex] = V_255;
        yellow_out[pixelIndex] = isYellow;
        white_out[pixelIndex] = isWhite;
    }else{
        __syncthreads(); //(1)
        __syncthreads(); //(2)
        __syncthreads(); //(3)
        __syncthreads(); //(4)
    }
    return;
}