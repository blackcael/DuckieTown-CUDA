
#include <math.h>
#include "img_to_lines.cuh"
#define TILE_WIDTH 16
#define BLOCK_SIZE_X TILE_WIDTH // tweak for performance
#define BLOCK_SIZE_Y TILE_WIDTH // tweak for performance
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
#define MAX_EDGE_MASK_SIZE MAX2(BLUR_MASK_SIZE, SOBLE_MASK_SIZE)
#define MAX_MASK_SIZE MAX2(MAX_COLOR_MASK_SIZE, MAX_EDGE_MASK_SIZE)

#define INVALID_MASK 0xCC
#define YELLOW_MASK 0xF0
#define WHITE_MASK 0x0F

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

__global__ void img_to_lines_kernel(
    char* pixel_array, 
    uint8_t image_height, 
    uint8_t image_width,
    uint8_t* gaussian_kernel,
    uint8_t gaussian_denominator,
    char* filter_ws,
    float* blur_ws_x,
    float* blur_ws_y,
    float* mag2_ws,
    char* yellow_out,
    char* white_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= image_height || colIndex >= image_width) return;

    // Calculate position in image
    int pixelIndex = colIndex * image_height + rowIndex;
    char r = pixel_array[NUM_CHANNELS * pixelIndex + 0];
    char g = pixel_array[NUM_CHANNELS * pixelIndex + 1];
    char b = pixel_array[NUM_CHANNELS * pixelIndex + 2];

    // ### Goal 1.0 Filter Color Masks
    // subGoal 1.1 Convert to HSV
    // Normalize to [0,1]
    float r_0 = r / 255.0f;
    float g_0 = g / 255.0f;
    float b_0 = b / 255.0f; 

    // Use your MAX3/MIN3 macros with floats
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
        H = 0.0f;  // Undefined, but we can set to 0
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
    char H_255 = (char)(H / 360.0f * 255.0f + 0.5f);
    char S_255 = (char)(S * 255.0f + 0.5f);
    char V_255 = (char)(V * 255.0f + 0.5f);

    // subGoal 1.2 Filter for Yellow and WHITE
    char isYELLOW_WHITE = 0x00;
    if(H_255 < YELLOW_UPPER_THRESH_H &&
       H_255 > YELLOW_LOWER_THRESH_H &&
       S_255 < YELLOW_UPPER_THRESH_S &&
       S_255 > YELLOW_LOWER_THRESH_S &&
       V_255 < YELLOW_UPPER_THRESH_V &&
       V_255 > YELLOW_LOWER_THRESH_V){
         isYELLOW_WHITE |= 0xF0;
    }
    if(H_255 < WHITE_UPPER_THRESH_H &&
       H_255 > WHITE_LOWER_THRESH_H &&
       S_255 < WHTIE_UPPER_THRESH_S &&
       S_255 > WHTIE_LOWER_THRESH_S &&
       V_255 < WHITE_UPPER_THRESH_V &&
       V_255 > WHITE_LOWER_THRESH_V){
         isYELLOW_WHITE |= 0x0F;
    }
    filter_ws[pixelIndex] = isYELLOW_WHITE;

    // subGoal 1.3 Erode and Dilate - NOTE: COULD BENEFIT MASSIVELY BY TILING
    __syncthreads();
    char maskArray[MAX_COLOR_MASK_SIZE][MAX_COLOR_MASK_SIZE];
    // Assemble mask
    for(int i = 0; i < MAX_COLOR_MASK_SIZE; i++){
        int erode_row = rowIndex - MAX_COLOR_MASK_SIZE/2 + i;
        for(int j; j< MAX_COLOR_MASK_SIZE; j++){
            int erode_col = colIndex - MAX_COLOR_MASK_SIZE/2 + i;
            // check if index is valid
            if (erode_col >= 0 && 
                erode_row >= 0 &&
                erode_col <= image_width &&
                erode_row <= image_height
            ){
                maskArray[i][j] = filter_ws[erode_row * image_width + erode_col];
            }else{
                maskArray[i][j] = INVALID_MASK;
            }
        }
    }

    // ERODE YELLOW
    int start_idx = (MAX_COLOR_MASK_SIZE - YELLOW_EROSION_SIZE)/2;
    for(int i = start_idx; i < start_idx + YELLOW_EROSION_SIZE; i++){
        for(int j = start_idx; j < start_idx + YELLOW_EROSION_SIZE; j++){
            if(maskArray[i][j] != INVALID_MASK){
                isYELLOW_WHITE &= (WHITE_MASK | maskArray[i][j]);
            }
        }
    }
    // DILATE YELLOW
    int start_idx = (MAX_COLOR_MASK_SIZE - YELLOW_DILATION_SIZE)/2;
    for(int i = start_idx; i < start_idx + YELLOW_DILATION_SIZE; i++){
        for(int j = start_idx; j < start_idx + YELLOW_DILATION_SIZE; j++){
            if(maskArray[i][j] != INVALID_MASK){
                isYELLOW_WHITE |= (YELLOW_MASK & maskArray[i][j]);
            }
        }
    }

    // ERODE WHITE
    int start_idx = (MAX_COLOR_MASK_SIZE - WHITE_EROSION_SIZE)/2;
    for(int i = start_idx; i < start_idx + WHITE_EROSION_SIZE; i++){
        for(int j = start_idx; j < start_idx + WHITE_EROSION_SIZE; j++){
            if(maskArray[i][j] != INVALID_MASK){
                isYELLOW_WHITE &= (YELLOW_MASK | maskArray[i][j]);
            }
        }
    }
    // DILATE WHITE
    int start_idx = (MAX_COLOR_MASK_SIZE - WHITE_DILATION_SIZE)/2;
    for(int i = start_idx; i < start_idx + WHITE_DILATION_SIZE; i++){
        for(int j = start_idx; j < start_idx + WHITE_DILATION_SIZE; j++){
            if(maskArray[i][j] != INVALID_MASK){
                isYELLOW_WHITE |= (WHITE_MASK & maskArray[i][j]);
            }
        }
    }
    // ### Goal 2.0 Canny Edge Detection
    // subGoal 2.1 Convert to GrayScale
    float gray_u8 = V;
    __syncthreads();
    filter_ws[pixelIndex] = gray_u8;
    
    // subGoal 2.2 Gaussian Blur
    __syncthreads();
    float gauss_sum = 0;
    for(int i = 0; i < BLUR_MASK_SIZE; i++){
        int erode_row = rowIndex - BLUR_MASK_SIZE/2 + i;
        for(int j; j< BLUR_MASK_SIZE; j++){
            int erode_col = colIndex - BLUR_MASK_SIZE/2 + i;
            // check if index is valid
            if (erode_col >= 0 && 
                erode_row >= 0 &&
                erode_col <= image_width &&
                erode_row <= image_height
            ){
                gauss_sum += gaussian_kernel[i * BLUR_MASK_SIZE + j] * filter_ws[erode_row * image_width + erode_col];
            }else{
                gauss_sum += gaussian_kernel[i * BLUR_MASK_SIZE + j] * filter_ws[pixelIndex];
            }
        }
    }
    __syncthreads();
    gray_u8 = gauss_sum / gaussian_denominator;
    filter_ws[pixelIndex] = gray_u8;
    __syncthreads();
    // subGoal 2.2 Run Canny Kernel
    int8_t soble_kernel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int8_t soble_kernel_y[9] = {1, 2, 1, 0, 0, 0, -1, -2, -1};
    float soble_sum_x = 0;
    float soble_sum_y = 0;
    float filter_val;
    for(int i = 0; i < SOBLE_MASK_SIZE; i++){
        int erode_row = rowIndex - SOBLE_MASK_SIZE/2 + i;
        for(int j; j< SOBLE_MASK_SIZE; j++){
            int erode_col = colIndex - SOBLE_MASK_SIZE/2 + i;
            // check if index is valid
            if (erode_col >= 0 && 
                erode_row >= 0 &&
                erode_col <= image_width &&
                erode_row <= image_height
            ){
                filter_val = filter_ws[erode_row * image_width + erode_col];
                soble_sum_x += soble_kernel_x[i * SOBLE_MASK_SIZE + j] * filter_val;
                soble_sum_y += soble_kernel_y[i * SOBLE_MASK_SIZE + j] * filter_val;
            }else{
                soble_sum_x += soble_kernel_x[i * SOBLE_MASK_SIZE + j] * gray_u8;
                soble_sum_y += soble_kernel_y[i * SOBLE_MASK_SIZE + j] * gray_u8;
            }
        }
    }
    //calc magnitude
    float magnitude2 = soble_sum_x * soble_sum_x + soble_sum_y + soble_sum_y;
    mag2_ws[pixelIndex] = magnitude2;
    
    //calc and sort angles
    float angle = atan2f(soble_sum_y, soble_sum_x);
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
    
    __syncthreads(); // wait until all calculations are finished
    float m1 = mag2_ws[y1 * image_width + x1];
    float m2 = mag2_ws[y2 * image_width + x2];

    float edge_strength = (magnitude2 >= m1 & magnitude2 >= m2) ? magnitude2 : 0.0f;
    char edge_strength_discrete;
    if (edge_strength >= CANNY_THRESH_HIGH){
        edge_strength_discrete = 2;
    }else if (edge_strength >= CANNY_THRESH_LOW){
        edge_strength_discrete = 1;
    } else edge_strength_discrete = 0;

    // TODO: implement hysterisis / flood fill (skip for now...)

    // ### Goal 3.0 AND the Color Masks and the Canny Edge Detection
    char isYellowEdge = ((edge_strength_discrete == 2) && (isYELLOW_WHITE & YELLOW_MASK)) ? 0xF0 : 0x00;
    char isWhiteEdge = ((edge_strength_discrete == 2) && (isYELLOW_WHITE & WHITE_MASK)) ? 0x0F : 0x00;

    // ### Goal 4.0  Each thread gets to do one best lines calc

    __syncthreads(); // wait until all calculations are finished

    // ### GOAL 5.0 Reduce and save only the best lines

}
