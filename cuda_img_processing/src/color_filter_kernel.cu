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

// At this point:
// - pixels is (width * height * 3 bytes)
// - Layout is row-major, interleaved RGB:
//   pixel (x, y) starts at index (y * width + x) * 3
//   R = pixels[idx + 0], G = pixels[idx + 1], B = pixels[idx + 2]


// KERNEL DECLARATION //
__global__ void color_filter_kernel(
    unsigned char* pixel_array, 
    int image_height, 
    int image_width,
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
     
        const uchar3* pixels3 = reinterpret_cast<const uchar3*>(pixel_array);
        uchar3 rgb = pixels3[pixelIndex];

        float inv255 = 1.0f / 255.0f;
        float r_0 = rgb.x * inv255;
        float g_0 = rgb.y * inv255;
        float b_0 = rgb.z * inv255; 

        // Use MAX3/MIN3 macros with floats
        float c_max   = MAX3(r_0, g_0, b_0);
        float c_min   = MIN3(r_0, g_0, b_0);
        float c_delta = c_max - c_min;  
        float inv_c_delta = (c_delta > 0.0f) ? (1.0f / c_delta) : 0.0f;

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
            H = 60.0f * ((g_0 - b_0) * inv_c_delta);
            if (H < 0.0f)
                H += 360.0f;
        } else if (c_max == g_0) {
            H = 60.0f * (((b_0 - r_0) * inv_c_delta) + 2.0f);
        } else { // c_max == b_0
            H = 60.0f * (((r_0 - g_0) * inv_c_delta) + 4.0f);
        }
        // cast back to chars (faster!)
        unsigned char H_255 = (char)(H / 360.0f * 255.0f + 0.5f);
        unsigned char S_255 = (char)(S * 255.0f + 0.5f);
        unsigned char V_255 = (char)(V * 255.0f + 0.5f);

        // subGoal 1.2 Filter for Yellow and WHITE
        char isYellow, isWhite;
        isYellow = (H_255 <= YELLOW_UPPER_THRESH_H &&
           H_255 >= YELLOW_LOWER_THRESH_H &&
           S_255 <= YELLOW_UPPER_THRESH_S &&
           S_255 >= YELLOW_LOWER_THRESH_S &&
           V_255 <= YELLOW_UPPER_THRESH_V &&
           V_255 >= YELLOW_LOWER_THRESH_V
        ) ? 0xFF : 0x00;

        isWhite = (H_255 <= WHITE_UPPER_THRESH_H &&
           H_255 >= WHITE_LOWER_THRESH_H &&
           S_255 <= WHTIE_UPPER_THRESH_S &&
           S_255 >= WHTIE_LOWER_THRESH_S &&
           V_255 <= WHITE_UPPER_THRESH_V &&
           V_255 >= WHITE_LOWER_THRESH_V
        ) ? 0xFF : 0x00;

        yellow_out[pixelIndex] = isYellow;
        white_out[pixelIndex] = isWhite;
        gray_scale_out[pixelIndex] = V_255;

    }
    return;
}