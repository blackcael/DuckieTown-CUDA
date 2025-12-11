#include "color_LUT_kernel.cuh"
#define NUM_CHANNELS 3
#define YELLOW_MASK  0b00000010
#define WHITE_MASK   0b00000001

// can I restrict the input array (pixels in ?)

__global__ void color_LUT_kernel(
    unsigned char* rgb_pixels_in, 
    int image_height, 
    int image_width,
    const unsigned char* __restrict__ LUT,
    unsigned char* yellow_out,
    unsigned char* white_out
){
    // Calculate indices
    int rowIndex = blockIdx.y * blockDim.y + threadIdx.y;
    int colIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned char inRange = 1;
    if (rowIndex >= image_height || colIndex >= image_width) inRange = 0;

    // Calculate position in image
    if(inRange){
        int pixelIndex = rowIndex * image_width + colIndex;

        const uchar3* pixels3 = reinterpret_cast<const uchar3*>(rgb_pixels_in);
        uchar3 rgb = pixels3[pixelIndex];
        unsigned char r = rgb.x;
        unsigned char g = rgb.y;
        unsigned char b = rgb.z;
        unsigned char lut_val = LUT[r * 256 * 256 + g * 256 + b];
        yellow_out[pixelIndex] = YELLOW_MASK & lut_val;
        white_out[pixelIndex] = WHITE_MASK & lut_val;
    }
    return;
}
