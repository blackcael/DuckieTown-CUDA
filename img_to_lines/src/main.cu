
#include "image_utils.h"
#include "img_to_lines.cuh"
#include <stdio.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

int main(int argc, char *argv[]) {
  char* filepath = argv[1];
  printf("Using File: %s\n", filepath);
  Image img = image_utils_load_image(filepath);


  // host inputs / outputs memory arrays
  char* host_pixels_in = img.pixels;
  char* host_pixels_yellow_out;
  char* host_pixels_white_out;

  // device inputs / outputs memory arrays
  char* device_pixels_in;
  char* device_pixels_yellow_out;
  char* device_pixels_white_out;
  
  // device workspace memory arrays
  char* device_filter_ws;
  float* device_blur_ws;
  float* device_mag2_ws;


  // declare sizes
  int img_size_chan = img.height * img.width * img.channels * sizeof(char);
  int img_array_size_c = img.height * img.width * sizeof(char);
  int img_array_size_f = img.height * img.width * sizeof(char);

  // allocate host memory
  host_pixels_yellow_out = (char*)malloc(img_array_size_c);
  host_pixels_white_out = (char*)malloc(img_array_size_c);

  // allocate memory on device
  // in/out
  cudaMalloc((void **) &device_pixels_in, img_size_chan);
  cudaMalloc((void **) &device_pixels_yellow_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_white_out, img_array_size_c);
  // ws
  cudaMalloc((void **) &device_filter_ws, img_array_size_c);
  cudaMalloc((void **) &device_mag2_ws, img_array_size_f);
  cudaMalloc((void **) &device_blur_ws, img_array_size_f);
  // copy data onto device
  cudaMemcpy(device_pixels_in, host_pixels_in, img_size_chan, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
  dim3 DimGrid((img.width-1)/BLOCK_SIZE_X+1, (img.height-1) / BLOCK_SIZE_Y+1,1);

  //@@ Launch the GPU Kernel here
  img_to_lines_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      (float)GAUSSIAN_DENOMINATOR,
                      device_filter_ws,
                      device_blur_ws,
                      device_mag2_ws,
                      device_pixels_yellow_out,
                      device_pixels_white_out
                    );
  cudaDeviceSynchronize();

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(host_pixels_yellow_out, device_pixels_yellow_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_white_out, device_pixels_white_out, img_array_size_c, cudaMemcpyDeviceToHost);


  //@@ Free the GPU memory here
  cudaFree(device_pixels_in);
  cudaFree(device_pixels_yellow_out);
  cudaFree(device_pixels_white_out);

  // Free Host Memory
  free(host_pixels_in);
  free(host_pixels_yellow_out);
  free(host_pixels_white_out);

  return 0;
}
