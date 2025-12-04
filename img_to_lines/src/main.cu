
#include "image_utils.h"
#include "color_filter_kernel.cuh"
#include <stdio.h>
#include <string.h>

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define NEW_HEIGHT 260


int main(int argc, char *argv[]) {
  char* filepath = argv[1];
  printf("Using File: %s\n", filepath);
  Image uncropped_img = image_utils_load_image(filepath);
  Image img = image_utils_crop_vertically(&uncropped_img, NEW_HEIGHT);


  // host inputs / outputs memory arrays
  unsigned char* host_pixels_in = img.pixels;
  unsigned char* host_pixels_yellow_out;
  unsigned char* host_pixels_white_out;
  unsigned char* host_pixels_gray_scale_out;

  // device inputs / outputs memory arrays
  unsigned char* device_pixels_in;
  unsigned char* device_pixels_yellow_out;
  unsigned char* device_pixels_white_out;
  unsigned char* device_pixels_gray_scale_out;

  
  // device workspace memory arrays
  unsigned char* device_filter_ws;
  // float* device_blur_ws;
  // float* device_mag2_ws;


  // declare sizes
  int img_size_3chan_c = img.height * img.width * img.channels * sizeof(unsigned char);
  int img_array_size_c = img.height * img.width * sizeof(unsigned char);
  // int img_array_size_f = img.height * img.width * sizeof(float);

  // allocate host memory
  host_pixels_yellow_out = (unsigned char*)malloc(img_array_size_c);
  host_pixels_white_out = (unsigned char*)malloc(img_array_size_c);
  host_pixels_gray_scale_out = (unsigned char*)malloc(img_array_size_c);

  // allocate memory on device
  // in/out
  cudaMalloc((void **) &device_pixels_in, img_size_3chan_c);
  cudaMalloc((void **) &device_pixels_yellow_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_white_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_gray_scale_out, img_array_size_c);
  // ws
  cudaMalloc((void **) &device_filter_ws, img_array_size_c);
  // cudaMalloc((void **) &device_mag2_ws, img_array_size_f);
  // cudaMalloc((void **) &device_blur_ws, img_array_size_f);

  // copy data onto device
  cudaMemcpy(device_pixels_in, host_pixels_in, img_size_3chan_c, cudaMemcpyHostToDevice);

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
  dim3 DimGrid((img.width-1)/BLOCK_SIZE_X+1, (img.height-1) / BLOCK_SIZE_Y+1,1);

  printf("begining kernel\n");

  //@@ Launch the GPU Kernel here
  color_filter_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      device_filter_ws,
                      device_pixels_yellow_out,
                      device_pixels_white_out,
                      device_pixels_gray_scale_out
                    );
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  printf("kernel error: %s\n", cudaGetErrorString(err));

  printf("Finishing kernel\n");

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(host_pixels_yellow_out, device_pixels_yellow_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_white_out, device_pixels_white_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_gray_scale_out, device_pixels_gray_scale_out, img_array_size_c, cudaMemcpyDeviceToHost);


  // Turn pixel arrays into jpegs
  printf("beginning jpegization\n");
  // for(int pixelindex = 0; pixelindex < img_array_size_c; pixelindex++){
  //   printf("PixelIndex: %d, PixelValue: %d\n", pixelindex, host_pixels_white_out[pixelindex]);
  // }
  Image output_image = {img.width, img.height, 1, host_pixels_gray_scale_out};
  char output_file_path[64];
  image_utils_build_output_path(output_file_path, filepath, 64);
  image_utils_save_jpeg(output_file_path, &output_image, 100);
  printf("ending jpegization\n");

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
