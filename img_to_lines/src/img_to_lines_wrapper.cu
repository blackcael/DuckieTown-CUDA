
#include "htk.h"
#include "image_utils.h"
#include "img_to_lines.cuh"

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

int main(int argc, char *argv[]) {
  char* filepath = argv[1];
  printf("Using File: %s\n", filepath);
  Image img = image_utils_load_image(filepath);

  // inputs / outputs memory arrays
  char *host_pixels_in = img.pixels;
  char *host_pixels_yellow_out;
  char *host_pixels_white_out;
  char *device_pixels_in;
  char *device_pixels_yellow_out;
  char *device_pixels_white_out;

  // declare sizes
  int img_size = img.height * img.width * img.channels * sizeof(char);
  int img_array_size = img.height * img.width * sizeof(float);

  // allocate memory on device
  htkTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **) &device_pixels_in, img_size);
  cudaMalloc((void **) &device_pixels_yellow_out, img_array_size);
  cudaMalloc((void **) &device_pixels_white_out, img_array_size);
  htkTime_stop(GPU, "Allocating GPU memory.");

  // copy data onto device
  htkTime_start(Copy, "Copying input memory to the GPU.");
  cudaMemcpy(device_pixels_in, host_pixels_in, img_size, cudaMemcpyHostToDevice);
  htkTime_stop(Copy, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
  dim3 DimGrid((img.width-1)/BLOCK_SIZE_X+1, (img.height-1) / BLOCK_SIZE_Y+1,1);

  htkTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  img_to_lines_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      // gaussian kernel (do this in constant memory)
                      // gaussian denominator
                      // filter ws (char)
                      // blur ws x (float)
                      // blur ws y (float)
                      // mag2_ws (float)
                      device_pixels_yellow_out,
                      device_pixels_white_out
                    );

  cudaDeviceSynchronize();
  htkTime_stop(Compute, "Performing CUDA computation");

  htkTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(host_pixels_yellow_out, device_pixels_yellow_out, img_array_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_white_out, device_pixels_white_out, img_array_size, cudaMemcpyDeviceToHost);

  htkTime_stop(Copy, "Copying output memory to the CPU");

  htkTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(device_pixels_in);
  cudaFree(device_pixels_yellow_out);
  cudaFree(device_pixels_white_out);

  htkTime_stop(GPU, "Freeing GPU Memory");

  free(host_pixels_in);
  free(host_pixels_yellow_out);
  free(host_pixels_white_out);

  return 0;
}
