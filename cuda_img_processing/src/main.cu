
#include "image_utils.h"
#include "color_filter_kernel.cuh"
#include "color_LUT_kernel.cuh"
#include "rgb_to_grayscale_kernel.cuh"
#include "erode_kernel.cuh"
#include "dilate_kernel.cuh"
#include "gaussian_blur_kernel.cuh"
#include "sobel_filter_kernel.cuh"
#include "NMS_kernel.cuh"
#include "bitwiseAND_kernel.cuh"
#include "timing.cuh"

#include <stdio.h>
#include <string.h>

#define USE_LUT 0
#define LUT_FILE_PATH "cuda_img_processing/LUT/HSV_LUT.bin"

// IMPORTANT NOTE!
// There is a 0.2ms delay on whatever kernel is first in the pipeline. no clue why :/


#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

#define NEW_HEIGHT 260


int main(int argc, char *argv[]) {
  char* filepath = argv[1];
  printf("\n ===== Using File: %s ===== \n", filepath);
  Image uncropped_img = image_utils_load_image(filepath);
  Image img = image_utils_crop_vertically(&uncropped_img, NEW_HEIGHT);

  // init timer
  timing_init();


  // host inputs / outputs memory arrays
  unsigned char* host_pixels_in = img.pixels;
  unsigned char* host_pixels_yellow_out;
  unsigned char* host_pixels_white_out;
  unsigned char* host_pixels_gray_scale_out;
  unsigned char* host_temp_out_c;  
  float* host_temp_out_f;

  // device inputs / outputs memory arrays
  unsigned char* device_pixels_in;
  unsigned char* device_pixels_yellow_out;
  unsigned char* device_pixels_white_out;
  unsigned char* device_pixels_gray_scale_out;
  unsigned char* device_pixels_blur_out;
  unsigned char* device_pixels_NMS_edge_out;
  unsigned char* device_pixels_yellow_edge_out;
  unsigned char* device_pixels_white_edge_out;

  // device workspace memory arrays
  unsigned char* device_filter_ws_y;
  unsigned char* device_filter_ws_w;
  float* device_blur_ws;
  float* device_mag2_ws;
  unsigned char* device_angle_ws;
  float* device_NMS_mag_ws;


  // declare sizes
  int img_size_3chan_c = img.height * img.width * img.channels * sizeof(unsigned char);
  int img_array_size_c = img.height * img.width * sizeof(unsigned char);
  int img_array_size_f = img.height * img.width * sizeof(float);

  // allocate host memory
  host_pixels_yellow_out = (unsigned char*)malloc(img_array_size_c);
  host_pixels_white_out = (unsigned char*)malloc(img_array_si59ze_c);
  host_pixels_gray_scale_out = (unsigned char*)malloc(img_array_size_c);
  host_temp_out_c = (unsigned char*)malloc(img_array_size_c);
  host_temp_out_f = (float*)malloc(img_array_size_f);

  // allocate memory on device
  // in/out
  cudaMalloc((void **) &device_pixels_in, img_size_3chan_c);
  cudaMalloc((void **) &device_pixels_yellow_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_white_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_gray_scale_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_blur_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_NMS_edge_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_yellow_edge_out, img_array_size_c);
  cudaMalloc((void **) &device_pixels_white_edge_out, img_array_size_c);

  // ws
  cudaMalloc((void **) &device_filter_ws_y, img_array_size_c);
  cudaMalloc((void **) &device_filter_ws_w, img_array_size_c);
  cudaMalloc((void **) &device_blur_ws, img_array_size_f);
  cudaMalloc((void **) &device_mag2_ws, img_array_size_f);
  cudaMalloc((void **) &device_angle_ws, img_array_size_c);
  cudaMalloc((void **) &device_NMS_mag_ws, img_array_size_f);

  // Load LUT
  unsigned char* host_color_LUT = image_utils_load_LUT_from_file(LUT_FILE_PATH);
  unsigned char* device_color_LUT;
  cudaMalloc((void **) &device_color_LUT, 256 * 256 * 256 * sizeof(unsigned char));
  cudaMemcpy(device_color_LUT, host_color_LUT, 256 * 256 * 256* sizeof(unsigned char), cudaMemcpyHostToDevice);
  free(host_color_LUT);

  // copy data onto device
  timing_start();
  cudaMemcpy(device_pixels_in, host_pixels_in, img_size_3chan_c, cudaMemcpyHostToDevice);
  timing_stop(&ms_memcpyHD);
  //@@ Initialize the grid and block dimensions here
  dim3 DimBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y,1);
  dim3 DimGrid((img.width-1)/BLOCK_SIZE_X+1, (img.height-1) / BLOCK_SIZE_Y+1,1);

  printf("begining kernel\n");


  //@@ Launch the GPU Kernel here
  #if USE_LUT

  // independent grayscale conversion
  timing_start();
  rgb_to_grayscale_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      device_pixels_gray_scale_out
                    );
  timing_stop(&ms_rgb_to_gray);
  // color filtering
  timing_start();
  color_LUT_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      device_color_LUT,
                      device_pixels_yellow_out,
                      device_pixels_white_out
                    );
  timing_stop(&ms_color);
  
  
  #else
  // independent grayscale conversion
  timing_start();
  rgb_to_grayscale_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      device_pixels_gray_scale_out
                    );
  timing_stop(&ms_rgb_to_gray);

  timing_start();
  // color filtering
  color_filter_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_in,
                      img.height,
                      img.width,
                      device_pixels_yellow_out,
                      device_pixels_white_out,
                      device_pixels_gray_scale_out
                    );
  timing_stop(&ms_color);
  #endif 
  
  //ERODES
  timing_start();
  erode_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_yellow_out,
                      img.height,
                      img.width,
                      device_filter_ws_y
                    );
  erode_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_white_out,
                      img.height,
                      img.width,
                      device_filter_ws_w
                    );
  timing_stop(&ms_erode);

  //DILATES
  timing_start();
  dilate_kernel<<<DimGrid,DimBlock>>>(
                      device_filter_ws_y,
                      img.height,
                      img.width,
                      device_pixels_yellow_out
                    );
  dilate_kernel<<<DimGrid,DimBlock>>>(
                      device_filter_ws_w,
                      img.height,
                      img.width,
                      device_pixels_white_out
                    );
  timing_stop(&ms_dilate);

  timing_start();
  gaussian_blur_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_gray_scale_out,
                      img.height,
                      img.width,
                      device_pixels_blur_out
                    );   
  timing_stop(&ms_blur);

  timing_start();
  sobel_filter_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_blur_out,
                      img.height,
                      img.width,
                      device_mag2_ws,
                      device_angle_ws
                    );   
  timing_stop(&ms_sobel);

  timing_start();
  NMS_kernel<<<DimGrid,DimBlock>>>(
                      device_mag2_ws,
                      device_angle_ws,
                      img.height,
                      img.width,
                      device_NMS_mag_ws,
                      device_pixels_NMS_edge_out
                    );   
  timing_stop(&ms_NMS);

  timing_start();
  bitwiseAND_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_yellow_out,
                      device_pixels_white_out,
                      device_pixels_NMS_edge_out,
                      img.height,
                      img.width,
                      device_pixels_yellow_edge_out,
                      device_pixels_white_edge_out
                    );   
  timing_stop(&ms_bitwiseAND);
  



  cudaError_t err = cudaGetLastError();
  printf("kernel error: %s\n", cudaGetErrorString(err));

  printf("Finishing kernel\n");



  //@@ Copy the GPU memory back to the CPU here
  timing_start();
  cudaMemcpy(host_pixels_yellow_out, device_pixels_yellow_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_white_out, device_pixels_white_out, img_array_size_c, cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_pixels_gray_scale_out, device_pixels_gray_scale_out, img_array_size_c, cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_temp_out_f, device_NMS_mag_ws, img_array_size_f, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_temp_out_c, device_pixels_white_edge_out, img_array_size_c, cudaMemcpyDeviceToHost);
  timing_stop(&ms_memcpyDH);

  timing_print_times();


  // Turn pixel arrays into jpegs
  printf("beginning jpegization\n");
  Image output_image = {img.width, img.height, 1, host_pixels_yellow_out};
  char output_file_path[64];
  image_utils_build_output_path(output_file_path, filepath, 64);
  image_utils_save_jpeg(output_file_path, &output_image, 100);
  printf("ending jpegization\n");


  // Free Device Resources
  cudaFree(device_pixels_in);
  cudaFree(device_pixels_yellow_out);
  cudaFree(device_pixels_white_out);
  cudaFree(device_pixels_gray_scale_out);
  cudaFree(device_pixels_blur_out);
  cudaFree(device_pixels_NMS_edge_out);
  cudaFree(device_pixels_yellow_edge_out);
  cudaFree(device_pixels_white_edge_out);
  cudaFree(device_filter_ws_y);
  cudaFree(device_filter_ws_w);
  cudaFree(device_blur_ws);
  cudaFree(device_mag2_ws);
  cudaFree(device_angle_ws);
  cudaFree(device_NMS_mag_ws);

  //Free Host Resources
  free(host_pixels_yellow_out);
  free(host_pixels_white_out);
  free(host_pixels_gray_scale_out);
  free(host_temp_out_c);
  free(host_temp_out_f);

  //Free Images
  image_utils_free_image(&img);
  image_utils_free_image(&uncropped_img);





  return 0;
}
