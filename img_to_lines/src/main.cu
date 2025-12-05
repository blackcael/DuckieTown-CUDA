
#include "image_utils.h"
#include "color_filter_kernel.cuh"
#include "erode_kernel.cuh"
#include "dilate_kernel.cuh"
#include "gaussian_blur_kernel.cuh"
#include "sobel_filter_kernel.cuh"
#include "NMS_kernel.cuh"
#include "bitwiseAND_kernel.cuh"

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
  host_pixels_white_out = (unsigned char*)malloc(img_array_size_c);
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
                      device_pixels_yellow_out,
                      device_pixels_white_out,
                      device_pixels_gray_scale_out
                    );
  cudaDeviceSynchronize();

  //ERODES
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
  cudaDeviceSynchronize();

  //DILATES
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
  cudaDeviceSynchronize();

  gaussian_blur_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_gray_scale_out,
                      img.height,
                      img.width,
                      device_pixels_blur_out
                    );   
  cudaDeviceSynchronize();

  sobel_filter_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_blur_out,
                      img.height,
                      img.width,
                      device_mag2_ws,
                      device_angle_ws
                    );   
  cudaDeviceSynchronize();

  NMS_kernel<<<DimGrid,DimBlock>>>(
                      device_mag2_ws,
                      device_angle_ws,
                      img.height,
                      img.width,
                      device_NMS_mag_ws,
                      device_pixels_NMS_edge_out
                    );   
  cudaDeviceSynchronize();

  bitwiseAND_kernel<<<DimGrid,DimBlock>>>(
                      device_pixels_yellow_out,
                      device_pixels_white_out,
                      device_pixels_NMS_edge_out,
                      img.height,
                      img.width,
                      device_pixels_yellow_edge_out,
                      device_pixels_white_edge_out
                    );   

  



  cudaError_t err = cudaGetLastError();
  printf("kernel error: %s\n", cudaGetErrorString(err));

  printf("Finishing kernel\n");

  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(host_pixels_yellow_out, device_pixels_yellow_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_white_out, device_pixels_white_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_pixels_gray_scale_out, device_pixels_gray_scale_out, img_array_size_c, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_temp_out_f, device_NMS_mag_ws, img_array_size_f, cudaMemcpyDeviceToHost);
  cudaMemcpy(host_temp_out_c, device_pixels_white_edge_out, img_array_size_c, cudaMemcpyDeviceToHost);

  // // Debug Print Loop -- Simple
  // for(int pixelindex = 0; pixelindex < img_array_size_c; pixelindex++){
  //   int pix_val = host_pixels_white_out[pixelindex];
  //   printf("PixelIndex: %d, PixelValue: %d\n", pixelindex, pix_val);
  // }

  // Debug Print Loop -- Float
  // for(int pixelindex = 0; pixelindex < img_array_size_c; pixelindex++){
  //   float pix_val = host_temp_out_f[pixelindex];
  //   printf("PixelIndex: %d, PixelValue: %f\n", pixelindex, pix_val);
  // }


  // Debug Print Loop -- Counting
  // int pixel_mag_sum = 0;
  // int pixel_mag_non_zero_cnt = 0;
  // for(int pixelindex = 0; pixelindex < img_array_size_c; pixelindex++){
  //   int pix_val = host_pixels_magnitude_out[pixelindex];
  //   if(pix_val != 0){
  //     printf("PixelIndex: %d, PixelValue: %d\n", pixelindex, pix_val);
  //     pixel_mag_non_zero_cnt++;
  //     pixel_mag_sum += pix_val;
  //   } 
  // }
  // printf("Non-Zero Pixel Avg:: %d\n", pixel_mag_sum / pixel_mag_non_zero_cnt);


  // Turn pixel arrays into jpegs
  printf("beginning jpegization\n");
  Image output_image = {img.width, img.height, 1, host_temp_out_c};
  char output_file_path[64];
  image_utils_build_output_path(output_file_path, filepath, 64);
  image_utils_save_jpeg(output_file_path, &output_image, 100);
  printf("ending jpegization\n");


  return 0;
}
