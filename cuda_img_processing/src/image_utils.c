#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define OUTPUT_FILE_DIRECTORY "cuda_img_processing/output_images/"

Image image_utils_load_image(char* file_path_str) {
    Image img = {0};

    img.pixels = (unsigned char*)stbi_load(file_path_str,
                           &img.width,
                           &img.height,
                           &img.channels,
                           0);  // 0 = keep original number of channels

    return img;
}


int image_utils_save_jpeg(const char *filename, const Image *img, int quality) {
    int channels = img->channels;

    // JPEG doesn't support alpha; if you have RGBA, just ignore A
    if (channels == 4) {
        channels = 3;
    }

    // stbi_write_jpg expects tightly packed rows:
    // data size = width * height * channels
    int ok = stbi_write_jpg(
        filename,
        img->width,
        img->height,
        channels,
        img->pixels,
        quality  // 1–100, usually 90 is nice
    );

    return ok != 0; // 1 on success, 0 on failure
}

void image_utils_build_output_path(
    const char* output_file_name,
    const char* input_file,
    size_t out_size
){
    // 1. Find last '/' in the input path
    const char *filename = strrchr(input_file, '/');
    if (filename)
        filename++;  // skip past '/'
    else
        filename = input_file;  // no slash found → whole string is filename

    // 2. Build: OUTPUT_FILE_DIRECTORY + filename
    snprintf(output_file_name, out_size, "%s%s", OUTPUT_FILE_DIRECTORY, filename);
}

Image image_utils_crop_vertically(Image* input, int new_height) {
    int width = input->width;
    int channels = input->channels;

    // allocate new buffer
    unsigned char* new_pixels = (unsigned char*) malloc(width * new_height * channels);

    // copy the **bottom new_height rows**
    int offset_src = (input->height - new_height) * width * channels;
    memcpy(new_pixels, &input->pixels[offset_src], width * new_height * channels);

    Image result = { width, new_height, channels, new_pixels };
    return result;
}

unsigned char* image_utils_load_LUT_from_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Could not open LUT file: %s\n", path);
        return NULL;
    }

    const size_t LUT_SIZE = 256 * 256 * 256; // 16,777,216 bytes

    // Allocate host buffer
    unsigned char* lut = (unsigned char*)malloc(LUT_SIZE);
    if (!lut) {
        fprintf(stderr, "ERROR: Could not allocate host memory for LUT\n");
        fclose(f);
        return NULL;
    }

    // Read all bytes from the file
    size_t read_bytes = fread(lut, 1, LUT_SIZE, f);
    fclose(f);

    // Validate that we read exactly the expected amount
    if (read_bytes != LUT_SIZE) {
        fprintf(stderr,
                "ERROR: LUT file read %zu bytes, but expected %zu bytes.\n",
                read_bytes, LUT_SIZE);
        free(lut);
        return NULL;
    }

    return lut;
}



void image_utils_free_image(Image *img) {
    if (img->pixels) {
        free(img->pixels);
        img->pixels = NULL;
    }
}
