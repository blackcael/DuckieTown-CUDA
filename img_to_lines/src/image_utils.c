#include <stdio.h>
#include <stdint.h>
#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image image_utils_load_image(char* file_path_str) {
    Image img = {0};

    img.pixels = stbi_load(file_path_str,
                           &img.width,
                           &img.height,
                           &img.channels,
                           0);  // 0 = keep original number of channels

    return img;
}

void image_utils_free_image(Image *img) {
    if (img->pixels) {
        free(img->pixels);
        img->pixels = NULL;
    }
}

