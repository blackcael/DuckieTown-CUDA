#include <stdio.h>
#include <stdint.h>
#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Image load_image(const char *path) {
    Image img = {0};

    img.pixels = stbi_load(path,
                           &img.width,
                           &img.height,
                           &img.channels,
                           0);  // 0 = keep original number of channels

    return img;
}

void free_image(Image *img) {
    if (img->pixels) {
        free(img->pixels);
        img->pixels = NULL;
    }
}

