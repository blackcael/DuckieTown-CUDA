#ifndef UTILS_H
#define UTILS_H

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// - pixels is width * height * 3 bytes
// - Layout is row-major, interleaved RGB:
//   pixel (x, y) starts at index (y * width + x) * 3
//   R = pixels[idx + 0], G = pixels[idx + 1], B = pixels[idx + 2]

typedef struct {
    int width;       // image width in pixels
    int height;      // image height in pixels
    int channels;    // number of channels (e.g., 3 = RGB, 4 = RGBA)
    char* pixels;  // pointer to width * height * channels bytes
} Image;

typedef struct {
    int x1;
    int x2;
    int y1;
    int y2;
} Line;

Image image_utils_load_image(
    char* file_path_str
);

void image_utils_free_image(
    Image* img
);

#endif //UTILS_H