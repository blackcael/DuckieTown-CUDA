#ifndef UTILS_H
#define UTILS_H


// - pixels is width * height * 3 bytes
// - Layout is row-major, interleaved RGB:
//   pixel (x, y) starts at index (y * width + x) * 3
//   R = pixels[idx + 0], G = pixels[idx + 1], B = pixels[idx + 2]

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int width;       // image width in pixels
    int height;      // image height in pixels
    int channels;    // number of channels (e.g., 3 = RGB, 4 = RGBA)
    unsigned char* pixels;  // pointer to width * height * channels bytes
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

int image_utils_save_jpeg(
    const char *filename, 
    const Image *img, 
    int quality
);

void image_utils_build_output_path(
    const char* output_file_name,
    const char* input_file,
    size_t out_size
);

// crops image from top down, leaving the bottom up
Image image_utils_crop_vertically(Image* input_image, int new_height);

unsigned char* image_utils_load_LUT_from_file(const char* path);

void image_utils_free_image(Image* img);

#ifdef __cplusplus
}
#endif


#endif //UTILS_H