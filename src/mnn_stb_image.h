#ifndef MNN_STB_IMAGE_H
#define MNN_STB_IMAGE_H

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include <stdint.h>
#include "stb_image.h"
#include "stb_image_resize2.h"

#ifdef __cplusplus
extern "C" {
#endif

int mnn_stbi_rgb_flatten_uint8(stbi_uc *data, int width, int height, int channels);
int mnn_stbi_add_f32(stbi_uc *data, int width, int height, int channels, float *value);
int mnn_stbi_sub_f32(stbi_uc *data, int width, int height, int channels, float *value);
int mnn_stbi_div_f32(stbi_uc *data, int width, int height, int channels, float *value);
int mnn_stbi_mul_f32(stbi_uc *data, int width, int height, int channels, float *value);

#ifdef __cplusplus
}
#endif

#endif // MNN_STB_IMAGE_H
