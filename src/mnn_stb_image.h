#ifndef MNN_STB_IMAGE_H
#define MNN_STB_IMAGE_H

#ifdef STB_IMAGE_STATIC
    #define STBIDEF static
#else
    #ifdef _WIN32
        #ifdef STB_IMAGE_EXPORTS
            #define STBIDEF __declspec(dllexport)
            #define STBIRDEF extern "C" __declspec(dllexport)
        #else
            #define STBIDEF __declspec(dllimport)
            #define STBIRDEF extern "C" __declspec(dllimport)
        #endif
    #else
        #define STBIDEF extern
        #ifdef __cplusplus
            #define STBIRDEF extern "C"
        #else
            #define STBIRDEF extern
        #endif
    #endif
#endif

#include "stb_image.h"
#include "stb_image_resize2.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

STBIDEF int mnn_stbi_rgb_flatten_uint8(stbi_uc *data, int width, int height, int channels);
STBIDEF int mnn_stbi_add_f32(float *data, int width, int height, int channels, float *value, float *dst);
STBIDEF int mnn_stbi_sub_f32(float *data, int width, int height, int channels, float *value, float *dst);
STBIDEF int mnn_stbi_div_f32(float *data, int width, int height, int channels, float *value, float *dst);
STBIDEF int mnn_stbi_mul_f32(float *data, int width, int height, int channels, float *value, float *dst);

#ifdef __cplusplus
}
#endif

#endif // MNN_STB_IMAGE_H
