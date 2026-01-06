
/*
 * image_process.h
 * MNN C API for ImageProcess
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#ifndef MNN_IMAGE_PROCESS_H
#define MNN_IMAGE_PROCESS_H

#include "error_code.h"
#include "mnn_type.h"
#include "tensor.h"
#include <stdint.h>

#ifdef __cplusplus
  #include "MNN/HalideRuntime.h"
  #include "MNN/ImageProcess.hpp"
  #include "MNN/Matrix.h"
  #include "MNN/Rect.h"
  #include <memory>
extern "C" {
#endif

#ifdef __cplusplus
typedef std::shared_ptr<MNN::CV::ImageProcess> *mnn_cv_image_process_t;
typedef MNN::CV::Matrix                        *mnn_cv_matrix_t;
#else
typedef void *mnn_cv_image_process_t;
typedef void *mnn_cv_matrix_t;
#endif

typedef struct {
  /** data filter */
  int filterType;
  /** format of source data */
  int sourceFormat;
  /** format of destination data */
  int destFormat;

  // Only valid if the dest type is float
  float mean[4];
  float normal[4];

  /** edge wrapper */
  int wrap;
} mnn_image_process_config_t;

typedef struct {
  float x;
  float y;
} mnn_cv_point_t;

typedef struct {
  float left;
  float top;
  float right;
  float bottom;
} mnn_cv_rect_t;

// Matrix operations
MNN_C_API mnn_cv_matrix_t mnn_cv_matrix_create();
MNN_C_API void            mnn_cv_matrix_destroy(mnn_cv_matrix_t self);

// Matrix setters
MNN_C_API void mnn_cv_matrix_set(mnn_cv_matrix_t self, int index, float value);
MNN_C_API void mnn_cv_matrix_set_all(
    mnn_cv_matrix_t self,
    float           scaleX,
    float           skewX,
    float           transX,
    float           skewY,
    float           scaleY,
    float           transY,
    float           pers0,
    float           pers1,
    float           pers2
);
MNN_C_API void mnn_cv_matrix_set_translate(mnn_cv_matrix_t self, float dx, float dy);
MNN_C_API void
mnn_cv_matrix_set_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py);
MNN_C_API void mnn_cv_matrix_set_rotate(mnn_cv_matrix_t self, float degrees, float px, float py);
MNN_C_API void
mnn_cv_matrix_set_sincos(mnn_cv_matrix_t self, float sin, float cos, float px, float py);
MNN_C_API void mnn_cv_matrix_set_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py);
MNN_C_API void mnn_cv_matrix_set_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t a, mnn_cv_matrix_t b);

// Matrix getters
MNN_C_API float mnn_cv_matrix_get(mnn_cv_matrix_t self, int index);

// Matrix type masks
MNN_C_API int  mnn_cv_matrix_get_type(mnn_cv_matrix_t self);
MNN_C_API bool mnn_cv_matrix_is_identity(mnn_cv_matrix_t self);
MNN_C_API bool mnn_cv_matrix_is_scale_translate(mnn_cv_matrix_t self);
MNN_C_API bool mnn_cv_matrix_is_translate(mnn_cv_matrix_t self);
MNN_C_API bool mnn_cv_matrix_rect_stays_rect(mnn_cv_matrix_t self);

MNN_C_API void mnn_cv_matrix_get9(mnn_cv_matrix_t self, float *m);
MNN_C_API void mnn_cv_matrix_set9(mnn_cv_matrix_t self, const float *m);
MNN_C_API void mnn_cv_matrix_reset(mnn_cv_matrix_t self);
MNN_C_API void mnn_cv_matrix_set_identity(mnn_cv_matrix_t self);
MNN_C_API bool mnn_cv_matrix_set_rect_to_rect(
    mnn_cv_matrix_t self, mnn_cv_rect_t src, mnn_cv_rect_t dst, int scale_to_fit
);
MNN_C_API bool mnn_cv_matrix_set_poly_to_poly(
    mnn_cv_matrix_t self, mnn_cv_point_t *src, mnn_cv_point_t *dst, int count
);
MNN_C_API bool mnn_cv_matrix_invert(mnn_cv_matrix_t self, mnn_cv_matrix_t dst);
MNN_C_API bool
mnn_cv_matrix_map_points(mnn_cv_matrix_t self, mnn_cv_point_t *dst, mnn_cv_point_t *src, int count);
MNN_C_API void
mnn_cv_matrix_map_points_inplace(mnn_cv_matrix_t self, mnn_cv_point_t *points, int count);
MNN_C_API void
mnn_cv_matrix_map_xy(mnn_cv_matrix_t self, float x, float y, float *mapped_x, float *mapped_y);
MNN_C_API bool mnn_cv_matrix_map_rect(mnn_cv_matrix_t self, mnn_cv_rect_t *dst, mnn_cv_rect_t *src);
MNN_C_API void mnn_cv_matrix_map_rect_scale_translate(
    mnn_cv_matrix_t self, mnn_cv_rect_t *dst, mnn_cv_rect_t *src
);
MNN_C_API bool mnn_cv_matrix_cheap_equal_to(mnn_cv_matrix_t self, mnn_cv_matrix_t other);
// MNN_C_API float mnn_cv_matrix_get_min_scale(mnn_cv_matrix_t self);
// MNN_C_API float mnn_cv_matrix_get_max_scale(mnn_cv_matrix_t self);
MNN_C_API void mnn_cv_matrix_dirty_matrix_type_cache(mnn_cv_matrix_t self);

MNN_C_API void mnn_cv_matrix_pre_translate(mnn_cv_matrix_t self, float dx, float dy);
MNN_C_API void
mnn_cv_matrix_pre_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py);
MNN_C_API void mnn_cv_matrix_pre_rotate(mnn_cv_matrix_t self, float degrees, float px, float py);
MNN_C_API void mnn_cv_matrix_pre_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py);
MNN_C_API void mnn_cv_matrix_pre_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t other);

MNN_C_API void mnn_cv_matrix_post_translate(mnn_cv_matrix_t self, float dx, float dy);
MNN_C_API void
mnn_cv_matrix_post_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py);
MNN_C_API void mnn_cv_matrix_post_idiv(mnn_cv_matrix_t self, int divx, int divy);
MNN_C_API void mnn_cv_matrix_post_rotate(mnn_cv_matrix_t self, float degrees, float px, float py);
MNN_C_API void
mnn_cv_matrix_post_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py);
MNN_C_API void mnn_cv_matrix_post_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t other);

MNN_C_API void
mnn_cv_matrix_set_scale_translate(mnn_cv_matrix_t self, float sx, float sy, float tx, float ty);

////////////////// ImageProcess //////////////////////////
MNN_C_API mnn_cv_image_process_t mnn_cv_image_process_create(
    const int    sourceFormat,
    const int    destFormat,
    const float *means,
    const int    mean_count,
    const float *normals,
    const int    normal_count,
    const int    filterType,
    const int    wrap,
    mnn_tensor_t dst_tensor
);

MNN_C_API mnn_cv_image_process_t
mnn_cv_image_process_create_with_config(mnn_image_process_config_t config, mnn_tensor_t dst_tensor);
void mnn_cv_image_process_destroy(mnn_cv_image_process_t self);

// Matrix operations for ImageProcess
MNN_C_API mnn_cv_matrix_t mnn_cv_image_process_get_matrix(mnn_cv_image_process_t self);
MNN_C_API mnn_error_code_t
mnn_cv_image_process_set_matrix(mnn_cv_image_process_t self, mnn_cv_matrix_t matrix);
MNN_C_API mnn_error_code_t mnn_cv_image_process_convert(
    mnn_cv_image_process_t self, const uint8_t *src, int iw, int ih, int stride, mnn_tensor_t dest
);
MNN_C_API mnn_error_code_t mnn_cv_image_process_convert_1(
    mnn_cv_image_process_t self,
    const uint8_t         *src,
    int                    iw,
    int                    ih,
    int                    stride,
    void                  *dst,
    int                    ow,
    int                    oh,
    int                    outputBpp,
    int                    outputStride,
    halide_type_c_t        type
);
MNN_C_API mnn_tensor_t mnn_cv_image_process_create_image_tensor(
    halide_type_c_t type, int width, int height, int bytes_per_channel, void *p
);
MNN_C_API void mnn_cv_image_process_set_padding(mnn_cv_image_process_t self, uint8_t value);
MNN_C_API void mnn_cv_image_process_set_draw(mnn_cv_image_process_t self);
MNN_C_API void mnn_cv_image_process_draw(
    mnn_cv_image_process_t self,
    uint8_t               *img,
    int                    w,
    int                    h,
    int                    c,
    const int             *regions,
    int                    num,
    const uint8_t         *color
);

#ifdef __cplusplus
}
#endif

#endif
