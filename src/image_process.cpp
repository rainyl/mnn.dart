#include "image_process.h"
#include "MNN/ImageProcess.hpp"
#include "MNN/Matrix.h"
#include "error_code.h"
#include <memory>

mnn_cv_matrix_t mnn_cv_matrix_create() { return new MNN::CV::Matrix(); }

void mnn_cv_matrix_destroy(mnn_cv_matrix_t self) {
  if (self != nullptr) { delete (MNN::CV::Matrix *)self; self = nullptr; }
}

void mnn_cv_matrix_set(mnn_cv_matrix_t self, int index, float value) {
  ((MNN::CV::Matrix *)self)->set(index, value);
}
void mnn_cv_matrix_set_all(
    mnn_cv_matrix_t self,
    float scaleX,
    float skewX,
    float transX,
    float skewY,
    float scaleY,
    float transY,
    float pers0,
    float pers1,
    float pers2
) {
  ((MNN::CV::Matrix *)self)
      ->setAll(scaleX, skewX, transX, skewY, scaleY, transY, pers0, pers1, pers2);
}

void mnn_cv_matrix_set_translate(mnn_cv_matrix_t self, float dx, float dy) {
  ((MNN::CV::Matrix *)self)->setTranslate(dx, dy);
}
void mnn_cv_matrix_set_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py) {
  ((MNN::CV::Matrix *)self)->setScale(sx, sy, px, py);
}
void mnn_cv_matrix_set_rotate(mnn_cv_matrix_t self, float degrees, float px, float py) {
  ((MNN::CV::Matrix *)self)->setRotate(degrees, px, py);
}

void mnn_cv_matrix_set_sincos(mnn_cv_matrix_t self, float sin, float cos, float px, float py) {
  ((MNN::CV::Matrix *)self)->setSinCos(sin, cos, px, py);
}

void mnn_cv_matrix_set_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py) {
  ((MNN::CV::Matrix *)self)->setSkew(kx, ky, px, py);
}

float mnn_cv_matrix_get(mnn_cv_matrix_t self, int index) {
  return ((MNN::CV::Matrix *)self)->get(index);
}

int mnn_cv_matrix_get_type(mnn_cv_matrix_t self) { return ((MNN::CV::Matrix *)self)->getType(); }

bool mnn_cv_matrix_is_identity(mnn_cv_matrix_t self) {
  return ((MNN::CV::Matrix *)self)->isIdentity();
}

bool mnn_cv_matrix_is_scale_translate(mnn_cv_matrix_t self) {
  return ((MNN::CV::Matrix *)self)->isScaleTranslate();
}

bool mnn_cv_matrix_is_translate(mnn_cv_matrix_t self) {
  return ((MNN::CV::Matrix *)self)->isTranslate();
}

bool mnn_cv_matrix_rect_stays_rect(mnn_cv_matrix_t self) {
  return ((MNN::CV::Matrix *)self)->rectStaysRect();
}

void mnn_cv_matrix_get9(mnn_cv_matrix_t self, float *m) { ((MNN::CV::Matrix *)self)->get9(m); }

void mnn_cv_matrix_set9(mnn_cv_matrix_t self, const float *m) {
  ((MNN::CV::Matrix *)self)->set9(m);
}

void mnn_cv_matrix_reset(mnn_cv_matrix_t self) { ((MNN::CV::Matrix *)self)->reset(); }

void mnn_cv_matrix_set_identity(mnn_cv_matrix_t self) { ((MNN::CV::Matrix *)self)->setIdentity(); }

void mnn_cv_matrix_set_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t a, mnn_cv_matrix_t b) {
  ((MNN::CV::Matrix *)self)->setConcat(*((MNN::CV::Matrix *)a), *((MNN::CV::Matrix *)b));
}

bool mnn_cv_matrix_set_rect_to_rect(
    mnn_cv_matrix_t self, mnn_cv_rect_t src, mnn_cv_rect_t dst, int scale_to_fit
) {
  auto s = MNN::CV::Rect::MakeLTRB(src.left, src.top, src.right, src.bottom);
  auto d = MNN::CV::Rect::MakeLTRB(dst.left, dst.top, dst.right, dst.bottom);
  return ((MNN::CV::Matrix *)self)->setRectToRect(s, d, (MNN::CV::Matrix::ScaleToFit)scale_to_fit);
}

bool mnn_cv_matrix_set_poly_to_poly(
    mnn_cv_matrix_t self, mnn_cv_point_t *src, mnn_cv_point_t *dst, int count
) {
  std::vector<MNN::CV::Point> s(count);
  std::vector<MNN::CV::Point> d(count);
  for (int i = 0; i < count; i++) {
    s[i].set(src[i].x, src[i].y);
    d[i].set(dst[i].x, dst[i].y);
  }
  return ((MNN::CV::Matrix *)self)->setPolyToPoly(s.data(), d.data(), count);
}

bool mnn_cv_matrix_invert(mnn_cv_matrix_t self, mnn_cv_matrix_t dst) {
  return ((MNN::CV::Matrix *)self)->invert((MNN::CV::Matrix *)dst);
}

bool mnn_cv_matrix_map_points(
    mnn_cv_matrix_t self, mnn_cv_point_t *dst, mnn_cv_point_t *src, int count
) {
  std::vector<MNN::CV::Point> s(count);
  std::vector<MNN::CV::Point> d(count);
  for (int i = 0; i < count; i++) { s[i].set(src[i].x, src[i].y); }
  ((MNN::CV::Matrix *)self)->mapPoints(d.data(), s.data(), count);
  for (int i = 0; i < count; i++) {
    dst[i].x = d[i].fX;
    dst[i].y = d[i].fY;
  }
  return true;
}

void mnn_cv_matrix_map_points_inplace(mnn_cv_matrix_t self, mnn_cv_point_t *points, int count) {
  std::vector<MNN::CV::Point> p(count);
  for (int i = 0; i < count; i++) { p[i].set(points[i].x, points[i].y); }
  ((MNN::CV::Matrix *)self)->mapPoints(p.data(), p.data(), count);
  for (int i = 0; i < count; i++) {
    points[i].x = p[i].fX;
    points[i].y = p[i].fY;
  }
}

void mnn_cv_matrix_map_xy(
    mnn_cv_matrix_t self, float x, float y, float *mapped_x, float *mapped_y
) {
  MNN::CV::Point dst;
  ((MNN::CV::Matrix *)self)->mapXY(x, y, &dst);
  *mapped_x = dst.fX;
  *mapped_y = dst.fY;
}

bool mnn_cv_matrix_map_rect(mnn_cv_matrix_t self, mnn_cv_rect_t *dst, mnn_cv_rect_t *src) {
  auto _src = MNN::CV::Rect::MakeLTRB(src->left, src->top, src->right, src->bottom);
  MNN::CV::Rect _dst;
  auto rval = ((MNN::CV::Matrix *)self)->mapRect(&_dst, _src);
  dst->left = _dst.left();
  dst->top = _dst.top();
  dst->right = _dst.right();
  dst->bottom = _dst.bottom();
  return rval;
}

void mnn_cv_matrix_map_rect_scale_translate(
    mnn_cv_matrix_t self, mnn_cv_rect_t *dst, mnn_cv_rect_t *src
) {
  MNN::CV::Rect _dst;
  auto _src = MNN::CV::Rect::MakeLTRB(src->left, src->top, src->right, src->bottom);
  ((MNN::CV::Matrix *)self)->mapRectScaleTranslate(&_dst, _src);
  dst->left = _dst.left();
  dst->top = _dst.top();
  dst->right = _dst.right();
  dst->bottom = _dst.bottom();
}

bool mnn_cv_matrix_cheap_equal_to(mnn_cv_matrix_t self, mnn_cv_matrix_t other) {
  return ((MNN::CV::Matrix *)self)->cheapEqualTo(*((MNN::CV::Matrix *)other));
}

// float mnn_cv_matrix_get_min_scale(mnn_cv_matrix_t self) {
//   return ((MNN::CV::Matrix *)self)->getMinScale();
// }

// float mnn_cv_matrix_get_max_scale(mnn_cv_matrix_t self) {
//   return ((MNN::CV::Matrix *)self)->getMaxScale();
// }

void mnn_cv_matrix_dirty_matrix_type_cache(mnn_cv_matrix_t self) {
  ((MNN::CV::Matrix *)self)->dirtyMatrixTypeCache();
}

MNN_C_API void mnn_cv_matrix_pre_translate(mnn_cv_matrix_t self, float dx, float dy) {
  ((MNN::CV::Matrix *)self)->preTranslate(dx, dy);
}
MNN_C_API void
mnn_cv_matrix_pre_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py) {
  ((MNN::CV::Matrix *)self)->preScale(sx, sy, px, py);
}
MNN_C_API void mnn_cv_matrix_pre_rotate(mnn_cv_matrix_t self, float degrees, float px, float py) {
  ((MNN::CV::Matrix *)self)->preRotate(degrees, px, py);
}
MNN_C_API void
mnn_cv_matrix_pre_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py) {
  ((MNN::CV::Matrix *)self)->preSkew(kx, ky, px, py);
}
MNN_C_API void mnn_cv_matrix_pre_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t other) {
  ((MNN::CV::Matrix *)self)->preConcat(*(MNN::CV::Matrix *)other);
}

MNN_C_API void mnn_cv_matrix_post_translate(mnn_cv_matrix_t self, float dx, float dy) {
  ((MNN::CV::Matrix *)self)->postTranslate(dx, dy);
}
MNN_C_API void
mnn_cv_matrix_post_scale(mnn_cv_matrix_t self, float sx, float sy, float px, float py) {
  ((MNN::CV::Matrix *)self)->postScale(sx, sy, px, py);
}
MNN_C_API void mnn_cv_matrix_post_idiv(mnn_cv_matrix_t self, int divx, int divy) {
  ((MNN::CV::Matrix *)self)->postIDiv(divx, divy);
}
MNN_C_API void mnn_cv_matrix_post_rotate(mnn_cv_matrix_t self, float degrees, float px, float py) {
  ((MNN::CV::Matrix *)self)->postRotate(degrees, px, py);
}
MNN_C_API void
mnn_cv_matrix_post_skew(mnn_cv_matrix_t self, float kx, float ky, float px, float py) {
  ((MNN::CV::Matrix *)self)->postSkew(kx, ky, px, py);
}
MNN_C_API void mnn_cv_matrix_post_concat(mnn_cv_matrix_t self, mnn_cv_matrix_t other) {
  ((MNN::CV::Matrix *)self)->postConcat(*(MNN::CV::Matrix *)other);
}

MNN_C_API void
mnn_cv_matrix_set_scale_translate(mnn_cv_matrix_t self, float sx, float sy, float tx, float ty) {
  ((MNN::CV::Matrix *)self)->setScaleTranslate(sx, sy, tx, ty);
}

////////////////// ImageProcess //////////////////////////
mnn_cv_image_process_t mnn_cv_image_process_create(
    const int sourceFormat,
    const int destFormat,
    const float *means,
    const int mean_count,
    const float *normals,
    const int normal_count,
    const int filterType,
    const int wrap,
    mnn_tensor_t dst_tensor
) {
  MNN::CV::ImageProcess::Config config;
  config.sourceFormat = static_cast<MNN::CV::ImageFormat>(sourceFormat);
  config.destFormat = static_cast<MNN::CV::ImageFormat>(destFormat);
  config.filterType = static_cast<MNN::CV::Filter>(filterType);
  config.wrap = static_cast<MNN::CV::Wrap>(wrap);

  if (means != nullptr && mean_count > 0) {
    for (int i = 0; i < mean_count && i < 4; i++) { config.mean[i] = means[i]; }
  }

  if (normals != nullptr && normal_count > 0) {
    for (int i = 0; i < normal_count && i < 4; i++) { config.normal[i] = normals[i]; }
  }
  auto p = MNN::CV::ImageProcess::create(config, (MNN::Tensor *)dst_tensor);

  return new std::shared_ptr<MNN::CV::ImageProcess>(p, MNN::CV::ImageProcess::destroy);
}

mnn_cv_image_process_t mnn_cv_image_process_create_with_config(
    mnn_image_process_config_t config, mnn_tensor_t dst_tensor
) {
  MNN::CV::ImageProcess::Config cppConfig;
  cppConfig.sourceFormat = static_cast<MNN::CV::ImageFormat>(config.sourceFormat);
  cppConfig.destFormat = static_cast<MNN::CV::ImageFormat>(config.destFormat);
  cppConfig.filterType = static_cast<MNN::CV::Filter>(config.filterType);
  cppConfig.wrap = static_cast<MNN::CV::Wrap>(config.wrap);

  for (int i = 0; i < 4; i++) {
    cppConfig.mean[i] = config.mean[i];
    cppConfig.normal[i] = config.normal[i];
  }

  auto p = MNN::CV::ImageProcess::create(cppConfig, (MNN::Tensor *)dst_tensor);
  return new std::shared_ptr<MNN::CV::ImageProcess>(p, MNN::CV::ImageProcess::destroy);
}

void mnn_cv_image_process_destroy(mnn_cv_image_process_t self) {
  if (self != nullptr) {
    self->reset();
    delete self;
    self = nullptr;
  }
}

mnn_cv_matrix_t mnn_cv_image_process_get_matrix(mnn_cv_image_process_t self) {
  auto m = self->get()->matrix();
  return new MNN::CV::Matrix(m);
}

mnn_error_code_t
mnn_cv_image_process_set_matrix(mnn_cv_image_process_t self, mnn_cv_matrix_t matrix) {
  if (!self || !matrix) return MNNC_INVALID_PTR;
  self->get()->setMatrix(*matrix);
  return MNNC_NO_ERROR;
}

mnn_error_code_t mnn_cv_image_process_convert(
    mnn_cv_image_process_t self, const uint8_t *src, int iw, int ih, int stride, mnn_tensor_t dest
) {
  auto code = self->get()->convert(src, iw, ih, stride, dest);
  return static_cast<mnn_error_code_t>(code);
}

mnn_error_code_t mnn_cv_image_process_convert_1(
    mnn_cv_image_process_t self,
    const uint8_t *src,
    int iw,
    int ih,
    int stride,
    void *dst,
    int ow,
    int oh,
    int outputBpp,
    int outputStride,
    halide_type_c_t type
) {
  const auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  auto code =
      self->get()->convert(src, iw, ih, stride, dst, ow, oh, outputBpp, outputStride, _type);
  return static_cast<mnn_error_code_t>(code);
}

mnn_tensor_t mnn_cv_image_process_create_image_tensor(
    halide_type_c_t type, int width, int height, int bytes_per_channel, void *p
) {
  const auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  auto tensor =
      MNN::CV::ImageProcess::createImageTensor(_type, width, height, bytes_per_channel, p);
  return (mnn_tensor_t)tensor;
}

void mnn_cv_image_process_set_padding(mnn_cv_image_process_t self, uint8_t value) {
  self->get()->setPadding(value);
}

void mnn_cv_image_process_set_draw(mnn_cv_image_process_t self) { self->get()->setDraw(); }

void mnn_cv_image_process_draw(
    mnn_cv_image_process_t self,
    uint8_t *img,
    int w,
    int h,
    int c,
    const int *regions,
    int num,
    const uint8_t *color
) {
  self->get()->draw(img, w, h, c, regions, num, color);
}
