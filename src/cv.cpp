//
// Created by rainy on 2026/1/1.
//

#include "cv.h"

#include <set>

#include "mnn_stb_image.h"
#ifdef __cplusplus
extern "C" {
#endif
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#include "stb_image.h"
#include "stb_image_write.h"
#ifdef __cplusplus
}
#endif

// helper functions
static void writeFunc(void *context, void *data, int size) {
  std::vector<uint8_t> *ctx = (std::vector<uint8_t> *)context;
  ctx->insert(ctx->end(), (uint8_t *)data, (uint8_t *)data + size);
}

static std::string getExt(std::string name) {
  auto ext = name.substr(name.rfind('.') + 1, -1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  return ext;
}

void mnn_cv_static_getVARPSize(VARP_t var, int *height, int *width, int *channel) {
  MNN::CV::getVARPSize(*var, height, width, channel);
}

int mnn_cv_getVARPHeight(VARP_t var) { return MNN::CV::getVARPHeight(*var); }

int mnn_cv_getVARPWidth(VARP_t var) { return MNN::CV::getVARPWidth(*var); }

int mnn_cv_getVARPChannel(VARP_t var) { return MNN::CV::getVARPChannel(*var); }

int mnn_cv_getVARPByte(VARP_t var) { return MNN::CV::getVARPByte(*var); }

VARP_t mnn_cv_buildImgVARP(uint8_t *img, int height, int width, int channel, int flags) {
  auto rgb = MNN::Express::_Const(
      img, {height, width, channel}, MNN::Express::NHWC, halide_type_of<uint8_t>()
  );
  MNN::Express::VARP res;
  switch (flags) {
  case IMREAD_COLOR: res = MNN::CV::cvtColor(rgb, MNN::CV::COLOR_RGB2BGR); break;
  case IMREAD_GRAYSCALE: res = MNN::CV::cvtColor(rgb, MNN::CV::COLOR_RGB2GRAY); break;
  case IMREAD_ANYDEPTH: res = MNN::Express::_Cast<float>(rgb); break;
  case IMREAD_COLOR_RGB: return new MNN::Express::VARP(rgb); break;
  default: MNN_ERROR("Don't support imread flags!"); return new MNN::Express::VARP(rgb);
  }
  return new MNN::Express::VARP(res);
}

MNN_C_API VARP_t mnn_cv_buildImgVarpYuvNV21(uint8_t *src, int height, int width, int flags) {
  auto rgb     = (uint8_t *)malloc(sizeof(uint8_t) * width * height * 3);
  auto nvStart = width * height;
  int  index = 0, rgbaIndex = 0;
  int  y, u, v;
  int  r, g, b;
  int  nvIndex = 0;

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      nvIndex = i / 2 * width + j - j % 2;

      y = src[rgbaIndex];
      u = src[nvStart + nvIndex];
      v = src[nvStart + nvIndex + 1];

      // r = y + (140 * (v - 128)) ~/ 100; // r
      // g = y - (34 * (u - 128)) ~/ 100 - (71 * (v - 128)) ~/ 100; // g
      // b = y + (177 * (u - 128)) ~/ 100; // b
      r = y + (1.13983 * (v - 128));                         // r
      g = y - (0.39465 * (u - 128)) - (0.58060 * (v - 128)); // g
      b = y + (2.03211 * (u - 128));                         // b

      r = r < 0 ? 0 : r > 255 ? 255 : r;
      g = g < 0 ? 0 : g > 255 ? 255 : g;
      b = b < 0 ? 0 : b > 255 ? 255 : b;

      // index = rgbaIndex % width + (height - i - 1) * width;
      index              = rgbaIndex % width + i * width;
      rgb[index * 3 + 0] = b;
      rgb[index * 3 + 1] = g;
      rgb[index * 3 + 2] = r;
      rgbaIndex++;
    }
  }

  auto res = mnn_cv_buildImgVARP(rgb, height, width, 3, flags);
  free(rgb);
  return res;
}

// core
bool mnn_cv_solve(VARP_t src1, VARP_t src2, int flags, VARP_t *out) {
  auto v = MNN::CV::solve(*src1, *src2, flags);
  *out   = new MNN::Express::VARP(v.second);
  return v.first;
}

// calib3d
VARP_t mnn_cv_Rodrigues(VARP_t src) { return new MNN::Express::VARP(MNN::CV::Rodrigues(*src)); }

void mnn_cv_solvePnP(
    VARP_t  objectPoints,
    VARP_t  imagePoints,
    VARP_t  cameraMatrix,
    VARP_t  distCoeffs,
    bool    useExtrinsicGuess,
    VARP_t *out1,
    VARP_t *out2
) {
  auto v =
      MNN::CV::solvePnP(*objectPoints, *imagePoints, *cameraMatrix, *distCoeffs, useExtrinsicGuess);
  *out1 = new MNN::Express::VARP(v.first);
  *out2 = new MNN::Express::VARP(v.second);
}

// imgcodecs.hpp
bool mnn_cv_haveImageReader(char *filename) {
  int width, height, channel;
  return stbi_info(filename, &width, &height, &channel);
}
bool mnn_cv_haveImageReaderFromMemory(uint8_t *buf, size_t length) {
  int width, height, channel;
  return stbi_info_from_memory(buf, length, &width, &height, &channel);
}
bool mnn_cv_haveImageWriter(char *filename) {
  static const std::set<std::string> supportImages{
      "jpg",
      "jpeg",
      "png",
      "bmp", /* "gif", "psd", "pic", "pnm", "hdr", "tga" */
  };
  return supportImages.find(getExt(filename)) != supportImages.end();
}
VARP_t mnn_cv_imdecode(uint8_t *buf, size_t length, int flags) {
  int  width, height, channel;
  auto img = stbi_load_from_memory(buf, length, &width, &height, &channel, 3);
  if (nullptr == img) {
    MNN_ERROR("Can't decode\n");
    return nullptr;
  }
  return mnn_cv_buildImgVARP(img, height, width, 3, flags);
}

bool mnn_cv_imencode(char *ext, VARP_t img, int *params, size_t params_length, VecU8 *out) {
  MNN::Express::VARP rgb = MNN::CV::cvtColor(*img, MNN::CV::COLOR_BGR2RGB);
  int                height, width, channel;
  MNN::CV::getVARPSize(rgb, &height, &width, &channel);
  auto                 _ext = getExt(ext);
  bool                 res  = false;
  std::vector<uint8_t> buf;
  if (_ext == "jpg" || "jpeg") {
    int quality = 95;
    for (size_t i = 0; i < params_length; i += 2) {
      if (params[i] == IMWRITE_JPEG_QUALITY) {
        quality = params[i + 1];
        break;
      }
    }
    res = stbi_write_jpg_to_func(
        writeFunc, (void *)&buf, width, height, channel, rgb->readMap<uint8_t>(), quality
    );
  }
  if (_ext == "png") {
    res = stbi_write_png_to_func(
        writeFunc, (void *)&buf, width, height, channel, rgb->readMap<uint8_t>(), 0
    );
  }
  if (_ext == "bmp") {
    res = stbi_write_bmp_to_func(
        writeFunc, (void *)&buf, width, height, channel, rgb->readMap<uint8_t>()
    );
  }
  *out = new std::vector<uint8_t>(buf);
  return res;
}

VARP_t mnn_cv_imread(char *filename, int flags) {
  int  width, height, channel;
  auto img = stbi_load(filename, &width, &height, &channel, 3);
  if (nullptr == img) {
    MNN_ERROR("Can't open %s\n", filename);
    return nullptr;
  }
  return mnn_cv_buildImgVARP(img, height, width, 3, flags);
}

bool mnn_cv_imwrite(char *filename, VARP_t img, int *params, size_t params_length) {
  auto _img = *img;
  if (_img->getInfo()->type != halide_type_of<uint8_t>()) {
    _img = MNN::Express::_Cast<uint8_t>(_img);
  }
  int height, width, channel;
  MNN::CV::getVARPSize(_img, &height, &width, &channel);
  if (channel == 3) {
    _img = MNN::CV::cvtColor(_img, MNN::CV::COLOR_BGR2RGB);
  } else {
    MNN_ERROR("MNN cv imwrite just support RGB/BGR format.");
  }
  auto ext = getExt(filename);
  if (ext == "jpg" || ext == "jpeg") {
    int quality = 95;
    for (size_t i = 0; i < params_length; i += 2) {
      if (params[i] == IMWRITE_JPEG_QUALITY) {
        quality = params[i + 1];
        break;
      }
    }
    return stbi_write_jpg(filename, width, height, channel, _img->readMap<uint8_t>(), quality);
  }
  if (ext == "png") {
    return stbi_write_png(filename, width, height, channel, _img->readMap<uint8_t>(), 0);
  }
  if (ext == "bmp") {
    return stbi_write_bmp(filename, width, height, channel, _img->readMap<uint8_t>());
  }
  return false;
}

// structural.hpp
MNN_C_API VecVARP_t mnn_cv_findContours(VARP_t image, int mode, int method, mnn_cv_point_t offset) {
  MNN::CV::Point _offset{};
  _offset.set(offset.x, offset.y);
  auto v = MNN::CV::findContours(*image, mode, method, _offset);
  return new std::vector<MNN::Express::VARP>(v);
}

MNN_C_API double mnn_cv_contourArea(VARP_t _contour, bool oriented) {
  return MNN::CV::contourArea(*_contour, oriented);
}

MNN_C_API VecI32 mnn_cv_convexHull(VARP_t _points, bool clockwise, bool returnPoints) {
  auto v = MNN::CV::convexHull(*_points, clockwise, returnPoints);
  return new std::vector<int>(v);
}

MNN_C_API mnn_cv_rotated_rect_t *mnn_cv_minAreaRect(VARP_t _points) {
  auto v = MNN::CV::minAreaRect(*_points);
  return new mnn_cv_rotated_rect_t{v.center.x, v.center.y, v.size.width, v.size.height, v.angle};
}

MNN_C_API mnn_cv_rect_t *mnn_cv_boundingRect(VARP_t points) {
  auto v = MNN::CV::boundingRect(*points);
  return new mnn_cv_rect_t{
      static_cast<float>(v.x),
      static_cast<float>(v.y),
      static_cast<float>(v.width),
      static_cast<float>(v.height)
  };
}

MNN_C_API VARP_t mnn_cv_boxPoints(mnn_cv_rotated_rect_t box) {
  auto _box   = MNN::CV::RotatedRect{};
  _box.center = MNN::CV::Point2f{box.center_x, box.center_y};
  _box.size   = MNN::CV::Size2f{box.width, box.height};
  _box.angle  = box.angle;
  auto v      = MNN::CV::boxPoints(_box);
  return new MNN::Express::VARP(v);
}

// miscellaneous.hpp
MNN_C_API VARP_t mnn_cv_adaptiveThreshold(
    VARP_t src, double max_value, int adaptiveMethod, int thresholdType, int blockSize, double C
) {
  return new MNN::Express::VARP(
      MNN::CV::adaptiveThreshold(*src, max_value, adaptiveMethod, thresholdType, blockSize, C)
  );
}

MNN_C_API VARP_t mnn_cv_blendLinear(VARP_t src1, VARP_t src2, VARP_t weight1, VARP_t weight2) {
  return new MNN::Express::VARP(MNN::CV::blendLinear(*src1, *src2, *weight1, *weight2));
}

// MNN_C_API void mnn_cv_distanceTransform(
//     VARP_t src, VARP_t *dst, VARP_t *labels, int distanceType, int maskSize, int labelType
// ) {
//     auto _dst = new MNN::Express::VARP();
//     auto _labels = new MNN::Express::VARP();
//   MNN::CV::distanceTransform(*src, *_dst, *_labels, distanceType, maskSize, labelType);
//     *dst = _dst;
//     *labels = _labels;
// }
// MNN_C_API int mnn_cv_floodFill(VARP_t image, int seedPoint_x, int seedPoint_y, float newVal) {
//   return MNN::CV::floodFill(*image, seedPoint_x, seedPoint_y, newVal);
// }
// MNN_C_API VARP_t mnn_cv_integral(VARP_t src, int sdepth) {
//   return new MNN::Express::VARP(MNN::CV::integral(*src, sdepth));
// }
MNN_C_API VARP_t mnn_cv_threshold(VARP_t src, double thresh, double maxval, int type) {
  return new MNN::Express::VARP(MNN::CV::threshold(*src, thresh, maxval, type));
}

// histograms.hpp
MNN_C_API VARP_t mnn_cv_calcHist(
    VecVARP_t images, VecI32 channels, VARP_t mask, VecI32 hist_size, VecF32 ranges, bool accumulate
) {
  return new MNN::Express::VARP(
      MNN::CV::calcHist(*images, *channels, *mask, *hist_size, *ranges, accumulate)
  );
}

// geometric.hpp
MNN_C_API mnn_cv_matrix_t
mnn_cv_getAffineTransform(const mnn_cv_point_t src[], const mnn_cv_point_t dst[]) {
  auto _src = new MNN::CV::Point[3];
  auto _dst = new MNN::CV::Point[3];
  for (int i = 0; i < 3; i++) {
    MNN::CV::Point psrc{};
    MNN::CV::Point pdst{};
    psrc.set(src[i].x, src[i].y);
    pdst.set(dst[i].x, dst[i].y);
    _src[i] = psrc;
    _dst[i] = pdst;
  }
  auto matrix = MNN::CV::getAffineTransform(_src, _dst);
  delete[] _src;
  delete[] _dst;
  return new MNN::CV::Matrix(matrix);
}

MNN_C_API mnn_cv_matrix_t
mnn_cv_getPerspectiveTransform(const mnn_cv_point_t src[], const mnn_cv_point_t dst[]) {
  auto _src = new MNN::CV::Point[4];
  auto _dst = new MNN::CV::Point[4];
  for (int i = 0; i < 4; i++) {
    MNN::CV::Point psrc = {src[i].x, src[i].y};
    MNN::CV::Point pdst = {dst[i].x, dst[i].y};
    _src[i]             = psrc;
    _dst[i]             = pdst;
  }
  auto matrix = MNN::CV::getPerspectiveTransform(_src, _dst);
  delete[] _src;
  delete[] _dst;
  return new MNN::CV::Matrix(matrix);
}

MNN_C_API VARP_t
mnn_cv_getRectSubPix(VARP_t image, mnn_cv_size2i_t patchSize, mnn_cv_point_t center) {
  MNN::CV::Size  _patchSize = {patchSize.width, patchSize.height};
  MNN::CV::Point _center    = {center.x, center.y};
  return new MNN::Express::VARP(MNN::CV::getRectSubPix(*image, _patchSize, _center));
}

MNN_C_API mnn_cv_matrix_t
mnn_cv_getRotationMatrix2D(mnn_cv_point_t center, double angle, double scale) {
  MNN::CV::Point _center = {center.x, center.y};
  return new MNN::CV::Matrix(MNN::CV::getRotationMatrix2D(_center, angle, scale));
}

MNN_C_API mnn_cv_matrix_t mnn_cv_invertAffineTransform(mnn_cv_matrix_t M) {
  return new MNN::CV::Matrix(MNN::CV::invertAffineTransform(*M));
}

MNN_C_API VARP_t mnn_cv_remap(
    VARP_t src, VARP_t map1, VARP_t map2, int interpolation, int borderMode, int borderValue
) {
  return new MNN::Express::VARP(
      MNN::CV::remap(*src, *map1, *map2, interpolation, borderMode, borderValue)
  );
}

MNN_C_API VARP_t mnn_cv_resize(
    VARP_t          src,
    mnn_cv_size2i_t dsize,
    double          fx,
    double          fy,
    int             interpolation,
    int             code,
    VecF32          mean,
    VecF32          norm
) {
  MNN::CV::Size _dsize = {dsize.width, dsize.height};
  return new MNN::Express::VARP(
      MNN::CV::resize(*src, _dsize, fx, fy, interpolation, code, *mean, *norm)
  );
}

MNN_C_API VARP_t mnn_cv_warpAffine(
    VARP_t          src,
    mnn_cv_matrix_t M,
    mnn_cv_size2i_t dsize,
    int             flags,
    int             borderMode,
    int             borderValue,
    int             code,
    VecF32          mean,
    VecF32          norm
) {
  MNN::CV::Size _dsize = {dsize.width, dsize.height};
  return new MNN::Express::VARP(
      MNN::CV::warpAffine(*src, *M, _dsize, flags, borderMode, borderValue, code, *mean, *norm)
  );
}

MNN_C_API VARP_t mnn_cv_warpPerspective(
    VARP_t src, mnn_cv_matrix_t M, mnn_cv_size2i_t dsize, int flags, int borderMode, int borderValue
) {
  MNN::CV::Size _dsize = {dsize.width, dsize.height};
  return new MNN::Express::VARP(
      MNN::CV::warpPerspective(*src, *M, _dsize, flags, borderMode, borderValue)
  );
}

MNN_C_API VARP_t mnn_cv_undistortPoints(VARP_t src, VARP_t cameraMatrix, VARP_t distCoeffs) {
  return new MNN::Express::VARP(MNN::CV::undistortPoints(*src, *cameraMatrix, *distCoeffs));
}

// filter.hpp
MNN_C_API VARP_t
mnn_cv_bilateralFilter(VARP_t src, int d, double sigmaColor, double sigmaSpace, int borderType) {
  return new MNN::Express::VARP(
      MNN::CV::bilateralFilter(*src, d, sigmaColor, sigmaSpace, borderType)
  );
}

MNN_C_API VARP_t mnn_cv_blur(VARP_t src, mnn_cv_size2i_t ksize, int borderType) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::blur(*src, _ksize, borderType));
}

MNN_C_API VARP_t
mnn_cv_boxFilter(VARP_t src, int ddepth, mnn_cv_size2i_t ksize, bool normalize, int borderType) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::boxFilter(*src, ddepth, _ksize, normalize, borderType));
}

MNN_C_API VARP_t mnn_cv_dilate(VARP_t src, VARP_t kernel, int iterations, int borderType) {
  return new MNN::Express::VARP(MNN::CV::dilate(*src, *kernel, iterations, borderType));
}

MNN_C_API VARP_t mnn_cv_erode(VARP_t src, VARP_t kernel, int iterations, int borderType) {
  return new MNN::Express::VARP(MNN::CV::erode(*src, *kernel, iterations, borderType));
}

MNN_C_API VARP_t
mnn_cv_filter2D(VARP_t src, int ddepth, VARP_t kernel, double delta, int borderType) {
  return new MNN::Express::VARP(MNN::CV::filter2D(*src, ddepth, *kernel, delta, borderType));
}

MNN_C_API VARP_t mnn_cv_GaussianBlur(
    VARP_t src, mnn_cv_size2i_t ksize, double sigmaX, double sigmaY, int borderType
) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::GaussianBlur(*src, _ksize, sigmaX, sigmaY, borderType));
}

// MNN_C_API std::pair<VARP_t, VARP_t> getDerivKernels(int dx, int dy, int ksize,
//   bool normalize);
MNN_C_API VARP_t mnn_cv_getGaborKernel(
    mnn_cv_size2i_t ksize, double sigma, double theta, double lambd, double gamma, double psi
) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::getGaborKernel(_ksize, sigma, theta, lambd, gamma, psi));
}

MNN_C_API VARP_t mnn_cv_getGaussianKernel(int n, double sigma) {
  return new MNN::Express::VARP(MNN::CV::getGaussianKernel(n, sigma));
}

MNN_C_API VARP_t mnn_cv_getStructuringElement(int shape, mnn_cv_size2i_t ksize) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::getStructuringElement(shape, _ksize));
}

MNN_C_API VARP_t
mnn_cv_Laplacian(VARP_t src, int ddepth, int ksize, double scale, double delta, int borderType) {
  return new MNN::Express::VARP(MNN::CV::Laplacian(*src, ddepth, ksize, scale, delta, borderType));
}

MNN_C_API VARP_t mnn_cv_pyrDown(VARP_t src, mnn_cv_size2i_t dstsize, int borderType) {
  MNN::CV::Size _dstsize = {dstsize.width, dstsize.height};
  return new MNN::Express::VARP(MNN::CV::pyrDown(*src, _dstsize, borderType));
}

MNN_C_API VARP_t mnn_cv_pyrUp(VARP_t src, mnn_cv_size2i_t dstsize, int borderType) {
  MNN::CV::Size _dstsize = {dstsize.width, dstsize.height};
  return new MNN::Express::VARP(MNN::CV::pyrUp(*src, _dstsize, borderType));
}

MNN_C_API VARP_t
mnn_cv_Scharr(VARP_t src, int ddepth, int dx, int dy, double scale, double delta, int borderType) {
  return new MNN::Express::VARP(MNN::CV::Scharr(*src, ddepth, dx, dy, scale, delta, borderType));
}

MNN_C_API VARP_t mnn_cv_sepFilter2D(
    VARP_t src, int ddepth, VARP_t kernelX, VARP_t kernelY, double delta, int borderType
) {
  return new MNN::Express::VARP(
      MNN::CV::sepFilter2D(*src, ddepth, *kernelX, *kernelY, delta, borderType)
  );
}

MNN_C_API VARP_t mnn_cv_Sobel(
    VARP_t src, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType
) {
  return new MNN::Express::VARP(
      MNN::CV::Sobel(*src, ddepth, dx, dy, ksize, scale, delta, borderType)
  );
}

// MNN_C_API std::pair<VARP_t, VARP_t> spatialGradient(VARP_t src, int ksize,
//  int borderType);
MNN_C_API VARP_t
mnn_cv_sqrBoxFilter(VARP_t src, int ddepth, mnn_cv_size2i_t ksize, bool normalize, int borderType) {
  MNN::CV::Size _ksize = {ksize.width, ksize.height};
  return new MNN::Express::VARP(MNN::CV::sqrBoxFilter(*src, ddepth, _ksize, normalize, borderType));
}

// draw.hpp
MNN_C_API void mnn_cv_arrowedLine(
    VARP_t          img,
    mnn_cv_point_t  pt1,
    mnn_cv_point_t  pt2,
    mnn_cv_scalar_t color,
    int             thickness,
    int             line_type,
    int             shift,
    double          tipLength
) {
  MNN::CV::Point  _pt1    = {pt1.x, pt1.y};
  MNN::CV::Point  _pt2    = {pt2.x, pt2.y};
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::arrowedLine(*img, _pt1, _pt2, _scalar, thickness, line_type, shift, tipLength);
}

MNN_C_API void mnn_cv_circle(
    VARP_t          img,
    mnn_cv_point_t  center,
    int             radius,
    mnn_cv_scalar_t color,
    int             thickness,
    int             line_type,
    int             shift
) {
  MNN::CV::Point  _center = {center.x, center.y};
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::circle(*img, _center, radius, _scalar, thickness, line_type, shift);
}

MNN_C_API void mnn_cv_ellipse(
    VARP_t          img,
    mnn_cv_point_t  center,
    mnn_cv_size2i_t axes,
    double          angle,
    double          start_angle,
    double          end_angle,
    mnn_cv_scalar_t color,
    int             thickness,
    int             line_type,
    int             shift
) {
  MNN::CV::Point  _center = {center.x, center.y};
  MNN::CV::Size   _axes   = {axes.width, axes.height};
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::ellipse(
      *img, _center, _axes, angle, start_angle, end_angle, _scalar, thickness, line_type, shift
  );
}

MNN_C_API void mnn_cv_line(
    VARP_t          img,
    mnn_cv_point_t  pt1,
    mnn_cv_point_t  pt2,
    mnn_cv_scalar_t color,
    int             thickness,
    int             lineType,
    int             shift
) {
  MNN::CV::Point  _pt1    = {pt1.x, pt1.y};
  MNN::CV::Point  _pt2    = {pt2.x, pt2.y};
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::line(*img, _pt1, _pt2, _scalar, thickness, lineType, shift);
}

MNN_C_API void mnn_cv_rectangle(
    VARP_t          img,
    mnn_cv_point_t  pt1,
    mnn_cv_point_t  pt2,
    mnn_cv_scalar_t color,
    int             thickness,
    int             lineType,
    int             shift
) {
  MNN::CV::Point  _pt1    = {pt1.x, pt1.y};
  MNN::CV::Point  _pt2    = {pt2.x, pt2.y};
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::rectangle(*img, _pt1, _pt2, _scalar, thickness, lineType, shift);
}

MNN_C_API void mnn_cv_drawContours(
    VARP_t           img,
    mnn_cv_point_t **contours,
    size_t          *contours_inner_length,
    size_t           contours_length,
    int              contourIdx,
    mnn_cv_scalar_t  color,
    int              thickness,
    int              lineType
) {
  std::vector<std::vector<MNN::CV::Point>> _contours(contours_length);
  for (size_t i = 0; i < contours_length; i++) {
    _contours[i].reserve(contours_inner_length[i]);
    for (size_t j = 0; j < contours_inner_length[i]; j++) {
      _contours[i][j] = {contours[i][j].x, contours[i][j].y};
    }
  }
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::drawContours(*img, _contours, contourIdx, _scalar, thickness, lineType);
}

MNN_C_API void mnn_cv_fillPoly(
    VARP_t           img,
    mnn_cv_point_t **pts,
    size_t          *pts_inner_length,
    size_t           pts_length,
    mnn_cv_scalar_t  color,
    int              line_type,
    int              shift,
    mnn_cv_point_t   offset
) {
  std::vector<std::vector<MNN::CV::Point>> _pts(pts_length);
  for (size_t i = 0; i < pts_length; i++) {
    _pts[i].reserve(pts_inner_length[i]);
    for (size_t j = 0; j < pts_inner_length[i]; j++) { _pts[i][j] = {pts[i][j].x, pts[i][j].y}; }
  }
  MNN::CV::Scalar _scalar = {color.val[0], color.val[1], color.val[2], color.val[3]};
  MNN::CV::fillPoly(*img, _pts, _scalar, line_type, shift, {offset.x, offset.y});
}

// color.hpp
MNN_C_API VARP_t mnn_cv_cvtColor(VARP_t src, int code, int dstCn) {
  return new MNN::Express::VARP(MNN::CV::cvtColor(*src, code, dstCn));
}

MNN_C_API VARP_t mnn_cv_cvtColorTwoPlane(VARP_t src1, VARP_t src2, int code) {
  return new MNN::Express::VARP(MNN::CV::cvtColorTwoPlane(*src1, *src2, code));
}

// MNN_C_API VARP_t mnn_cv_demosaicing(VARP_t src, int code, int dstCn) {
//   return new MNN::Express::VARP(MNN::CV::demosaicing(*src, code, dstCn));
// }
