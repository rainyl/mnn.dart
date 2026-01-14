//
// Created by rainy on 2026/1/1.
//

#ifndef MNN_C_API_CV_H
#define MNN_C_API_CV_H
#include "mnn_c/expr.h"
#include "mnn_c/image_process.h"
#include "mnn_c/stdvec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mnn_cv_size2i_t {
  int width;
  int height;
} mnn_cv_size2i_t;

typedef struct mnn_cv_scalar_t {
  double val[4];
} mnn_cv_scalar_t;

typedef struct mnn_cv_rotated_rect_t {
  float center_x;
  float center_y;
  float width;
  float height;
  float angle;
} mnn_cv_rotated_rect_t;

enum ImreadModes {
  IMREAD_GRAYSCALE = 0, // uint8_t gray
  IMREAD_COLOR     = 1, // uint8_t bgr
  IMREAD_COLOR_BGR = 1, // uint8_t bgr
  IMREAD_ANYDEPTH  = 2, // float bgr
  IMREAD_COLOR_RGB = 256,
};

enum ImwriteFlags {
  IMWRITE_JPEG_QUALITY = 1, // jpg, default is 95
};

// types.h
MNN_C_API void   mnn_cv_static_getVARPSize(VARP_t var, int *height, int *width, int *channel);
MNN_C_API int    mnn_cv_getVARPHeight(VARP_t var);
MNN_C_API int    mnn_cv_getVARPWidth(VARP_t var);
MNN_C_API int    mnn_cv_getVARPChannel(VARP_t var);
MNN_C_API int    mnn_cv_getVARPByte(VARP_t var);
MNN_C_API VARP_t mnn_cv_buildImgVARP(uint8_t *img, int height, int width, int channel, int flags);
MNN_C_API VARP_t mnn_cv_buildImgVarpYuvNV21(uint8_t *img, int height, int width, int flags);

// core
MNN_C_API bool mnn_cv_solve(VARP_t src1, VARP_t src2, int flags, VARP_t *out);

// calib3d
MNN_C_API VARP_t mnn_cv_Rodrigues(VARP_t src);
MNN_C_API void   mnn_cv_solvePnP(
      VARP_t  objectPoints,
      VARP_t  imagePoints,
      VARP_t  cameraMatrix,
      VARP_t  distCoeffs,
      bool    useExtrinsicGuess,
      VARP_t *out1,
      VARP_t *out2
  );

// imgcodecs.hpp
MNN_C_API bool   mnn_cv_haveImageReader(char *filename);
MNN_C_API bool   mnn_cv_haveImageReaderFromMemory(uint8_t *buf, size_t length);
MNN_C_API bool   mnn_cv_haveImageWriter(char *filename);
MNN_C_API VARP_t mnn_cv_imdecode(uint8_t *buf, size_t length, int flags);
MNN_C_API bool
mnn_cv_imencode(char *ext, VARP_t img, int *params, size_t params_length, VecU8 *out);
MNN_C_API VARP_t mnn_cv_imread(char *filename, int flags);
MNN_C_API bool   mnn_cv_imwrite(char *filename, VARP_t img, int *params, size_t params_length);

// structural.hpp
MNN_C_API VecVARP_t mnn_cv_findContours(VARP_t image, int mode, int method, mnn_cv_point_t offset);
MNN_C_API double    mnn_cv_contourArea(VARP_t _contour, bool oriented);
MNN_C_API VecI32    mnn_cv_convexHull(VARP_t _points, bool clockwise, bool returnPoints);
MNN_C_API mnn_cv_rotated_rect_t *mnn_cv_minAreaRect(VARP_t _points);
MNN_C_API mnn_cv_rect_t         *mnn_cv_boundingRect(VARP_t points);
// int connectedComponentsWithStats(VARP image, VARP& labels, VARP& statsv, VARP& centroids, int
// connectivity = 8);
MNN_C_API VARP_t mnn_cv_boxPoints(mnn_cv_rotated_rect_t box);

// miscellaneous.hpp
MNN_C_API VARP_t mnn_cv_adaptiveThreshold(
    VARP_t src, double max_value, int adaptiveMethod, int thresholdType, int blockSize, double C
);
MNN_C_API VARP_t mnn_cv_blendLinear(VARP_t src1, VARP_t src2, VARP_t weight1, VARP_t weight2);
// TODO: not implemented by MNN::CV
// MNN_C_API void mnn_cv_distanceTransform(
//     VARP_t src, VARP_t *dst, VARP_t *labels, int distanceType, int maskSize, int labelType
// );
// MNN_C_API int mnn_cv_floodFill(VARP_t image, int seedPoint_x, int seedPoint_y, float newVal);
// MNN_C_API VARP_t mnn_cv_integral(VARP_t src, int sdepth);
MNN_C_API VARP_t mnn_cv_threshold(VARP_t src, double thresh, double maxval, int type);

// histograms.hpp
MNN_C_API VARP_t mnn_cv_calcHist(
    VecVARP_t images, VecI32 channels, VARP_t mask, VecI32 hist_size, VecF32 ranges, bool accumulate
);

// geometric.hpp

// std::pair<VARP, VARP> convertMaps(VARP map1, VARP map2, int dstmap1type,
//                                          bool interpolation = false);
MNN_C_API mnn_cv_matrix_t
mnn_cv_getAffineTransform(const mnn_cv_point_t src[], const mnn_cv_point_t dst[]);
MNN_C_API mnn_cv_matrix_t
mnn_cv_getPerspectiveTransform(const mnn_cv_point_t src[], const mnn_cv_point_t dst[]);
MNN_C_API VARP_t
mnn_cv_getRectSubPix(VARP_t image, mnn_cv_size2i_t patchSize, mnn_cv_point_t center);
MNN_C_API mnn_cv_matrix_t
mnn_cv_getRotationMatrix2D(mnn_cv_point_t center, double angle, double scale);
MNN_C_API mnn_cv_matrix_t mnn_cv_invertAffineTransform(mnn_cv_matrix_t M);
MNN_C_API VARP_t          mnn_cv_remap(
             VARP_t src, VARP_t map1, VARP_t map2, int interpolation, int borderMode, int borderValue
         );
MNN_C_API VARP_t mnn_cv_resize(
    VARP_t          src,
    mnn_cv_size2i_t dsize,
    double          fx,
    double          fy,
    int             interpolation,
    int             code,
    VecF32          mean,
    VecF32          norm
);
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
);
MNN_C_API VARP_t mnn_cv_warpPerspective(
    VARP_t src, mnn_cv_matrix_t M, mnn_cv_size2i_t dsize, int flags, int borderMode, int borderValue
);
MNN_C_API VARP_t mnn_cv_undistortPoints(VARP_t src, VARP_t cameraMatrix, VARP_t distCoeffs);

// filter.hpp
MNN_C_API VARP_t
mnn_cv_bilateralFilter(VARP_t src, int d, double sigmaColor, double sigmaSpace, int borderType);
MNN_C_API VARP_t mnn_cv_blur(VARP_t src, mnn_cv_size2i_t ksize, int borderType);
MNN_C_API VARP_t
mnn_cv_boxFilter(VARP_t src, int ddepth, mnn_cv_size2i_t ksize, bool normalize, int borderType);
MNN_C_API VARP_t mnn_cv_dilate(VARP_t src, VARP_t kernel, int iterations, int borderType);
MNN_C_API VARP_t mnn_cv_erode(VARP_t src, VARP_t kernel, int iterations, int borderType);
MNN_C_API VARP_t
mnn_cv_filter2D(VARP_t src, int ddepth, VARP_t kernel, double delta, int borderType);
MNN_C_API VARP_t mnn_cv_GaussianBlur(
    VARP_t src, mnn_cv_size2i_t ksize, double sigmaX, double sigmaY, int borderType
);
// MNN_C_API std::pair<VARP_t, VARP_t> getDerivKernels(int dx, int dy, int ksize,
//   bool normalize);
MNN_C_API VARP_t mnn_cv_getGaborKernel(
    mnn_cv_size2i_t ksize, double sigma, double theta, double lambd, double gamma, double psi
);
MNN_C_API VARP_t mnn_cv_getGaussianKernel(int n, double sigma);
MNN_C_API VARP_t mnn_cv_getStructuringElement(int shape, mnn_cv_size2i_t ksize);
MNN_C_API VARP_t
mnn_cv_Laplacian(VARP_t src, int ddepth, int ksize, double scale, double delta, int borderType);
MNN_C_API VARP_t mnn_cv_pyrDown(VARP_t src, mnn_cv_size2i_t dstsize, int borderType);
MNN_C_API VARP_t mnn_cv_pyrUp(VARP_t src, mnn_cv_size2i_t dstsize, int borderType);
MNN_C_API VARP_t
mnn_cv_Scharr(VARP_t src, int ddepth, int dx, int dy, double scale, double delta, int borderType);
MNN_C_API VARP_t mnn_cv_sepFilter2D(
    VARP_t src, int ddepth, VARP_t kernelX, VARP_t kernelY, double delta, int borderType
);
MNN_C_API VARP_t mnn_cv_Sobel(
    VARP_t src, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType
);
// MNN_C_API std::pair<VARP_t, VARP_t> spatialGradient(VARP_t src, int ksize,
//  int borderType);
MNN_C_API VARP_t
mnn_cv_sqrBoxFilter(VARP_t src, int ddepth, mnn_cv_size2i_t ksize, bool normalize, int borderType);

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
);
MNN_C_API void mnn_cv_circle(
    VARP_t          img,
    mnn_cv_point_t  center,
    int             radius,
    mnn_cv_scalar_t color,
    int             thickness,
    int             line_type,
    int             shift
);
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
);
MNN_C_API void mnn_cv_line(
    VARP_t          img,
    mnn_cv_point_t  pt1,
    mnn_cv_point_t  pt2,
    mnn_cv_scalar_t color,
    int             thickness,
    int             lineType,
    int             shift
);
MNN_C_API void mnn_cv_rectangle(
    VARP_t          img,
    mnn_cv_point_t  pt1,
    mnn_cv_point_t  pt2,
    mnn_cv_scalar_t color,
    int             thickness,
    int             lineType,
    int             shift
);
MNN_C_API void mnn_cv_drawContours(
    VARP_t           img,
    mnn_cv_point_t **contours,
    size_t          *contours_inner_length,
    size_t           contours_length,
    int              contourIdx,
    mnn_cv_scalar_t  color,
    int              thickness,
    int              lineType
);
MNN_C_API void mnn_cv_fillPoly(
    VARP_t           img,
    mnn_cv_point_t **pts,
    size_t          *pts_inner_length,
    size_t           pts_length,
    mnn_cv_scalar_t  color,
    int              line_type,
    int              shift,
    mnn_cv_point_t   offset
);

// color.hpp
MNN_C_API VARP_t mnn_cv_cvtColor(VARP_t src, int code, int dstCn);
MNN_C_API VARP_t mnn_cv_cvtColorTwoPlane(VARP_t src1, VARP_t src2, int code);
// MNN_C_API VARP_t mnn_cv_demosaicing(VARP_t src, int code, int dstCn);
#ifdef __cplusplus
}
#endif

#endif // MNN_C_API_CV_H
