//
// Created by rainy on 2026/1/1.
//

#ifndef MNN_C_API_CV_H
#define MNN_C_API_CV_H
#include "expr.h"

#ifdef __cplusplus
#include "MNN/expr/Expr.hpp"
#include "cv/cv.hpp"
extern "C" {
#endif

typedef struct mnn_cv_scalar {
  double val[4];
} mnn_cv_scalar;

typedef struct mnn_cv_rotated_rect {
  float center_x;
  float center_y;
  float width;
  float height;
  float angle;
} mnn_cv_rotated_rect;

// types.h
MNN_C_API void mnn_cv_static_getVARPSize(VARP_t var, int *height, int *width, int *channel);
MNN_C_API int mnn_cv_getVARPHeight(VARP_t var);
MNN_C_API int mnn_cv_getVARPWidth(VARP_t var);
MNN_C_API int mnn_cv_getVARPChannel(VARP_t var);
MNN_C_API int mnn_cv_getVARPByte(VARP_t var);

// core
MNN_C_API bool mnn_cv_solve(VARP_t src1, VARP_t src2, int flags, VARP_t *out);

// calib3d
MNN_C_API VARP_t mnn_cv_Rodrigues(VARP_t src);
MNN_C_API void mnn_cv_solvePnP(
    VARP_t objectPoints,
    VARP_t imagePoints,
    VARP_t cameraMatrix,
    VARP_t distCoeffs,
    bool useExtrinsicGuess,
    VARP_t *out1,
    VARP_t *out2
);

// imgcodecs.hpp
MNN_C_API bool mnn_cv_haveImageReader(char *filename);
MNN_C_API bool mnn_cv_haveImageWriter(char *filename);
MNN_C_API VARP_t mnn_cv_imdecode(uint8_t *buf, size_t length, int flags);
MNN_C_API bool
mnn_cv_imencode(char *ext, VARP_t img, int *params, size_t params_length, VecU8 *out);
MNN_C_API VARP_t mnn_cv_imread(char *filename, int flags);
MNN_C_API bool mnn_cv_imwrite(char *filename, VARP_t img, int *params, size_t params_length);

// structural.hpp
MNN_C_API VecVARP_t
mnn_cv_findContours(VARP_t image, int mode, int method, int offset_x, int offset_y);
MNN_C_API double mnn_cv_contourArea(VARP_t _contour, bool oriented);
MNN_C_API VecI32 mnn_cv_convexHull(VARP_t _points, bool clockwise, bool returnPoints);
MNN_C_API mnn_cv_rotated_rect *mnn_cv_minAreaRect(VARP_t _points);
MNN_C_API void mnn_cv_boundingRect(VARP_t points, int *x, int *y);
// int connectedComponentsWithStats(VARP image, VARP& labels, VARP& statsv, VARP& centroids, int
// connectivity = 8);
MNN_C_API VARP_t mnn_cv_boxPoints(mnn_cv_rotated_rect box);

// miscellaneous.hpp
MNN_C_API VARP_t mnn_cv_adaptiveThreshold(
    VARP_t src, double max_value, int adaptiveMethod, int thresholdType, int blockSize, double C
);
MNN_C_API VARP_t mnn_cv_blendLinear(VARP_t src1, VARP_t src2, VARP_t weight1, VARP_t weight2);
MNN_C_API void mnn_cv_distanceTransform(
    VARP_t src, VARP_t *dst, VARP_t *labels, int distanceType, int maskSize, int labelType
);
MNN_C_API int mnn_cv_floodFill(VARP_t image, int seedPoint_x, int seedPoint_y, float newVal);
MNN_C_API VARP_t mnn_cv_integral(VARP_t src, int sdepth);
MNN_C_API VARP_t mnn_cv_threshold(VARP_t src, double thresh, double maxval, int type);

// histograms.hpp
MNN_C_API VARP_t mnn_cv_calcHist(
    VecVARP_t images,
    int *channels,
    size_t channels_length,
    VARP_t mask,
    int *hist_size,
    size_t hist_size_length,
    float *ranges,
    size_t ranges_length,
    bool accumulate
);

// geometric.hpp

// filter.hpp

// draw.hpp

// color.hpp
MNN_C_API VARP_t mnn_cv_cvtColor(VARP_t src, int code, int dstCn);
MNN_C_API VARP_t mnn_cv_cvtColorTwoPlane(VARP_t src1, VARP_t src2, int code);
MNN_C_API VARP_t mnn_cv_demosaicing(VARP_t src, int code, int dstCn);
#ifdef __cplusplus
}
#endif

#endif // MNN_C_API_CV_H
