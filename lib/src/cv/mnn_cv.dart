/// Copyright (c) 2026, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../core/base.dart';
import '../core/vec.dart';
import '../expr/expr.dart';
import '../g/mnn.g.dart' as c;
import 'enums.dart';
import 'matrix.dart';
import 'types.dart';

(int height, int width, int channels) getVARPSize(VARP varp) {
  final pHeight = malloc<ffi.Int>();
  final pWidth = malloc<ffi.Int>();
  final pChannels = malloc<ffi.Int>();
  c.mnn_cv_static_getVARPSize(varp.ptr, pHeight, pWidth, pChannels);
  final rval = (pHeight.value, pWidth.value, pChannels.value);
  malloc.free(pHeight);
  malloc.free(pWidth);
  malloc.free(pChannels);
  return rval;
}

int getVARPHeight(VARP varp) => c.mnn_cv_getVARPHeight(varp.ptr);
int getVARPWidth(VARP varp) => c.mnn_cv_getVARPWidth(varp.ptr);
int getVARPChannels(VARP varp) => c.mnn_cv_getVARPChannel(varp.ptr);
int getVARPByte(VARP varp) => c.mnn_cv_getVARPByte(varp.ptr);

VARP buildImgVARP(Uint8List img, int height, int width, int channels, {int flags = IMREAD_COLOR}) {
  final pImg = calloc<ffi.Uint8>(img.length)..asTypedList(img.length).setAll(0, img);
  final pOut = c.mnn_cv_buildImgVARP(pImg, height, width, channels, flags);
  final rval = VARP.fromPointer(pOut);
  calloc.free(pImg);
  return rval;
}

// core
(bool success, VARP out) solve(VARP src1, VARP src2, {int flags = DECOMP_LU}) {
  final pOut = calloc<c.VARP_t>();
  final success = c.mnn_cv_solve(src1.ptr, src2.ptr, flags, pOut);
  final rval = (success, VARP.fromPointer(pOut.value));
  calloc.free(pOut);
  return rval;
}

// calib3d
VARP Rodrigues(VARP src) {
  return VARP.fromPointer(c.mnn_cv_Rodrigues(src.ptr));
}

(VARP, VARP) solvePnP(
  VARP objectPoints,
  VARP imagePoints,
  VARP cameraMatrix,
  VARP distCoeffs, {
  bool useExtrinsicGuess = false,
}) {
  final pRvec = calloc<c.VARP_t>();
  final pTvec = calloc<c.VARP_t>();
  c.mnn_cv_solvePnP(
    objectPoints.ptr,
    imagePoints.ptr,
    cameraMatrix.ptr,
    distCoeffs.ptr,
    useExtrinsicGuess,
    pRvec,
    pTvec,
  );
  final rval = (VARP.fromPointer(pRvec.value), VARP.fromPointer(pTvec.value));
  calloc.free(pRvec);
  calloc.free(pTvec);
  return rval;
}

// imgcodecs.hpp
bool haveImageReader(String filename) {
  final cFilename = filename.toNativeUtf8().cast<ffi.Char>();
  try {
    return c.mnn_cv_haveImageReader(cFilename);
  } finally {
    malloc.free(cFilename);
  }
}

bool haveImageReaderFromMemory(Uint8List bytes) {
  final pBytes = calloc<ffi.Uint8>(bytes.length)..asTypedList(bytes.length).setAll(0, bytes);
  try {
    return c.mnn_cv_haveImageReaderFromMemory(pBytes, bytes.length);
  } finally {
    calloc.free(pBytes);
  }
}

bool haveImageWriter(String filename) {
  final cFilename = filename.toNativeUtf8().cast<ffi.Char>();
  try {
    return c.mnn_cv_haveImageWriter(cFilename);
  } finally {
    malloc.free(cFilename);
  }
}

VARP imdecode(Uint8List data, {int flags = IMREAD_COLOR}) {
  final pData = calloc<ffi.Uint8>(data.length)..asTypedList(data.length).setAll(0, data);
  final pOut = c.mnn_cv_imdecode(pData, data.length, flags);
  final rval = VARP.fromPointer(pOut);
  calloc.free(pData);
  return rval;
}

(bool success, VecU8 data) imencode(String ext, VARP src, {List<int> params = const []}) {
  final cExt = ext.toNativeUtf8().cast<ffi.Char>();
  final pOut = calloc<c.VecU8>();
  final pParams = params.isEmpty ? ffi.nullptr : calloc<ffi.Int32>(params.length)
    ..asTypedList(params.length).setAll(0, params);
  try {
    final success = c.mnn_cv_imencode(cExt, src.ptr, pParams.cast(), params.length, pOut);
    final rval = (success, VecU8.fromPointer(pOut.value));
    return rval;
  } finally {
    calloc.free(pOut);
    calloc.free(cExt);
    calloc.free(pParams);
  }
}

VARP imread(String filename, {int flags = IMREAD_COLOR}) {
  final cFilename = filename.toNativeUtf8().cast<ffi.Char>();
  try {
    return VARP.fromPointer(c.mnn_cv_imread(cFilename, flags));
  } finally {
    malloc.free(cFilename);
  }
}

bool imwrite(String filename, VARP img, {List<int> params = const []}) {
  final cFilename = filename.toNativeUtf8().cast<ffi.Char>();
  final pParams = params.isEmpty ? ffi.nullptr : calloc<ffi.Int32>(params.length)
    ..asTypedList(params.length).setAll(0, params);
  try {
    return c.mnn_cv_imwrite(cFilename, img.ptr, pParams.cast(), params.length);
  } finally {
    malloc.free(cFilename);
    calloc.free(pParams);
  }
}

// structural.hpp
/// [method] only support [CHAIN_APPROX_NONE], [CHAIN_APPROX_SIMPLE]
VecVARP findContours(VARP img, int mode, int method, {(double, double) offset = const (0, 0)}) {
  if (method != CHAIN_APPROX_NONE && method != CHAIN_APPROX_SIMPLE) {
    throw ArgumentError.value(method, 'method', 'Only support [CHAIN_APPROX_NONE], [CHAIN_APPROX_SIMPLE]');
  }
  final cOffset = Point.fromTuple(offset);
  final pOut = c.mnn_cv_findContours(img.ptr, mode, method, cOffset.ref);
  final rval = VecVARP.fromPointer(pOut);
  cOffset.dispose();
  return rval;
}

double contourArea(VARP contour, {bool oriented = false}) {
  return c.mnn_cv_contourArea(contour.ptr, oriented);
}

VecI32 convexHull(VARP points, {bool clockwise = false, bool returnPoints = true}) {
  final pOut = c.mnn_cv_convexHull(points.ptr, clockwise, returnPoints);
  final rval = VecI32.fromPointer(pOut);
  return rval;
}

RotatedRect minAreaRect(VARP points) {
  final pOut = c.mnn_cv_minAreaRect(points.ptr);
  final rval = RotatedRect.fromPointer(pOut);
  return rval;
}

Rect boundingRect(VARP points) {
  final pOut = c.mnn_cv_boundingRect(points.ptr);
  final rval = Rect.fromPointer(pOut);
  return rval;
}

// int connectedComponentsWithStats(VARP image, VARP& labels, VARP& statsv, VARP& centroids, int connectivity = 8);

VARP boxPoints(RotatedRect box) {
  final pOut = c.mnn_cv_boxPoints(box.ref);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

// miscellaneous.hpp
VARP adaptiveThreshold(
  VARP src,
  double maxValue,
  int adaptiveMethod,
  int thresholdType,
  int blockSize,
  double C,
) {
  final pOut = c.mnn_cv_adaptiveThreshold(src.ptr, maxValue, adaptiveMethod, thresholdType, blockSize, C);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP blendLinear(VARP src1, VARP src2, VARP weight1, VARP weight2) {
  final pOut = c.mnn_cv_blendLinear(src1.ptr, src2.ptr, weight1.ptr, weight2.ptr);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP threshold(VARP src, double thresh, double maxval, int type) {
  MnnAssert(
    type != THRESH_MASK && type != THRESH_OTSU && type != THRESH_TRIANGLE,
    "Don't support THRESH_MASK/THRESH_OTSU/THRESH_TRIANGLE.",
  );
  final pOut = c.mnn_cv_threshold(src.ptr, thresh, maxval, type);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

// histograms.hpp
// VARP calcHist(VARPS images, const std::vector<int>& channels, VARP mask,
//                          const std::vector<int>& histSize, const std::vector<float>& ranges, bool accumulate = false);
VARP calcHist(
  VecVARP images, {
  List<int> channels = const [],
  VARP? mask,
  List<int> histSize = const [],
  List<double> ranges = const [],
  bool accumulate = false,
}) {
  final cChannels = channels.i32;
  final cHistSize = histSize.i32;
  final cRanges = ranges.f32;
  final pOut = c.mnn_cv_calcHist(
    images.ptr,
    cChannels.ptr,
    mask?.ptr ?? ffi.nullptr,
    cHistSize.ptr,
    cRanges.ptr,
    accumulate,
  );
  final rval = VARP.fromPointer(pOut);
  cChannels.dispose();
  cHistSize.dispose();
  cRanges.dispose();
  return rval;
}

// geometric.hpp
Matrix getAffineTransform(List<Point> src, List<Point> dst) {
  MnnAssert(src.length == 3, 'src must have 3 points');
  MnnAssert(dst.length == 3, 'dst must have 3 points');
  final pSrc = calloc<c.mnn_cv_point_t>(3);
  final pDst = calloc<c.mnn_cv_point_t>(3);
  for (int i = 0; i < 3; i++) {
    pSrc[i] = src[i].ref;
    pDst[i] = dst[i].ref;
  }
  final rval = Matrix.fromPointer(c.mnn_cv_getAffineTransform(pSrc, pDst));
  calloc.free(pSrc);
  calloc.free(pDst);
  return rval;
}

Matrix getPerspectiveTransform(List<Point> src, List<Point> dst) {
  MnnAssert(src.length == 4, 'src must have 4 points');
  MnnAssert(dst.length == 4, 'dst must have 4 points');
  final pSrc = calloc<c.mnn_cv_point_t>(4);
  final pDst = calloc<c.mnn_cv_point_t>(4);
  for (int i = 0; i < 4; i++) {
    pSrc[i] = src[i].ref;
    pDst[i] = dst[i].ref;
  }
  final rval = Matrix.fromPointer(c.mnn_cv_getPerspectiveTransform(pSrc, pDst));
  calloc.free(pSrc);
  calloc.free(pDst);
  return rval;
}

VARP getRectSubPix(VARP image, (int, int) patchSize, (double, double) center) {
  final cPatchSize = Size.fromTuple(patchSize);
  final cCenter = Point.fromTuple(center);
  final rval = VARP.fromPointer(c.mnn_cv_getRectSubPix(image.ptr, cPatchSize.ref, cCenter.ref));
  cPatchSize.dispose();
  cCenter.dispose();
  return rval;
}

Matrix getRotationMatrix2D((double, double) center, double angle, double scale) {
  final cCenter = Point.fromTuple(center);
  final rval = Matrix.fromPointer(c.mnn_cv_getRotationMatrix2D(cCenter.ref, angle, scale));
  cCenter.dispose();
  return rval;
}

Matrix invertAffineTransform(Matrix M) {
  final rval = Matrix.fromPointer(c.mnn_cv_invertAffineTransform(M.ptr));
  return rval;
}

/// src need float, NC4HW4, dims = 4
VARP remap(
  VARP src,
  VARP map1,
  VARP map2,
  int interpolation, {
  int borderMode = BORDER_CONSTANT,
  int borderValue = 0,
}) {
  final pOut = c.mnn_cv_remap(src.ptr, map1.ptr, map2.ptr, interpolation, borderMode, borderValue);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP resize(
  VARP src,
  (int, int) dsize, {
  double fx = 0,
  double fy = 0,
  int interpolation = INTER_LINEAR,
  int code = -1,
  List<double> mean = const [],
  List<double> norm = const [],
}) {
  final cMean = mean.f32;
  final cNorm = norm.f32;
  final cDsize = Size.fromTuple(dsize);
  final pOut = c.mnn_cv_resize(
    src.ptr,
    cDsize.ref,
    fx,
    fy,
    interpolation,
    code,
    cMean.ptr,
    cNorm.ptr,
  );
  final rval = VARP.fromPointer(pOut);
  cMean.dispose();
  cNorm.dispose();
  cDsize.dispose();
  return rval;
}

VARP warpAffine(
  VARP src,
  Matrix M,
  (int, int) dsize, {
  int flags = INTER_LINEAR,
  int borderMode = BORDER_CONSTANT,
  int borderValue = 0,
  int code = -1,
  List<double> mean = const [],
  List<double> norm = const [],
}) {
  MnnAssert(
    borderMode == BORDER_CONSTANT || borderMode == BORDER_REPLICATE || borderMode == BORDER_TRANSPARENT,
    'borderMode must be BORDER_CONSTANT or BORDER_REPLICATE or BORDER_TRANSPARENT',
  );
  final cMean = mean.f32;
  final cNorm = norm.f32;
  final cDsize = Size.fromTuple(dsize);
  final pOut = c.mnn_cv_warpAffine(
    src.ptr,
    M.ptr,
    cDsize.ref,
    flags,
    borderMode,
    borderValue,
    code,
    cMean.ptr,
    cNorm.ptr,
  );
  final rval = VARP.fromPointer(pOut);
  cMean.dispose();
  cNorm.dispose();
  cDsize.dispose();
  return rval;
}

VARP warpPerspective(
  VARP src,
  Matrix M,
  (int, int) dsize, {
  int flags = INTER_LINEAR,
  int borderMode = BORDER_CONSTANT,
  int borderValue = 0,
}) {
  final cDsize = Size.fromTuple(dsize);
  final pOut = c.mnn_cv_warpPerspective(
    src.ptr,
    M.ptr,
    cDsize.ref,
    flags,
    borderMode,
    borderValue,
  );
  final rval = VARP.fromPointer(pOut);
  cDsize.dispose();
  return rval;
}

VARP undistortPoints(VARP src, VARP cameraMatrix, VARP distCoeffs) {
  final pOut = c.mnn_cv_undistortPoints(src.ptr, cameraMatrix.ptr, distCoeffs.ptr);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

// filter.hpp
VARP bilateralFilter(
  VARP src,
  int d,
  double sigmaColor,
  double sigmaSpace, {
  int borderType = BORDER_REFLECT,
}) {
  final pOut = c.mnn_cv_bilateralFilter(src.ptr, d, sigmaColor, sigmaSpace, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP blur(VARP src, (int, int) ksize, {int borderType = BORDER_REFLECT}) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_blur(src.ptr, cKsize.ref, borderType);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

VARP boxFilter(
  VARP src,
  int ddepth,
  (int, int) ksize, {
  bool normalize = true,
  int borderType = BORDER_REFLECT,
}) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_boxFilter(src.ptr, ddepth, cKsize.ref, normalize, borderType);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

VARP dilate(VARP src, VARP kernel, {int iterations = 1, int borderType = BORDER_REFLECT}) {
  final pOut = c.mnn_cv_dilate(src.ptr, kernel.ptr, iterations, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP erode(VARP src, VARP kernel, {int iterations = 1, int borderType = BORDER_CONSTANT}) {
  final pOut = c.mnn_cv_erode(src.ptr, kernel.ptr, iterations, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP filter2D(VARP src, int ddepth, VARP kernel, {double delta = 0, int borderType = BORDER_REFLECT}) {
  final pOut = c.mnn_cv_filter2D(src.ptr, ddepth, kernel.ptr, delta, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP GaussianBlur(
  VARP src,
  (int, int) ksize,
  double sigmaX, {
  double sigmaY = 0,
  int borderType = BORDER_REFLECT,
}) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_GaussianBlur(src.ptr, cKsize.ref, sigmaX, sigmaY, borderType);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

VARP getGaborKernel(
  (int, int) ksize,
  double sigma,
  double theta,
  double lambd,
  double gamma, {
  double psi = math.pi * 0.5,
}) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_getGaborKernel(cKsize.ref, sigma, theta, lambd, gamma, psi);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

VARP getGaussianKernel(int n, double sigma) {
  final pOut = c.mnn_cv_getGaussianKernel(n, sigma);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP getStructuringElement(int shape, (int, int) ksize) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_getStructuringElement(shape, cKsize.ref);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

VARP Laplacian(
  VARP src,
  int ddepth, {
  int ksize = 1,
  double scale = 1,
  double delta = 0,
  int borderType = BORDER_REFLECT,
}) {
  MnnAssert(ksize == 1 || ksize == 3, "Laplacian ksize > 3 not supported");
  final pOut = c.mnn_cv_Laplacian(src.ptr, ddepth, ksize, scale, delta, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP pyrDown(VARP src, (int, int) dstsize, {int borderType = BORDER_REFLECT}) {
  final cDstsize = Size.fromTuple(dstsize);
  final pOut = c.mnn_cv_pyrDown(src.ptr, cDstsize.ref, borderType);
  final rval = VARP.fromPointer(pOut);
  cDstsize.dispose();
  return rval;
}

VARP pyrUp(VARP src, (int, int) dstsize, {int borderType = BORDER_REFLECT}) {
  final cDstsize = Size.fromTuple(dstsize);
  final pOut = c.mnn_cv_pyrUp(src.ptr, cDstsize.ref, borderType);
  final rval = VARP.fromPointer(pOut);
  cDstsize.dispose();
  return rval;
}

VARP Scharr(
  VARP src,
  int ddepth,
  int dx,
  int dy, {
  double scale = 1,
  double delta = 0,
  int borderType = BORDER_REFLECT,
}) {
  final pOut = c.mnn_cv_Scharr(src.ptr, ddepth, dx, dy, scale, delta, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP sepFilter2D(
  VARP src,
  int ddepth,
  VARP kernelX,
  VARP kernelY, {
  double delta = 0,
  int borderType = BORDER_REFLECT,
}) {
  final pOut = c.mnn_cv_sepFilter2D(src.ptr, ddepth, kernelX.ptr, kernelY.ptr, delta, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP Sobel(
  VARP src,
  int ddepth,
  int dx,
  int dy, {
  int ksize = 3,
  double scale = 1,
  double delta = 0,
  int borderType = BORDER_REFLECT,
}) {
  final pOut = c.mnn_cv_Sobel(src.ptr, ddepth, dx, dy, ksize, scale, delta, borderType);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

// std::pair<VARP, VARP> spatialGradient(VARP src, int ksize = 3, int borderType = REFLECT);

VARP sqrBoxFilter(
  VARP src,
  int ddepth,
  (int, int) ksize, {
  bool normalize = true,
  int borderType = BORDER_REFLECT,
}) {
  final cKsize = Size.fromTuple(ksize);
  final pOut = c.mnn_cv_sqrBoxFilter(src.ptr, ddepth, cKsize.ref, normalize, borderType);
  final rval = VARP.fromPointer(pOut);
  cKsize.dispose();
  return rval;
}

// draw.hpp

// color.hpp
VARP cvtColor(VARP src, int code, {int dstCn = 0}) {
  final pOut = c.mnn_cv_cvtColor(src.ptr, code, dstCn);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP cvtColorTwoPlane(VARP src1, VARP src2, int code) {
  final pOut = c.mnn_cv_cvtColorTwoPlane(src1.ptr, src2.ptr, code);
  final rval = VARP.fromPointer(pOut);
  return rval;
}

VARP demosaicing(VARP src, int code, {int dstCn = 0}) {
  throw UnimplementedError();
  // final pOut = c.mnn_cv_demosaicing(src.ptr, code, dstCn);
  // final rval = VARP.fromPointer(pOut);
  // return rval;
}
