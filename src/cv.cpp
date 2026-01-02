//
// Created by rainy on 2026/1/1.
//

#include "cv.h"

void mnn_cv_static_getVARPSize(VARP_t var, int *height, int *width, int *channel) {
  MNN::CV::getVARPSize(*var, height, width, channel);
}

int mnn_cv_getVARPHeight(VARP_t var) { return MNN::CV::getVARPHeight(*var); }

int mnn_cv_getVARPWidth(VARP_t var) { return MNN::CV::getVARPWidth(*var); }

int mnn_cv_getVARPChannel(VARP_t var) { return MNN::CV::getVARPChannel(*var); }

int mnn_cv_getVARPByte(VARP_t var) { return MNN::CV::getVARPByte(*var); }

// core
bool mnn_cv_solve(VARP_t src1, VARP_t src2, int flags, VARP_t *out) {
  auto v = MNN::CV::solve(*src1, *src2, flags);
  *out = new MNN::Express::VARP(v.second);
  return v.first;
}

// calib3d
VARP_t mnn_cv_Rodrigues(VARP_t src) { return new MNN::Express::VARP(*src); }
void mnn_cv_solvePnP(
    VARP_t objectPoints,
    VARP_t imagePoints,
    VARP_t cameraMatrix,
    VARP_t distCoeffs,
    bool useExtrinsicGuess,
    VARP_t *out1,
    VARP_t *out2
) {
  auto v =
      MNN::CV::solvePnP(*objectPoints, *imagePoints, *cameraMatrix, *distCoeffs, useExtrinsicGuess);
  *out1 = new MNN::Express::VARP(v.first);
  *out2 = new MNN::Express::VARP(v.second);
}

// imgcodecs.hpp
bool mnn_cv_haveImageReader(char *filename) { return MNN::CV::haveImageReader(filename); }
bool mnn_cv_haveImageWriter(char *filename) { return MNN::CV::haveImageWriter(filename); }
VARP_t mnn_cv_imdecode(uint8_t *buf, size_t length, int flags) {
  return new MNN::Express::VARP(MNN::CV::imdecode(std::vector<uint8_t>(buf, buf + length), flags));
}
bool mnn_cv_imencode(char *ext, VARP_t img, int *params, size_t params_length, VecU8 *out) {
  auto v = MNN::CV::imencode(ext, *img, std::vector<int>(params, params + params_length));
  *out = new std::vector<uint8_t>(v.second);
  return v.first;
}
VARP_t mnn_cv_imread(char *filename, int flags) {
  return new MNN::Express::VARP(MNN::CV::imread(filename, flags));
}
bool mnn_cv_imwrite(char *filename, VARP_t img, int *params, size_t params_length) {
  return MNN::CV::imwrite(filename, *img, std::vector<int>(params, params + params_length));
}
