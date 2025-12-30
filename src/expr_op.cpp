//
// Created by rainy on 2025/9/11.
//

#include "expr_op.h"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"

// Math Op
// BinaryOPs
VARP_t mnn_expr_Add(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Add(*x, *y));
}
VARP_t mnn_expr_Subtract(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Subtract(*x, *y));
}
VARP_t mnn_expr_Multiply(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Multiply(*x, *y));
}
VARP_t mnn_expr_Divide(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Divide(*x, *y));
}
VARP_t mnn_expr_Pow(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Pow(*x, *y));
}
VARP_t mnn_expr_Minimum(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Minimum(*x, *y));
}
VARP_t mnn_expr_Maximum(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Maximum(*x, *y));
}
VARP_t mnn_expr_BiasAdd(VARP_t value, VARP_t bias) {
  return new MNN::Express::VARP(MNN::Express::_BiasAdd(*value, *bias));
}
VARP_t mnn_expr_Greater(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Greater(*x, *y));
}
VARP_t mnn_expr_GreaterEqual(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_GreaterEqual(*x, *y));
}
VARP_t mnn_expr_Less(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Less(*x, *y));
}
VARP_t mnn_expr_FloorDiv(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_FloorDiv(*x, *y));
}
VARP_t mnn_expr_SquaredDifference(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_SquaredDifference(*x, *y));
}
VARP_t mnn_expr_Equal(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Equal(*x, *y));
}
VARP_t mnn_expr_LessEqual(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_LessEqual(*x, *y));
}
VARP_t mnn_expr_FloorMod(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_FloorMod(*x, *y));
}
VARP_t mnn_expr_Atan2(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Atan2(*x, *y));
}
VARP_t mnn_expr_LogicalOr(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_LogicalOr(*x, *y));
}
VARP_t mnn_expr_NotEqual(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_NotEqual(*x, *y));
}
VARP_t mnn_expr_BitwiseAnd(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseAnd(*x, *y));
}
VARP_t mnn_expr_BitwiseOr(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseOr(*x, *y));
}
VARP_t mnn_expr_BitwiseXor(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseXor(*x, *y));
}

// UnaryOPs
VARP_t mnn_expr_Sign(VARP_t a) { return new MNN::Express::VARP(MNN::Express::_Sign(*a)); }
VARP_t mnn_expr_Abs(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Abs(*x)); }
VARP_t mnn_expr_Negative(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Negative(*x)); }
VARP_t mnn_expr_Floor(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Floor(*x)); }
VARP_t mnn_expr_Round(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Round(*x)); }
VARP_t mnn_expr_Ceil(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Ceil(*x)); }
VARP_t mnn_expr_Square(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Square(*x)); }
VARP_t mnn_expr_Sqrt(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Sqrt(*x)); }
VARP_t mnn_expr_Rsqrt(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Rsqrt(*x)); }
VARP_t mnn_expr_Exp(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Exp(*x)); }
VARP_t mnn_expr_Log(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Log(*x)); }
VARP_t mnn_expr_Sin(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Sin(*x)); }
VARP_t mnn_expr_Sinh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Sinh(*x)); }
VARP_t mnn_expr_Cos(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Cos(*x)); }
VARP_t mnn_expr_Cosh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Cosh(*x)); }
VARP_t mnn_expr_Tan(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Tan(*x)); }
VARP_t mnn_expr_Asin(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Asin(*x)); }
VARP_t mnn_expr_Asinh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Asinh(*x)); }
VARP_t mnn_expr_Acos(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Acos(*x)); }
VARP_t mnn_expr_Acosh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Acosh(*x)); }
VARP_t mnn_expr_Atan(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Atan(*x)); }
VARP_t mnn_expr_Atanh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Atanh(*x)); }
VARP_t mnn_expr_Reciprocal(VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Reciprocal(*x));
}
VARP_t mnn_expr_Log1p(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Log1p(*x)); }
VARP_t mnn_expr_Gelu(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Gelu(*x)); }
VARP_t mnn_expr_Tanh(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Tanh(*x)); }
VARP_t mnn_expr_Sigmoid(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Sigmoid(*x)); }
VARP_t mnn_expr_Erf(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Erf(*x)); }
VARP_t mnn_expr_Erfc(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Erfc(*x)); }
VARP_t mnn_expr_Erfinv(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Erfinv(*x)); }
VARP_t mnn_expr_Expm1(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Expm1(*x)); }
VARP_t mnn_expr_Hardswish(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Hardswish(*x)); }
VARP_t mnn_expr_Silu(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Silu(*x)); }

// ReduceOPs
VARP_t mnn_expr_ReduceSum(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceSum(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceMean(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMean(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceVariance(VARP_t input_variable, VecI32 axis, bool keepDims) {
  auto mean = _ReduceMean(*input_variable, *axis, true); // to use broadcast of subtract
  auto variance = _ReduceMean(_Square(_Subtract(*input_variable, mean)), *axis, keepDims);
  return new MNN::Express::VARP(variance);
}
VARP_t mnn_expr_ReduceMax(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMax(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceMin(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMin(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceProd(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceProd(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceAny(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAny(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceAll(VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAll(*input_variable, *axis, keepDims));
}

VARP_t mnn_expr_ReduceSumMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceSumMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceMeanMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMeanMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceMaxMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMaxMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceMinMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMinMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceProdMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceProdMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceAnyMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAnyMutable(*input_variable, *axis, keepDims));
}
VARP_t mnn_expr_ReduceAllMutable(VARP_t input_variable, VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAllMutable(*input_variable, *axis, keepDims));
}

// EltwiseOPs
VARP_t mnn_expr_Prod(VARP_t a, VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Prod(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
VARP_t mnn_expr_Sum(VARP_t a, VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Sum(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
VARP_t mnn_expr_Max(VARP_t a, VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Max(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
VARP_t mnn_expr_Sub(VARP_t a, VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Sub(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
VARP_t mnn_expr_Mod(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Mod(*x, *y));
}

VARP_t mnn_expr_Cast(VARP_t x, halide_type_c_t dtype) {
  const auto _dtype = halide_type_t((halide_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  return new MNN::Express::VARP(MNN::Express::_Cast(*x, _dtype));
}
VARP_t mnn_expr_MatMul(VARP_t a, VARP_t b, bool tranposeA, bool tranposeB) {
  return new MNN::Express::VARP(MNN::Express::_MatMul(*a, *b, tranposeA, tranposeB));
}
VARP_t mnn_expr_Normalize(
    VARP_t x,
    int32_t acrossSpatial,
    int32_t channelShared,
    float eps,
    const float *scale,
    size_t scaleLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Normalize(
          *x, acrossSpatial, channelShared, eps, std::vector<float>(scale, scale + scaleLength)
      )
  );
}
VARP_t mnn_expr_ArgMax(VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ArgMax(*input, axis));
}
VARP_t mnn_expr_ArgMin(VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ArgMin(*input, axis));
}
VARP_t mnn_expr_BatchMatMul(VARP_t x, VARP_t y, bool adj_x, bool adj_y) {
  return new MNN::Express::VARP(MNN::Express::_BatchMatMul(*x, *y, adj_x, adj_y));
}
VARP_t mnn_expr_UnravelIndex(VARP_t indices, VARP_t dims) {
  return new MNN::Express::VARP(MNN::Express::_UnravelIndex(*indices, *dims));
}
VARP_t mnn_expr_ScatterNd(VARP_t indices, VARP_t updates, VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape));
}
VARP_t mnn_expr_ScatterNd_1(VARP_t indices, VARP_t updates, VARP_t shape, VARP_t input) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape, *input));
}
VARP_t mnn_expr_ScatterNd_2(VARP_t indices, VARP_t updates, VARP_t shape, int reduction) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape, reduction));
}
VARP_t
mnn_expr_ScatterNd_3(VARP_t indices, VARP_t updates, VARP_t shape, VARP_t input, int reduction) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterNd(*indices, *updates, *shape, *input, reduction)
  );
}
VARP_t mnn_expr_ScatterElements(VARP_t data, VARP_t indices, VARP_t updates, int reduction) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterElements(*data, *indices, *updates, reduction)
  );
}
VARP_t mnn_expr_ScatterElements_1(
    VARP_t data, VARP_t indices, VARP_t updates, VARP_t axis, int reduction
) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterElements(*data, *indices, *updates, *axis, reduction)
  );
}
VARP_t mnn_expr_OneHot(VARP_t indices, VARP_t depth, VARP_t onValue, VARP_t offValue, int axis) {
  return new MNN::Express::VARP(MNN::Express::_OneHot(*indices, *depth, *onValue, *offValue, axis));
}
VARP_t mnn_expr_BroadcastTo(VARP_t a, VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_BroadcastTo(*a, *shape));
}
VARP_t mnn_expr_LinSpace(VARP_t start, VARP_t stop, VARP_t num) {
  return new MNN::Express::VARP(MNN::Express::_LinSpace(*start, *stop, *num));
}
VARP_t mnn_expr_RandomUniform(
    VARP_t shape, halide_type_c_t dtype, float low, float high, int seed0, int seed1
) {
  const auto _dtype = halide_type_t((halide_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  return new MNN::Express::VARP(
      MNN::Express::_RandomUnifom(*shape, _dtype, low, high, seed0, seed1)
  );
}
VARP_t mnn_expr_CumSum(VARP_t x, int axis, bool exclusive, bool reverse) {
  return new MNN::Express::VARP(MNN::Express::_CumSum(*x, axis, exclusive, reverse));
}
VARP_t mnn_expr_CumProd(VARP_t x, int axis) {
  return new MNN::Express::VARP(MNN::Express::_CumProd(*x, axis));
}
VecVARP_t mnn_expr_Svd(VARP_t x) { return new MNN::Express::VARPS(MNN::Express::_Svd(*x)); }
VARP_t mnn_expr_Histogram(VARP_t x, int bin, int min, int max, int channel) {
  return new MNN::Express::VARP(MNN::Express::_Histogram(*x, bin, min, max, channel));
}

// Neural Network Ops
VARP_t
mnn_expr_Input(const int *shape, size_t shapeLength, int data_format, halide_type_c_t dtype) {
  const auto _dtype = halide_type_t((halide_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  return new MNN::Express::VARP(
      MNN::Express::_Input(
          std::vector<int>(shape, shape + shapeLength),
          static_cast<MNN::Express::Dimensionformat>(data_format),
          _dtype
      )
  );
}
VARP_t mnn_expr_Clone(VARP_t source, bool deepCopy) {
  return new MNN::Express::VARP(MNN::Express::_Clone(*source, deepCopy));
}
VARP_t mnn_expr_Scalar(const void *ptr, halide_type_c_t type) {
  const auto _dtype = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  return new MNN::Express::VARP(MNN::Express::_Scalar(ptr, _dtype));
}

VARP_t mnn_expr_Const(
    void *value, const int *shape, size_t shapeLength, int format, halide_type_c_t type
) {
  auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  auto _shape = std::vector<int>(shape, shape + shapeLength);
  return new MNN::Express::VARP(
      MNN::Express::_Const(value, _shape, static_cast<MNN::Express::Dimensionformat>(format), _type)
  );
}

VARP_t mnn_expr_Conv(
    VARP_t weight,
    VARP_t bias,
    VARP_t x,
    int pad,
    const int *stride,
    size_t strideLength,
    const int *dilate,
    size_t dilateLength,
    int group,
    const int *pads,
    size_t padsLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Conv(
          *weight,
          *bias,
          *x,
          static_cast<MNN::Express::PaddingMode>(pad),
          std::vector<int>(stride, stride + strideLength),
          std::vector<int>(dilate, dilate + dilateLength),
          group,
          std::vector<int>(pads, pads + padsLength)
      )
  );
}
VARP_t mnn_expr_Deconv(
    VARP_t weight,
    VARP_t bias,
    VARP_t x,
    int pad,
    const int *stride,
    size_t strideLength,
    const int *dilate,
    size_t dilateLength,
    int group,
    const int *pads,
    size_t padsLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Deconv(
          *weight,
          *bias,
          *x,
          static_cast<MNN::Express::PaddingMode>(pad),
          std::vector<int>(stride, stride + strideLength),
          std::vector<int>(dilate, dilate + dilateLength),
          group,
          std::vector<int>(pads, pads + padsLength)
      )
  );
}
VARP_t mnn_expr_MaxPool(
    VARP_t x,
    const int *kernel,
    size_t kernelLength,
    const int *stride,
    size_t strideLength,
    int pad,
    const int *pads,
    size_t padsLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_MaxPool(
          *x,
          std::vector<int>(kernel, kernel + kernelLength),
          std::vector<int>(stride, stride + strideLength),
          static_cast<MNN::Express::PaddingMode>(pad),
          std::vector<int>(pads, pads + padsLength)
      )
  );
}
VARP_t mnn_expr_AvePool(
    VARP_t x,
    const int *kernel,
    size_t kernelLength,
    const int *stride,
    size_t strideLength,
    int pad,
    const int *pads,
    size_t padsLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_AvePool(
          *x,
          std::vector<int>(kernel, kernel + kernelLength),
          std::vector<int>(stride, stride + strideLength),
          static_cast<MNN::Express::PaddingMode>(pad),
          std::vector<int>(pads, pads + padsLength)
      )
  );
}
VARP_t mnn_expr_Reshape(VARP_t x, const int *shape, size_t shapeLength, int original_format) {
  return new MNN::Express::VARP(
      MNN::Express::_Reshape(
          *x,
          std::vector<int>(shape, shape + shapeLength),
          static_cast<MNN::Express::Dimensionformat>(original_format)
      )
  );
}
VARP_t mnn_expr_Reshape_1(VARP_t x, VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_Reshape(*x, *shape));
}
VARP_t mnn_expr_Scale(
    VARP_t x, int channels, float *scales, size_t scaleLength, float *bias, size_t biasLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Scale(
          *x,
          channels,
          std::vector<float>(scales, scales + scaleLength),
          std::vector<float>(bias, bias + biasLength)
      )
  );
}

VARP_t mnn_expr_Relu(VARP_t x, float slope) {
  return new MNN::Express::VARP(MNN::Express::_Relu(*x, slope));
}
VARP_t mnn_expr_Relu6(VARP_t x, float minValue, float maxValue) {
  return new MNN::Express::VARP(MNN::Express::_Relu6(*x, minValue, maxValue));
}
VARP_t mnn_expr_PRelu(VARP_t x, float *slopes, size_t slopeLength) {
  return new MNN::Express::VARP(
      MNN::Express::_PRelu(*x, std::vector<float>(slopes, slopes + slopeLength))
  );
}
VARP_t mnn_expr_Softmax(VARP_t logits, int axis) {
  return new MNN::Express::VARP(MNN::Express::_Softmax(*logits, axis));
}
VARP_t mnn_expr_Softplus(VARP_t features) {
  return new MNN::Express::VARP(MNN::Express::_Softplus(*features));
}
VARP_t mnn_expr_Softsign(VARP_t features) {
  return new MNN::Express::VARP(MNN::Express::_Softsign(*features));
}
VecVARP_t mnn_expr_Split(VARP_t value, const int *size_splits, size_t size_splitsLength, int axis) {
  return new MNN::Express::VARPS(
      MNN::Express::_Split(
          *value, std::vector<int>(size_splits, size_splits + size_splitsLength), axis
      )
  );
}
VARP_t mnn_expr_Slice(VARP_t x, VARP_t starts, VARP_t sizes) {
  return new MNN::Express::VARP(MNN::Express::_Slice(*x, *starts, *sizes));
}
VARP_t mnn_expr_StridedSlice(
    VARP_t input,
    VARP_t begin,
    VARP_t end,
    VARP_t strided,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
) {
  return new MNN::Express::VARP(
      MNN::Express::_StridedSlice(
          *input,
          *begin,
          *end,
          *strided,
          beginMask,
          endMask,
          ellipsisMask,
          newAxisMask,
          shrinkAxisMask
      )
  );
}
VARP_t mnn_expr_StridedSliceWrite(
    VARP_t input,
    VARP_t begin,
    VARP_t end,
    VARP_t strided,
    VARP_t write,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
) {
  return new MNN::Express::VARP(
      MNN::Express::_StridedSliceWrite(
          *input,
          *begin,
          *end,
          *strided,
          *write,
          beginMask,
          endMask,
          ellipsisMask,
          newAxisMask,
          shrinkAxisMask
      )
  );
}
VARP_t mnn_expr_Concat(VecVARP_t values, int axis) {
  return new MNN::Express::VARP(MNN::Express::_Concat(*values, axis));
}
VARP_t mnn_expr_Convert(VARP_t input, int format) {
  return new MNN::Express::VARP(
      MNN::Express::_Convert(*input, static_cast<MNN::Express::Dimensionformat>(format))
  );
}
VARP_t mnn_expr_Transpose(VARP_t x, const int *perm, size_t permLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Transpose(*x, std::vector<int>(perm, perm + permLength))
  );
}
VARP_t mnn_expr_Transpose_1(VARP_t x, VARP_t perm) {
  return new MNN::Express::VARP(MNN::Express::_Transpose(*x, *perm));
}
VARP_t mnn_expr_ChannelShuffle(VARP_t x, int group) {
  return new MNN::Express::VARP(MNN::Express::_ChannelShuffle(*x, group));
}
VARP_t mnn_expr_ChangeInputFormat(VARP_t input, int format) {
  return new MNN::Express::VARP(
      MNN::Express::_ChangeInputFormat(*input, static_cast<MNN::Express::Dimensionformat>(format))
  );
}
VARP_t mnn_expr_Reverse(VARP_t x, VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_Reverse(*x, *axis));
}
VARP_t mnn_expr_ReverseSequence(VARP_t x, VARP_t y, int batchDim, int seqDim) {
  return new MNN::Express::VARP(MNN::Express::_ReverseSequence(*x, *y, batchDim, seqDim));
}
VARP_t mnn_expr_Crop(VARP_t images, VARP_t size, int axis, const int *offset, size_t offsetLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Crop(*images, *size, axis, std::vector<int>(offset, offset + offsetLength))
  );
}
VARP_t mnn_expr_Resize(VARP_t images, float xScale, float yScale) {
  return new MNN::Express::VARP(MNN::Express::_Resize(*images, xScale, yScale));
}
VARP_t mnn_expr_Pad(VARP_t x, VARP_t paddings, int mode) {
  return new MNN::Express::VARP(
      MNN::Express::_Pad(*x, *paddings, static_cast<MNN::Express::PadValueMode>(mode))
  );
}
VARP_t mnn_expr_ExpandDims(VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ExpandDims(*input, axis));
}
VARP_t mnn_expr_ExpandDims_1(VARP_t input, VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_ExpandDims(*input, *axis));
}
VARP_t mnn_expr_Shape(VARP_t input, bool nchw) {
  return new MNN::Express::VARP(MNN::Express::_Shape(*input, nchw));
}
VARP_t mnn_expr_Stack(VecVARP_t values, int axis) {
  return new MNN::Express::VARP(MNN::Express::_Stack(*values, axis));
}
// enum InterpolationMethod {BILINEAR, NEAREST};
VARP_t mnn_expr_CropAndResize(
    VARP_t image,
    VARP_t boxes,
    VARP_t box_ind,
    VARP_t crop_size,
    int method,
    float extrapolation_value
) {
  return new MNN::Express::VARP(
      MNN::Express::_CropAndResize(
          *image,
          *boxes,
          *box_ind,
          *crop_size,
          static_cast<MNN::Express::InterpolationMethod>(method),
          extrapolation_value
      )
  );
}
VARP_t mnn_expr_Fill(VARP_t dims, VARP_t value) {
  return new MNN::Express::VARP(MNN::Express::_Fill(*dims, *value));
}
VARP_t mnn_expr_Tile(VARP_t input, VARP_t multiples) {
  return new MNN::Express::VARP(MNN::Express::_Tile(*input, *multiples));
}
VARP_t mnn_expr_Gather(VARP_t params, VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_Gather(*params, *indices));
}
VARP_t mnn_expr_GatherV2(VARP_t params, VARP_t indices, VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_GatherV2(*params, *indices, *axis));
}
VARP_t mnn_expr_Squeeze(VARP_t input, const int *axis, size_t axisLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Squeeze(*input, std::vector<int>(axis, axis + axisLength))
  );
}
VARP_t mnn_expr_Unsqueeze(VARP_t input, const int *axis, size_t axisLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Unsqueeze(*input, std::vector<int>(axis, axis + axisLength))
  );
}
VARP_t mnn_expr_BatchToSpaceND(VARP_t input, VARP_t block_shape, VARP_t crops) {
  return new MNN::Express::VARP(MNN::Express::_BatchToSpaceND(*input, *block_shape, *crops));
}
VARP_t mnn_expr_GatherND(VARP_t params, VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_GatherND(*params, *indices));
}
VARP_t mnn_expr_GatherElements(VARP_t params, VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_GatherElements(*params, *indices));
}
VARP_t mnn_expr_GatherElements_1(VARP_t params, VARP_t indices, VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_GatherElements(*params, *indices, *axis));
}
VARP_t mnn_expr_Selu(VARP_t features, float scale, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Selu(*features, scale, alpha));
}
VARP_t mnn_expr_Size(VARP_t input) { return new MNN::Express::VARP(MNN::Express::_Size(*input)); }
VARP_t mnn_expr_Elu(VARP_t features, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Elu(*features, alpha));
}
VARP_t mnn_expr_Threshold(VARP_t features, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Threshold(*features, alpha));
}
VARP_t mnn_expr_MatrixBandPart(VARP_t input, VARP_t num_lower, VARP_t num_upper) {
  return new MNN::Express::VARP(MNN::Express::_MatrixBandPart(*input, *num_lower, *num_upper));
}
VecVARP_t
mnn_expr_Moments(VARP_t x, const int *axis, size_t axisLength, VARP_t shift, bool keepDims) {
  return new MNN::Express::VARPS(
      MNN::Express::_Moments(*x, std::vector<int>(axis, axis + axisLength), *shift, keepDims)
  );
}
VARP_t mnn_expr_SetDiff1D(VARP_t x, VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_SetDiff1D(*x, *y));
}
VARP_t mnn_expr_SpaceToDepth(VARP_t input, int block_size) {
  return new MNN::Express::VARP(MNN::Express::_SpaceToDepth(*input, block_size));
}
VARP_t mnn_expr_SpaceToBatchND(VARP_t input, VARP_t block_shape, VARP_t paddings) {
  return new MNN::Express::VARP(MNN::Express::_SpaceToBatchND(*input, *block_shape, *paddings));
}
VARP_t mnn_expr_ZerosLike(VARP_t input) {
  return new MNN::Express::VARP(MNN::Express::_ZerosLike(*input));
}
VecVARP_t mnn_expr_Unstack(VARP_t value, int axis) {
  return new MNN::Express::VARPS(MNN::Express::_Unstack(*value, axis));
}
VARP_t mnn_expr_Rank(VARP_t input) { return new MNN::Express::VARP(MNN::Express::_Rank(*input)); }
VARP_t mnn_expr_Range(VARP_t start, VARP_t limit, VARP_t delta) {
  return new MNN::Express::VARP(MNN::Express::_Range(*start, *limit, *delta));
}
VARP_t mnn_expr_DepthToSpace(VARP_t input, int block_size) {
  return new MNN::Express::VARP(MNN::Express::_DepthToSpace(*input, block_size));
}

VARP_t mnn_expr_Permute(VARP_t input, const int *dims, size_t dimsLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Permute(*input, std::vector<int>(dims, dims + dimsLength))
  );
}
VARP_t mnn_expr_Interp(
    VecVARP_t xs,
    float widthScale,
    float heightScale,
    int outputWidth,
    int outputHeight,
    int resizeType,
    bool alignCorners
) {
  return new MNN::Express::VARP(
      MNN::Express::_Interp(
          *xs, widthScale, heightScale, outputWidth, outputHeight, resizeType, alignCorners
      )
  );
}
VARP_t mnn_expr_ZeroGrad(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_ZeroGrad(*x)); }
VARP_t mnn_expr_CosineSimilarity(VARP_t input0, VARP_t input1, VARP_t inputDim) {
  return new MNN::Express::VARP(MNN::Express::_CosineSimilarity(*input0, *input1, *inputDim));
}
// enum GridSamplePaddingMode {GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER,
// GRID_SAMPLE_PADDING_REFLECTION};
VARP_t
mnn_expr_GridSample(VARP_t input, VARP_t grid, int mode, int paddingMode, bool alignCorners) {
  return new MNN::Express::VARP(
      MNN::Express::_GridSample(
          *input,
          *grid,
          static_cast<MNN::Express::InterpolationMethod>(mode),
          static_cast<MNN::Express::GridSamplePaddingMode>(paddingMode),
          alignCorners
      )
  );
}
VARP_t mnn_expr_FloatToInt8(VARP_t x, VARP_t scale, char minValue, char maxValue) {
  return new MNN::Express::VARP(MNN::Express::_FloatToInt8(*x, *scale, minValue, maxValue));
}
VARP_t
mnn_expr_FloatToInt8_1(VARP_t x, VARP_t scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint) {
  return new MNN::Express::VARP(
      MNN::Express::_FloatToInt8(*x, *scale, minValue, maxValue, zeroPoint)
  );
}
VARP_t mnn_expr_Int8ToFloat(VARP_t x, VARP_t scale) {
  return new MNN::Express::VARP(MNN::Express::_Int8ToFloat(*x, *scale));
}
VARP_t mnn_expr_Int8ToFloat_1(VARP_t x, VARP_t scale, int8_t zeroPoint) {
  return new MNN::Express::VARP(MNN::Express::_Int8ToFloat(*x, *scale, zeroPoint));
}

VARP_t mnn_expr_Select(VARP_t select, VARP_t input0, VARP_t input1) {
  return new MNN::Express::VARP(MNN::Express::_Select(*select, *input0, *input1));
}
VecVARP_t mnn_expr_TopKV2(VARP_t input0, VARP_t input1) {
  return new MNN::Express::VARPS(MNN::Express::_TopKV2(*input0, *input1));
}
// MNN_PUBLIC VARP _ImageProcess(VARP input, CV::ImageProcess::Config config, CV::Matrix matrix, int
// oh, int ow, int oc, int dtype, uint8_t padVal = 0); mnn_expr_VARP_t
// mnn_expr_ImageProcess(mnn_expr_VARP_t input, CV::ImageProcess::Config config, CV::Matrix matrix,
// int oh, int ow, int oc, int dtype, uint8_t padVal);
VARP_t mnn_expr_Where(VARP_t x) { return new MNN::Express::VARP(MNN::Express::_Where(*x)); }
VARP_t mnn_expr_Sort(VARP_t x, int axis, bool arg, bool descend) {
  return new MNN::Express::VARP(MNN::Express::_Sort(*x, axis, arg, descend));
}
VARP_t mnn_expr_Raster(
    VecVARP_t vars, const int *regions, size_t regionsLength, const int *shape, size_t shapeLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Raster(
          *vars,
          std::vector<int>(regions, regions + regionsLength),
          std::vector<int>(shape, shape + shapeLength)
      )
  );
}
VARP_t mnn_expr_RasterRaw(
    VecVARP_t vars,
    const int *region,
    size_t regionLength,
    const int *shape,
    size_t shapeLength,
    halide_type_t dataType,
    int format
) {
  const auto _dtype =
      halide_type_t((halide_type_code_t)dataType.code, dataType.bits, dataType.lanes);
  return new MNN::Express::VARP(
      MNN::Express::_RasterRaw(
          *vars,
          std::vector<int>(region, region + regionLength),
          std::vector<int>(shape, shape + shapeLength),
          _dtype,
          static_cast<MNN::Express::Dimensionformat>(format)
      )
  );
}

VARP_t mnn_expr_Nms(
    VARP_t boxes, VARP_t scores, int maxDetections, float iouThreshold, float scoreThreshold
) {
  return new MNN::Express::VARP(
      MNN::Express::_Nms(*boxes, *scores, maxDetections, iouThreshold, scoreThreshold)
  );
}
VARP_t mnn_expr_Im2Col(
    VARP_t x,
    const int *kernelSize,
    size_t kernelSizeLength,
    const int *dilate,
    size_t dilateLength,
    const int *pads,
    size_t padsLength,
    const int *stride,
    size_t strideLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Im2Col(
          *x,
          std::vector<int>(kernelSize, kernelSize + kernelSizeLength),
          std::vector<int>(dilate, dilate + dilateLength),
          std::vector<int>(pads, pads + padsLength),
          std::vector<int>(stride, stride + strideLength)
      )
  );
}
VARP_t mnn_expr_Col2Im(
    VARP_t x,
    VARP_t outputShape,
    const int *kernelSize,
    size_t kernelSizeLength,
    const int *dilate,
    size_t dilateLength,
    const int *pads,
    size_t padsLength,
    const int *stride,
    size_t strideLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Col2Im(
          *x,
          *outputShape,
          std::vector<int>(kernelSize, kernelSize + kernelSizeLength),
          std::vector<int>(dilate, dilate + dilateLength),
          std::vector<int>(pads, pads + padsLength),
          std::vector<int>(stride, stride + strideLength)
      )
  );
}
