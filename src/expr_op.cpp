//
// Created by rainy on 2025/9/11.
//

#include "expr_op.h"
#include "MNN/expr/MathOp.hpp"
#include "MNN/expr/NeuralNetWorkOp.hpp"

// Math Op
// BinaryOPs
mnn_expr_VARP_t mnn_expr_Add(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Add(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Subtract(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Subtract(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Multiply(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Multiply(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Divide(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Divide(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Pow(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Pow(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Minimum(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Minimum(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Maximum(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Maximum(*x, *y));
}
mnn_expr_VARP_t mnn_expr_BiasAdd(mnn_expr_VARP_t value, mnn_expr_VARP_t bias) {
  return new MNN::Express::VARP(MNN::Express::_BiasAdd(*value, *bias));
}
mnn_expr_VARP_t mnn_expr_Greater(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Greater(*x, *y));
}
mnn_expr_VARP_t mnn_expr_GreaterEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_GreaterEqual(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Less(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Less(*x, *y));
}
mnn_expr_VARP_t mnn_expr_FloorDiv(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_FloorDiv(*x, *y));
}
mnn_expr_VARP_t mnn_expr_SquaredDifference(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_SquaredDifference(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Equal(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Equal(*x, *y));
}
mnn_expr_VARP_t mnn_expr_LessEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_LessEqual(*x, *y));
}
mnn_expr_VARP_t mnn_expr_FloorMod(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_FloorMod(*x, *y));
}
mnn_expr_VARP_t mnn_expr_Atan2(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Atan2(*x, *y));
}
mnn_expr_VARP_t mnn_expr_LogicalOr(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_LogicalOr(*x, *y));
}
mnn_expr_VARP_t mnn_expr_NotEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_NotEqual(*x, *y));
}
mnn_expr_VARP_t mnn_expr_BitwiseAnd(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseAnd(*x, *y));
}
mnn_expr_VARP_t mnn_expr_BitwiseOr(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseOr(*x, *y));
}
mnn_expr_VARP_t mnn_expr_BitwiseXor(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_BitwiseXor(*x, *y));
}

// UnaryOPs
mnn_expr_VARP_t mnn_expr_Sign(mnn_expr_VARP_t a) {
  return new MNN::Express::VARP(MNN::Express::_Sign(*a));
}
mnn_expr_VARP_t mnn_expr_Abs(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Abs(*x));
}
mnn_expr_VARP_t mnn_expr_Negative(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Negative(*x));
}
mnn_expr_VARP_t mnn_expr_Floor(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Floor(*x));
}
mnn_expr_VARP_t mnn_expr_Round(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Round(*x));
}
mnn_expr_VARP_t mnn_expr_Ceil(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Ceil(*x));
}
mnn_expr_VARP_t mnn_expr_Square(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Square(*x));
}
mnn_expr_VARP_t mnn_expr_Sqrt(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Sqrt(*x));
}
mnn_expr_VARP_t mnn_expr_Rsqrt(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Rsqrt(*x));
}
mnn_expr_VARP_t mnn_expr_Exp(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Exp(*x));
}
mnn_expr_VARP_t mnn_expr_Log(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Log(*x));
}
mnn_expr_VARP_t mnn_expr_Sin(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Sin(*x));
}
mnn_expr_VARP_t mnn_expr_Sinh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Sinh(*x));
}
mnn_expr_VARP_t mnn_expr_Cos(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Cos(*x));
}
mnn_expr_VARP_t mnn_expr_Cosh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Cosh(*x));
}
mnn_expr_VARP_t mnn_expr_Tan(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Tan(*x));
}
mnn_expr_VARP_t mnn_expr_Asin(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Asin(*x));
}
mnn_expr_VARP_t mnn_expr_Asinh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Asinh(*x));
}
mnn_expr_VARP_t mnn_expr_Acos(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Acos(*x));
}
mnn_expr_VARP_t mnn_expr_Acosh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Acosh(*x));
}
mnn_expr_VARP_t mnn_expr_Atan(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Atan(*x));
}
mnn_expr_VARP_t mnn_expr_Atanh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Atanh(*x));
}
mnn_expr_VARP_t mnn_expr_Reciprocal(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Reciprocal(*x));
}
mnn_expr_VARP_t mnn_expr_Log1p(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Log1p(*x));
}
mnn_expr_VARP_t mnn_expr_Gelu(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Gelu(*x));
}
mnn_expr_VARP_t mnn_expr_Tanh(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Tanh(*x));
}
mnn_expr_VARP_t mnn_expr_Sigmoid(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Sigmoid(*x));
}
mnn_expr_VARP_t mnn_expr_Erf(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Erf(*x));
}
mnn_expr_VARP_t mnn_expr_Erfc(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Erfc(*x));
}
mnn_expr_VARP_t mnn_expr_Erfinv(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Erfinv(*x));
}
mnn_expr_VARP_t mnn_expr_Expm1(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Expm1(*x));
}
mnn_expr_VARP_t mnn_expr_Hardswish(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Hardswish(*x));
}
mnn_expr_VARP_t mnn_expr_Silu(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Silu(*x));
}

// ReduceOPs
mnn_expr_VARP_t mnn_expr_ReduceSum(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceSum(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceMean(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMean(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceMax(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMax(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceMin(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMin(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceProd(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceProd(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceAny(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAny(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t mnn_expr_ReduceAll(mnn_expr_VARP_t input_variable, VecI32 axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAll(*input_variable, *axis, keepDims));
}

mnn_expr_VARP_t
mnn_expr_ReduceSumMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceSumMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceMeanMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMeanMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceMaxMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMaxMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceMinMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceMinMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceProdMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceProdMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceAnyMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAnyMutable(*input_variable, *axis, keepDims));
}
mnn_expr_VARP_t
mnn_expr_ReduceAllMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims) {
  return new MNN::Express::VARP(MNN::Express::_ReduceAllMutable(*input_variable, *axis, keepDims));
}

// EltwiseOPs
mnn_expr_VARP_t
mnn_expr_Prod(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Prod(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
mnn_expr_VARP_t mnn_expr_Sum(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Sum(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
mnn_expr_VARP_t mnn_expr_Max(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Max(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
mnn_expr_VARP_t mnn_expr_Sub(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize) {
  return new MNN::Express::VARP(
      MNN::Express::_Sub(*a, *b, std::vector<float>(coeff, coeff + coeffSize))
  );
}
mnn_expr_VARP_t mnn_expr_Mod(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_Mod(*x, *y));
}

mnn_expr_VARP_t mnn_expr_Cast(mnn_expr_VARP_t x, halide_type_c_t dtype) {
  const auto _dtype = halide_type_t((halide_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  return new MNN::Express::VARP(MNN::Express::_Cast(*x, _dtype));
}
mnn_expr_VARP_t
mnn_expr_MatMul(mnn_expr_VARP_t a, mnn_expr_VARP_t b, bool tranposeA, bool tranposeB) {
  return new MNN::Express::VARP(MNN::Express::_MatMul(*a, *b, tranposeA, tranposeB));
}
mnn_expr_VARP_t mnn_expr_Normalize(
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t mnn_expr_ArgMax(mnn_expr_VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ArgMax(*input, axis));
}
mnn_expr_VARP_t mnn_expr_ArgMin(mnn_expr_VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ArgMin(*input, axis));
}
mnn_expr_VARP_t mnn_expr_BatchMatMul(mnn_expr_VARP_t x, mnn_expr_VARP_t y, bool adj_x, bool adj_y) {
  return new MNN::Express::VARP(MNN::Express::_BatchMatMul(*x, *y, adj_x, adj_y));
}
mnn_expr_VARP_t mnn_expr_UnravelIndex(mnn_expr_VARP_t indices, mnn_expr_VARP_t dims) {
  return new MNN::Express::VARP(MNN::Express::_UnravelIndex(*indices, *dims));
}
mnn_expr_VARP_t
mnn_expr_ScatterNd(mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape));
}
mnn_expr_VARP_t mnn_expr_ScatterNd_1(
    mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape, mnn_expr_VARP_t input
) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape, *input));
}
mnn_expr_VARP_t mnn_expr_ScatterNd_2(
    mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape, int reduction
) {
  return new MNN::Express::VARP(MNN::Express::_ScatterNd(*indices, *updates, *shape, reduction));
}
mnn_expr_VARP_t mnn_expr_ScatterNd_3(
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t updates,
    mnn_expr_VARP_t shape,
    mnn_expr_VARP_t input,
    int reduction
) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterNd(*indices, *updates, *shape, *input, reduction)
  );
}
mnn_expr_VARP_t mnn_expr_ScatterElements(
    mnn_expr_VARP_t data, mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, int reduction
) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterElements(*data, *indices, *updates, reduction)
  );
}
mnn_expr_VARP_t mnn_expr_ScatterElements_1(
    mnn_expr_VARP_t data,
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t updates,
    mnn_expr_VARP_t axis,
    int reduction
) {
  return new MNN::Express::VARP(
      MNN::Express::_ScatterElements(*data, *indices, *updates, *axis, reduction)
  );
}
mnn_expr_VARP_t mnn_expr_OneHot(
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t depth,
    mnn_expr_VARP_t onValue,
    mnn_expr_VARP_t offValue,
    int axis
) {
  return new MNN::Express::VARP(MNN::Express::_OneHot(*indices, *depth, *onValue, *offValue, axis));
}
mnn_expr_VARP_t mnn_expr_BroadcastTo(mnn_expr_VARP_t a, mnn_expr_VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_BroadcastTo(*a, *shape));
}
mnn_expr_VARP_t
mnn_expr_LinSpace(mnn_expr_VARP_t start, mnn_expr_VARP_t stop, mnn_expr_VARP_t num) {
  return new MNN::Express::VARP(MNN::Express::_LinSpace(*start, *stop, *num));
}
mnn_expr_VARP_t mnn_expr_RandomUniform(
    mnn_expr_VARP_t shape, halide_type_c_t dtype, float low, float high, int seed0, int seed1
) {
  const auto _dtype = halide_type_t((halide_type_code_t)dtype.code, dtype.bits, dtype.lanes);
  return new MNN::Express::VARP(
      MNN::Express::_RandomUnifom(*shape, _dtype, low, high, seed0, seed1)
  );
}
mnn_expr_VARP_t mnn_expr_CumSum(mnn_expr_VARP_t x, int axis, bool exclusive, bool reverse) {
  return new MNN::Express::VARP(MNN::Express::_CumSum(*x, axis, exclusive, reverse));
}
mnn_expr_VARP_t mnn_expr_CumProd(mnn_expr_VARP_t x, int axis) {
  return new MNN::Express::VARP(MNN::Express::_CumProd(*x, axis));
}
VecVARP *mnn_expr_Svd(mnn_expr_VARP_t x) {
  auto p = new MNN::Express::VARPS(MNN::Express::_Svd(*x));
  return new VecVARP{p->data(), p->size()};
}
mnn_expr_VARP_t mnn_expr_Histogram(mnn_expr_VARP_t x, int bin, int min, int max, int channel) {
  return new MNN::Express::VARP(MNN::Express::_Histogram(*x, bin, min, max, channel));
}

// Neural Network Ops
mnn_expr_VARP_t
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
mnn_expr_VARP_t mnn_expr_Clone(mnn_expr_VARP_t source, bool deepCopy) {
  return new MNN::Express::VARP(MNN::Express::_Clone(*source, deepCopy));
}
mnn_expr_VARP_t mnn_expr_Scalar(const void *ptr, halide_type_c_t type) {
  const auto _dtype = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  return new MNN::Express::VARP(MNN::Express::_Scalar(ptr, _dtype));
}

mnn_expr_VARP_t mnn_expr_Const(void* value, const int *shape, size_t shapeLength, int format, halide_type_c_t type) {
  auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
  auto _shape = std::vector<int>(shape, shape + shapeLength);
  return new MNN::Express::VARP(
      MNN::Express::_Const(value, _shape, static_cast<MNN::Express::Dimensionformat>(format), _type)
  );
}

mnn_expr_VARP_t mnn_expr_Conv(
    mnn_expr_VARP_t weight,
    mnn_expr_VARP_t bias,
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t mnn_expr_Deconv(
    mnn_expr_VARP_t weight,
    mnn_expr_VARP_t bias,
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t mnn_expr_MaxPool(
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t mnn_expr_AvePool(
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t
mnn_expr_Reshape(mnn_expr_VARP_t x, const int *shape, size_t shapeLength, int original_format) {
  return new MNN::Express::VARP(
      MNN::Express::_Reshape(
          *x,
          std::vector<int>(shape, shape + shapeLength),
          static_cast<MNN::Express::Dimensionformat>(original_format)
      )
  );
}
mnn_expr_VARP_t mnn_expr_Reshape_1(mnn_expr_VARP_t x, mnn_expr_VARP_t shape) {
  return new MNN::Express::VARP(MNN::Express::_Reshape(*x, *shape));
}
mnn_expr_VARP_t mnn_expr_Scale(
    mnn_expr_VARP_t x,
    int channels,
    float *scales,
    size_t scaleLength,
    float *bias,
    size_t biasLength
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

mnn_expr_VARP_t mnn_expr_Relu(mnn_expr_VARP_t x, float slope) {
  return new MNN::Express::VARP(MNN::Express::_Relu(*x, slope));
}
mnn_expr_VARP_t mnn_expr_Relu6(mnn_expr_VARP_t x, float minValue, float maxValue) {
  return new MNN::Express::VARP(MNN::Express::_Relu6(*x, minValue, maxValue));
}
mnn_expr_VARP_t mnn_expr_PRelu(mnn_expr_VARP_t x, float *slopes, size_t slopeLength) {
  return new MNN::Express::VARP(
      MNN::Express::_PRelu(*x, std::vector<float>(slopes, slopes + slopeLength))
  );
}
mnn_expr_VARP_t mnn_expr_Softmax(mnn_expr_VARP_t logits, int axis) {
  return new MNN::Express::VARP(MNN::Express::_Softmax(*logits, axis));
}
mnn_expr_VARP_t mnn_expr_Softplus(mnn_expr_VARP_t features) {
  return new MNN::Express::VARP(MNN::Express::_Softplus(*features));
}
mnn_expr_VARP_t mnn_expr_Softsign(mnn_expr_VARP_t features) {
  return new MNN::Express::VARP(MNN::Express::_Softsign(*features));
}
VecVARP *
mnn_expr_Split(mnn_expr_VARP_t value, const int *size_splits, size_t size_splitsLength, int axis) {
  auto p = new MNN::Express::VARPS(
      MNN::Express::_Split(
          *value, std::vector<int>(size_splits, size_splits + size_splitsLength), axis
      )
  );
  return new VecVARP{p->data(), p->size()};
}
mnn_expr_VARP_t mnn_expr_Slice(mnn_expr_VARP_t x, mnn_expr_VARP_t starts, mnn_expr_VARP_t sizes) {
  return new MNN::Express::VARP(MNN::Express::_Slice(*x, *starts, *sizes));
}
mnn_expr_VARP_t mnn_expr_StridedSlice(
    mnn_expr_VARP_t input,
    mnn_expr_VARP_t begin,
    mnn_expr_VARP_t end,
    mnn_expr_VARP_t strided,
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
mnn_expr_VARP_t mnn_expr_StridedSliceWrite(
    mnn_expr_VARP_t input,
    mnn_expr_VARP_t begin,
    mnn_expr_VARP_t end,
    mnn_expr_VARP_t strided,
    mnn_expr_VARP_t write,
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
mnn_expr_VARP_t mnn_expr_Concat(VecVARP values, int axis) {
  return new MNN::Express::VARP(
      MNN::Express::_Concat(MNN::Express::VARPS(values.ptr, values.ptr + values.size), axis)
  );
}
mnn_expr_VARP_t mnn_expr_Convert(mnn_expr_VARP_t input, int format) {
  return new MNN::Express::VARP(
      MNN::Express::_Convert(*input, static_cast<MNN::Express::Dimensionformat>(format))
  );
}
mnn_expr_VARP_t mnn_expr_Transpose(mnn_expr_VARP_t x, const int *perm, size_t permLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Transpose(*x, std::vector<int>(perm, perm + permLength))
  );
}
mnn_expr_VARP_t mnn_expr_Transpose_1(mnn_expr_VARP_t x, mnn_expr_VARP_t perm) {
  return new MNN::Express::VARP(MNN::Express::_Transpose(*x, *perm));
}
mnn_expr_VARP_t mnn_expr_ChannelShuffle(mnn_expr_VARP_t x, int group) {
  return new MNN::Express::VARP(MNN::Express::_ChannelShuffle(*x, group));
}
mnn_expr_VARP_t mnn_expr_ChangeInputFormat(mnn_expr_VARP_t input, int format) {
  return new MNN::Express::VARP(
      MNN::Express::_ChangeInputFormat(*input, static_cast<MNN::Express::Dimensionformat>(format))
  );
}
mnn_expr_VARP_t mnn_expr_Reverse(mnn_expr_VARP_t x, mnn_expr_VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_Reverse(*x, *axis));
}
mnn_expr_VARP_t
mnn_expr_ReverseSequence(mnn_expr_VARP_t x, mnn_expr_VARP_t y, int batchDim, int seqDim) {
  return new MNN::Express::VARP(MNN::Express::_ReverseSequence(*x, *y, batchDim, seqDim));
}
mnn_expr_VARP_t mnn_expr_Crop(
    mnn_expr_VARP_t images, mnn_expr_VARP_t size, int axis, const int *offset, size_t offsetLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Crop(*images, *size, axis, std::vector<int>(offset, offset + offsetLength))
  );
}
mnn_expr_VARP_t mnn_expr_Resize(mnn_expr_VARP_t images, float xScale, float yScale) {
  return new MNN::Express::VARP(MNN::Express::_Resize(*images, xScale, yScale));
}
mnn_expr_VARP_t mnn_expr_Pad(mnn_expr_VARP_t x, mnn_expr_VARP_t paddings, int mode) {
  return new MNN::Express::VARP(
      MNN::Express::_Pad(*x, *paddings, static_cast<MNN::Express::PadValueMode>(mode))
  );
}
mnn_expr_VARP_t mnn_expr_ExpandDims(mnn_expr_VARP_t input, int axis) {
  return new MNN::Express::VARP(MNN::Express::_ExpandDims(*input, axis));
}
mnn_expr_VARP_t mnn_expr_ExpandDims_1(mnn_expr_VARP_t input, mnn_expr_VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_ExpandDims(*input, *axis));
}
mnn_expr_VARP_t mnn_expr_Shape(mnn_expr_VARP_t input, bool nchw) {
  return new MNN::Express::VARP(MNN::Express::_Shape(*input, nchw));
}
mnn_expr_VARP_t mnn_expr_Stack(VecVARP values, int axis) {
  return new MNN::Express::VARP(
      MNN::Express::_Stack(MNN::Express::VARPS(values.ptr, values.ptr + values.size), axis)
  );
}
// enum InterpolationMethod {BILINEAR, NEAREST};
mnn_expr_VARP_t mnn_expr_CropAndResize(
    mnn_expr_VARP_t image,
    mnn_expr_VARP_t boxes,
    mnn_expr_VARP_t box_ind,
    mnn_expr_VARP_t crop_size,
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
mnn_expr_VARP_t mnn_expr_Fill(mnn_expr_VARP_t dims, mnn_expr_VARP_t value) {
  return new MNN::Express::VARP(MNN::Express::_Fill(*dims, *value));
}
mnn_expr_VARP_t mnn_expr_Tile(mnn_expr_VARP_t input, mnn_expr_VARP_t multiples) {
  return new MNN::Express::VARP(MNN::Express::_Tile(*input, *multiples));
}
mnn_expr_VARP_t mnn_expr_Gather(mnn_expr_VARP_t params, mnn_expr_VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_Gather(*params, *indices));
}
mnn_expr_VARP_t
mnn_expr_GatherV2(mnn_expr_VARP_t params, mnn_expr_VARP_t indices, mnn_expr_VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_GatherV2(*params, *indices, *axis));
}
mnn_expr_VARP_t mnn_expr_Squeeze(mnn_expr_VARP_t input, const int *axis, size_t axisLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Squeeze(*input, std::vector<int>(axis, axis + axisLength))
  );
}
mnn_expr_VARP_t mnn_expr_Unsqueeze(mnn_expr_VARP_t input, const int *axis, size_t axisLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Unsqueeze(*input, std::vector<int>(axis, axis + axisLength))
  );
}
mnn_expr_VARP_t
mnn_expr_BatchToSpaceND(mnn_expr_VARP_t input, mnn_expr_VARP_t block_shape, mnn_expr_VARP_t crops) {
  return new MNN::Express::VARP(MNN::Express::_BatchToSpaceND(*input, *block_shape, *crops));
}
mnn_expr_VARP_t mnn_expr_GatherND(mnn_expr_VARP_t params, mnn_expr_VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_GatherND(*params, *indices));
}
mnn_expr_VARP_t mnn_expr_GatherElements(mnn_expr_VARP_t params, mnn_expr_VARP_t indices) {
  return new MNN::Express::VARP(MNN::Express::_GatherElements(*params, *indices));
}
mnn_expr_VARP_t
mnn_expr_GatherElements_1(mnn_expr_VARP_t params, mnn_expr_VARP_t indices, mnn_expr_VARP_t axis) {
  return new MNN::Express::VARP(MNN::Express::_GatherElements(*params, *indices, *axis));
}
mnn_expr_VARP_t mnn_expr_Selu(mnn_expr_VARP_t features, float scale, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Selu(*features, scale, alpha));
}
mnn_expr_VARP_t mnn_expr_Size(mnn_expr_VARP_t input) {
  return new MNN::Express::VARP(MNN::Express::_Size(*input));
}
mnn_expr_VARP_t mnn_expr_Elu(mnn_expr_VARP_t features, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Elu(*features, alpha));
}
mnn_expr_VARP_t mnn_expr_Threshold(mnn_expr_VARP_t features, float alpha) {
  return new MNN::Express::VARP(MNN::Express::_Threshold(*features, alpha));
}
mnn_expr_VARP_t mnn_expr_MatrixBandPart(
    mnn_expr_VARP_t input, mnn_expr_VARP_t num_lower, mnn_expr_VARP_t num_upper
) {
  return new MNN::Express::VARP(MNN::Express::_MatrixBandPart(*input, *num_lower, *num_upper));
}
VecVARP *mnn_expr_Moments(
    mnn_expr_VARP_t x, const int *axis, size_t axisLength, mnn_expr_VARP_t shift, bool keepDims
) {
  auto p = new MNN::Express::VARPS(
      MNN::Express::_Moments(*x, std::vector<int>(axis, axis + axisLength), *shift, keepDims)
  );
  return new VecVARP{p->data(), p->size()};
}
mnn_expr_VARP_t mnn_expr_SetDiff1D(mnn_expr_VARP_t x, mnn_expr_VARP_t y) {
  return new MNN::Express::VARP(MNN::Express::_SetDiff1D(*x, *y));
}
mnn_expr_VARP_t mnn_expr_SpaceToDepth(mnn_expr_VARP_t input, int block_size) {
  return new MNN::Express::VARP(MNN::Express::_SpaceToDepth(*input, block_size));
}
mnn_expr_VARP_t mnn_expr_SpaceToBatchND(
    mnn_expr_VARP_t input, mnn_expr_VARP_t block_shape, mnn_expr_VARP_t paddings
) {
  return new MNN::Express::VARP(MNN::Express::_SpaceToBatchND(*input, *block_shape, *paddings));
}
mnn_expr_VARP_t mnn_expr_ZerosLike(mnn_expr_VARP_t input) {
  return new MNN::Express::VARP(MNN::Express::_ZerosLike(*input));
}
VecVARP *mnn_expr_Unstack(mnn_expr_VARP_t value, int axis) {
  auto p = new MNN::Express::VARPS(MNN::Express::_Unstack(*value, axis));
  return new VecVARP{p->data(), p->size()};
}
mnn_expr_VARP_t mnn_expr_Rank(mnn_expr_VARP_t input) {
  return new MNN::Express::VARP(MNN::Express::_Rank(*input));
}
mnn_expr_VARP_t
mnn_expr_Range(mnn_expr_VARP_t start, mnn_expr_VARP_t limit, mnn_expr_VARP_t delta) {
  return new MNN::Express::VARP(MNN::Express::_Range(*start, *limit, *delta));
}
mnn_expr_VARP_t mnn_expr_DepthToSpace(mnn_expr_VARP_t input, int block_size) {
  return new MNN::Express::VARP(MNN::Express::_DepthToSpace(*input, block_size));
}

mnn_expr_VARP_t mnn_expr_Permute(mnn_expr_VARP_t input, const int *dims, size_t dimsLength) {
  return new MNN::Express::VARP(
      MNN::Express::_Permute(*input, std::vector<int>(dims, dims + dimsLength))
  );
}
mnn_expr_VARP_t mnn_expr_Interp(
    VecVARP xs,
    float widthScale,
    float heightScale,
    int outputWidth,
    int outputHeight,
    int resizeType,
    bool alignCorners
) {
  return new MNN::Express::VARP(
      MNN::Express::_Interp(
          MNN::Express::VARPS(xs.ptr, xs.ptr + xs.size),
          widthScale,
          heightScale,
          outputWidth,
          outputHeight,
          resizeType,
          alignCorners
      )
  );
}
mnn_expr_VARP_t mnn_expr_ZeroGrad(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_ZeroGrad(*x));
}
mnn_expr_VARP_t mnn_expr_CosineSimilarity(
    mnn_expr_VARP_t input0, mnn_expr_VARP_t input1, mnn_expr_VARP_t inputDim
) {
  return new MNN::Express::VARP(MNN::Express::_CosineSimilarity(*input0, *input1, *inputDim));
}
// enum GridSamplePaddingMode {GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER,
// GRID_SAMPLE_PADDING_REFLECTION};
mnn_expr_VARP_t mnn_expr_GridSample(
    mnn_expr_VARP_t input, mnn_expr_VARP_t grid, int mode, int paddingMode, bool alignCorners
) {
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
mnn_expr_VARP_t
mnn_expr_FloatToInt8(mnn_expr_VARP_t x, mnn_expr_VARP_t scale, char minValue, char maxValue) {
  return new MNN::Express::VARP(MNN::Express::_FloatToInt8(*x, *scale, minValue, maxValue));
}
mnn_expr_VARP_t mnn_expr_FloatToInt8_1(
    mnn_expr_VARP_t x, mnn_expr_VARP_t scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint
) {
  return new MNN::Express::VARP(
      MNN::Express::_FloatToInt8(*x, *scale, minValue, maxValue, zeroPoint)
  );
}
mnn_expr_VARP_t mnn_expr_Int8ToFloat(mnn_expr_VARP_t x, mnn_expr_VARP_t scale) {
  return new MNN::Express::VARP(MNN::Express::_Int8ToFloat(*x, *scale));
}
mnn_expr_VARP_t mnn_expr_Int8ToFloat_1(mnn_expr_VARP_t x, mnn_expr_VARP_t scale, int8_t zeroPoint) {
  return new MNN::Express::VARP(MNN::Express::_Int8ToFloat(*x, *scale, zeroPoint));
}

mnn_expr_VARP_t
mnn_expr_Select(mnn_expr_VARP_t select, mnn_expr_VARP_t input0, mnn_expr_VARP_t input1) {
  return new MNN::Express::VARP(MNN::Express::_Select(*select, *input0, *input1));
}
VecVARP *mnn_expr_TopKV2(mnn_expr_VARP_t input0, mnn_expr_VARP_t input1) {
  auto p = new MNN::Express::VARPS(MNN::Express::_TopKV2(*input0, *input1));
  return new VecVARP{p->data(), p->size()};
}
// MNN_PUBLIC VARP _ImageProcess(VARP input, CV::ImageProcess::Config config, CV::Matrix matrix, int
// oh, int ow, int oc, int dtype, uint8_t padVal = 0); mnn_expr_VARP_t
// mnn_expr_ImageProcess(mnn_expr_VARP_t input, CV::ImageProcess::Config config, CV::Matrix matrix,
// int oh, int ow, int oc, int dtype, uint8_t padVal);
mnn_expr_VARP_t mnn_expr_Where(mnn_expr_VARP_t x) {
  return new MNN::Express::VARP(MNN::Express::_Where(*x));
}
mnn_expr_VARP_t mnn_expr_Sort(mnn_expr_VARP_t x, int axis, bool arg, bool descend) {
  return new MNN::Express::VARP(MNN::Express::_Sort(*x, axis, arg, descend));
}
mnn_expr_VARP_t mnn_expr_Raster(
    VecVARP vars, const int *regions, size_t regionsLength, const int *shape, size_t shapeLength
) {
  return new MNN::Express::VARP(
      MNN::Express::_Raster(
          MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size),
          std::vector<int>(regions, regions + regionsLength),
          std::vector<int>(shape, shape + shapeLength)
      )
  );
}
mnn_expr_VARP_t mnn_expr_RasterRaw(
    VecVARP vars,
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
          MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size),
          std::vector<int>(region, region + regionLength),
          std::vector<int>(shape, shape + shapeLength),
          _dtype,
          static_cast<MNN::Express::Dimensionformat>(format)
      )
  );
}

mnn_expr_VARP_t mnn_expr_Nms(
    mnn_expr_VARP_t boxes,
    mnn_expr_VARP_t scores,
    int maxDetections,
    float iouThreshold,
    float scoreThreshold
) {
  return new MNN::Express::VARP(
      MNN::Express::_Nms(*boxes, *scores, maxDetections, iouThreshold, scoreThreshold)
  );
}
mnn_expr_VARP_t mnn_expr_Im2Col(
    mnn_expr_VARP_t x,
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
mnn_expr_VARP_t mnn_expr_Col2Im(
    mnn_expr_VARP_t x,
    mnn_expr_VARP_t outputShape,
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
