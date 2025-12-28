// ignore_for_file: camel_case_types

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mnn/src/base.dart';
import 'package:mnn/src/expr/expr.dart';
import 'package:mnn/src/expr/utils.dart';
import 'package:mnn/src/g/mnn.g.dart' as C;
import 'package:mnn/src/halide_runtime.dart';
import 'package:mnn/src/vec.dart';

enum PaddingMode {
  CAFFE(0),
  VALID(1),
  SAME(2)
  ;

  final int value;

  const PaddingMode(this.value);

  factory PaddingMode.fromValue(int value) => switch (value) {
    0 => CAFFE,
    1 => VALID,
    2 => SAME,
    _ => throw ArgumentError.value(value, 'value', 'Invalid PaddingMode value'),
  };
}

enum PoolingMode {
  MAXPOOL(0),
  AVEPOOL(1)
  ;

  final int value;

  const PoolingMode(this.value);

  factory PoolingMode.fromValue(int value) => switch (value) {
    0 => MAXPOOL,
    1 => AVEPOOL,
    _ => throw ArgumentError.value(value, 'value', 'Invalid PoolingMode value'),
  };
}

enum PadValueMode {
  CONSTANT(0),
  REFLECT(1),
  SYMMETRIC(2),
  EDGE(3)
  ;

  final int value;

  const PadValueMode(this.value);

  factory PadValueMode.fromValue(int value) => switch (value) {
    0 => CONSTANT,
    1 => REFLECT,
    2 => SYMMETRIC,
    3 => EDGE,
    _ => throw ArgumentError.value(value, 'value', 'Invalid PadValueMode value'),
  };
}

enum InterpolationMethod {
  // BILINEAR, NEAREST
  BILINEAR(0),
  NEAREST(1)
  ;

  final int value;
  const InterpolationMethod(this.value);

  factory InterpolationMethod.fromValue(int value) => switch (value) {
    0 => BILINEAR,
    1 => NEAREST,
    _ => throw ArgumentError.value(value, 'value', 'Invalid InterpolationMethod value'),
  };
}

enum GridSamplePaddingMode {
  GRID_SAMPLE_PADDING_ZEROS(0),
  GRID_SAMPLE_PADDING_BORDER(1),
  GRID_SAMPLE_PADDING_REFLECTION(2)
  ;

  final int value;
  const GridSamplePaddingMode(this.value);

  factory GridSamplePaddingMode.fromValue(int value) => switch (value) {
    0 => GRID_SAMPLE_PADDING_ZEROS,
    1 => GRID_SAMPLE_PADDING_BORDER,
    2 => GRID_SAMPLE_PADDING_REFLECTION,
    _ => throw ArgumentError.value(value, 'value', 'Invalid GridSamplePaddingMode value'),
  };
}

// BinaryOPs

/// Returns [x] + [y] element-wise.
///
/// Args:
/// - [x] A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
/// - [y] A variable. Must have the same type as [x].
///
/// Returns:
/// - A variable. Has the same type as [x].
VARP add(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Add(x.ptr, y.ptr));

/// Returns [x] - [y] element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
/// - y: A variable. Must have the same type as [x].
///
/// Returns:
/// - A variable. Has the same type as [x].
VARP subtract(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Subtract(x.ptr, y.ptr));

/// Returns x * y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP multiply(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Multiply(x.ptr, y.ptr));

/// Computes Python style division of x by y.
///
/// Args:
///   - [x]: A variable. Must be one of the following types:
///       Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8.
///   - [y]: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP divide(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Divide(x.ptr, y.ptr));

/// Computes the power of one value to another.
///
/// Args:
/// - x: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
/// - y: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
///
/// Returns:
/// - A variable. Has the same type as x.
VARP pow(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Pow(x.ptr, y.ptr));

/// Returns the min of x and y (i.e. x < y ? x : y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP minimum(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Minimum(x.ptr, y.ptr));

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types:
///     Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP maximum(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Maximum(x.ptr, y.ptr));

/// Adds bias to value.
///
/// This is (mostly) a special case of add where bias is restricted to 1-D.
/// Broadcasting is supported, so value may have any number of dimensions.
/// Unlike add, the type of bias is allowed to differ from value in the case where both types are quantized.
///
/// Args:
/// - value: A variable with type Halide_Type_Float, Halide_Type_Int
/// - bias: A 1-D variable with size matching the channel dimension of value.
///     Must be the same type as value unless value is a quantized type, in which case a different quantized type may be used.
///
/// Returns:
/// - A variable with the same type as value.
VARP biasAdd(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BiasAdd(x.ptr, y.ptr));

/// Returns the truth value of (x > y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable of type bool.
VARP greater(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Greater(x.ptr, y.ptr));

/// Returns the truth value of (x >= y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable of type bool.
VARP greaterEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_GreaterEqual(x.ptr, y.ptr));

/// Returns the truth value of (x < y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable of type bool.
VARP less(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Less(x.ptr, y.ptr));

/// Returns the value of (x // y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP floorDiv(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_FloorDiv(x.ptr, y.ptr));

/// Returns the value of (x - y)(x - y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP squaredDifference(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_SquaredDifference(x.ptr, y.ptr));

/// Returns the truth value of (x == y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable of type bool.
VARP equal(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Equal(x.ptr, y.ptr));

/// Returns the truth value of (x <= y) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable of type bool.
VARP lessEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_LessEqual(x.ptr, y.ptr));

/// Returns element-wise remainder of division
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP floorMod(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_FloorMod(x.ptr, y.ptr));

/// Computes arctangent of `y/x` element-wise, respecting signs of the arguments.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP atan2(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Atan2(x.ptr, y.ptr));

/// Returns the truth value of x OR y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP logicalOr(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_LogicalOr(x.ptr, y.ptr));

/// Returns the truth value of x != y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP notEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_NotEqual(x.ptr, y.ptr));

/// Returns the truth value of x & y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP bitwiseAnd(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseAnd(x.ptr, y.ptr));

/// Returns the truth value of x | y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP bitwiseOr(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseOr(x.ptr, y.ptr));

/// Returns the truth value of x ^ y element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int
/// - y: A variable. Must have the same type as x.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP bitwiseXor(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseXor(x.ptr, y.ptr));

// UnaryOPs

/// Computes sign of x eltment-wise
///
///  sign(x) = 0 if x=0
///  sign(x) =-1 if x<0
///  sign(x) = 1 if x>0
VARP sign(VARP x) => VARP.fromPointer(C.mnn_expr_Sign(x.ptr));

/// Computes the absolute value of a variable.
/// Given a variable of integer or floating-point values, this operation returns a variable of the same type,
/// where each element contains the absolute value of the corresponding element in the input.
/// ```c++
/// x = MNN.const((-1.0, -2.0, 3.0), (3, ))
/// x = MNN.abs(x)  # (1.0, 2.0, 3.0)
///```
///
/// Args:
/// - x: A variable of type Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable the same size, type as x with absolute values.
VARP abs(VARP x) => VARP.fromPointer(C.mnn_expr_Abs(x.ptr));

/// Computes numerical negative value element-wise.
/// ```c++
/// x = MNN.const((-1.0, -2.0, 3.0), (3, ))
/// x = MNN.negative(x) #(1.0, 2.0, -3.0)
///```
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP negative(VARP x) => VARP.fromPointer(C.mnn_expr_Negative(x.ptr));

/// Returns element-wise largest integer not greater than x.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP floor(VARP x) => VARP.fromPointer(C.mnn_expr_Floor(x.ptr));

/// Returns element-wise rounded integer not less than x.
///
/// Args:
/// - x: A variable. Must be Halide_Type_Float
///
/// Returns:
/// - A variable. Halide_Type_Float.
VARP round(VARP x) => VARP.fromPointer(C.mnn_expr_Round(x.ptr));

/// Returns element-wise smallest integer not less than x.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP ceil(VARP x) => VARP.fromPointer(C.mnn_expr_Ceil(x.ptr));

/// Computes square of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP square(VARP x) => VARP.fromPointer(C.mnn_expr_Square(x.ptr));

/// Computes square root of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP sqrt(VARP x) => VARP.fromPointer(C.mnn_expr_Sqrt(x.ptr));

/// Computes reciprocal of square root of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP rsqrt(VARP x) => VARP.fromPointer(C.mnn_expr_Rsqrt(x.ptr));

/// Computes exponential of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP exp(VARP x) => VARP.fromPointer(C.mnn_expr_Exp(x.ptr));

/// Computes natural logarithm of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP log(VARP x) => VARP.fromPointer(C.mnn_expr_Log(x.ptr));

/// Computes sine of x element-wise.
///
/// Given an input variable, this function computes sine of every element in the variable.
/// Input range is (-inf, inf) and output range is [-1,1].
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP sin(VARP x) => VARP.fromPointer(C.mnn_expr_Sin(x.ptr));

/// Computes sinh of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP sinh(VARP x) => VARP.fromPointer(C.mnn_expr_Sinh(x.ptr));

/// Computes cos of x element-wise.
///
/// Given an input variable, this function computes cosine of every element in the variable.
/// Input range is (-inf, inf) and output range is [-1,1]. If input lies outside the boundary, nan is returned.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP cos(VARP x) => VARP.fromPointer(C.mnn_expr_Cos(x.ptr));

/// Computes cosh of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP cosh(VARP x) => VARP.fromPointer(C.mnn_expr_Cosh(x.ptr));

/// Computes tan of x element-wise.
///
/// Given an input variable, this function computes tangent of every element in the variable.
/// Input range is (-inf, inf) and output range is (-inf, inf). If input lies outside the boundary, nan is returned.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP tan(VARP x) => VARP.fromPointer(C.mnn_expr_Tan(x.ptr));

/// Computes hyperbolic tangent of x element-wise.
///
/// Given an input variable, this function computes hyperbolic tangent of every element in the variable.
///
/// Input range is [-inf, inf] and output range is [-1,1].
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP tanh(VARP x) => VARP.fromPointer(C.mnn_expr_Tanh(x.ptr));

/// Computes the trignometric inverse sine of x element-wise.
///
/// The asin operation returns the inverse of sin, such that if y = sin(x) then, x = asin(y).
///
/// Note: The output of asin will lie within the invertible range of sine, i.e [-pi/2, pi/2].
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP asin(VARP x) => VARP.fromPointer(C.mnn_expr_Asin(x.ptr));

/// Computes asinh of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP asinh(VARP x) => VARP.fromPointer(C.mnn_expr_Asinh(x.ptr));

/// Computes acos of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP acos(VARP x) => VARP.fromPointer(C.mnn_expr_Acos(x.ptr));

/// Computes acosh of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within (0, +inf). The input lies in [1, +inf)
///
/// Returns:
/// - A variable. Has the same type as x.
VARP acosh(VARP x) => VARP.fromPointer(C.mnn_expr_Acosh(x.ptr));

/// Computes the trignometric inverse tangent of x element-wise.
///
/// The atan operation returns the inverse of tan, such that if y = tan(x) then, x = atan(y).
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (-pi/2, pi/2).
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP atan(VARP x) => VARP.fromPointer(C.mnn_expr_Atan(x.ptr));

/// Computes atanh of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The input of atanh will lie within (-1, 1). The output of atan will lie within (-inf, +inf).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP atanh(VARP x) => VARP.fromPointer(C.mnn_expr_Atanh(x.ptr));

/// Computes the reciprocal of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP reciprocal(VARP x) => VARP.fromPointer(C.mnn_expr_Reciprocal(x.ptr));

/// Computes natural logarithm of (1 + x) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP log1p(VARP x) => VARP.fromPointer(C.mnn_expr_Log1p(x.ptr));

/// Computes Gelu of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x .
VARP gelu(VARP x) => VARP.fromPointer(C.mnn_expr_Gelu(x.ptr));

/// Computes sigmoid of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP sigmoid(VARP x) => VARP.fromPointer(C.mnn_expr_Sigmoid(x.ptr));

/// Computes the Gauss error function of `x` element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within (-1.0, 1.0). The input will lie in (-inf, inf)
///
/// Returns:
/// - A variable. Has the same type as x.
VARP erf(VARP x) => VARP.fromPointer(C.mnn_expr_Erf(x.ptr));

/// Computes the complementary error function of `x` element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The output of atan will lie within the invertible range of tan, i.e (0.0, pi).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP erfc(VARP x) => VARP.fromPointer(C.mnn_expr_Erfc(x.ptr));

/// Computes the inverse function for erf, for `x` element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float
///
/// Note: The input of atan will lie within (-1, 1).
///
/// Returns:
/// - A variable. Has the same type as x.
VARP erfinv(VARP x) => VARP.fromPointer(C.mnn_expr_Erfinv(x.ptr));

/// Computes ((exponential of x) - 1) element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP expm1(VARP x) => VARP.fromPointer(C.mnn_expr_Expm1(x.ptr));

/// Computes Hardswish of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x .
VARP hardswish(VARP x) => VARP.fromPointer(C.mnn_expr_Hardswish(x.ptr));

/// Computes sigmoid of x element-wise.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
///
/// Returns:
/// - A variable. Has the same type as x.
VARP silu(VARP x) => VARP.fromPointer(C.mnn_expr_Silu(x.ptr));

// ReduceOPs

/// Computes the sum of elements across dimensions of a variable
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceSum(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceSum(x.ptr, axis.i32.ptr, keepDims));

/// Computes the mean of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceMean(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMean(x.ptr, axis.i32.ptr, keepDims));

/// Computes the variance of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
// VARP reduceVariance(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
//     VARP.fromPointer(C.mnn_expr_ReduceVariance(x.ptr, axis.i32.ptr, keepDims));

/// Computes the maximum of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceMax(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMax(x.ptr, axis.i32.ptr, keepDims));

/// Computes the minimum of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceMin(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMin(x.ptr, axis.i32.ptr, keepDims));

/// Computes the product of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have numeric type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceProd(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceProd(x.ptr, axis.i32.ptr, keepDims));

/// Computes the "logical or" of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have booling type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceAny(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAny(x.ptr, axis.i32.ptr, keepDims));

/// Computes the "logical and" of elements across dimensions of a variable.
///
/// Reduces input_variable along the dimensions given in axis.
///
/// Unless keepdims is true, the rank of the variable is reduced by 1 for each entry in axis.
///
/// If keepdims is true, the reduced dimensions are retained with length 1.
///
/// If axis is empty, all dimensions are reduced, and a variable with a single element is returned.
///
/// Args:
/// - input_variable: The variable to reduce. Should have booling type.
/// - axis: The dimensions to reduce. If empty(the default), reduces all dimensions.
///        Must be in the range [-rank(input_variable), rank(input_variable)).
/// - keepdims: If true, retains reduced dimensions with length 1.
///
/// Returns:
/// - The reduced variable, of the same dtype as the input_variable.
VARP reduceAll(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAll(x.ptr, axis.i32.ptr, keepDims));

VARP reduceSumMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceSumMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceSumMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceMeanMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceMeanMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceMeanMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceMaxMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceMaxMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceMaxMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceMinMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceMinMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceMinMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceProdMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceProdMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceProdMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceAnyMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceAnyMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceAnyMutable(x.ptr, axis.ptr, keepDims));
}

VARP reduceAllMutable(VARP x, {VARP? axis, bool keepDims = false}) {
  if (axis == null) {
    return VARP.fromPointer(C.mnn_expr_ReduceAllMutable(reshape(x, [-1]).ptr, scalar<i32>(0).ptr, keepDims));
  }
  return VARP.fromPointer(C.mnn_expr_ReduceAllMutable(x.ptr, axis.ptr, keepDims));
}

// EltwiseOPs

/// Compute the element-wise prod
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
/// - y: A variable. Must be one of the following types: Halide_Type_Float
/// - coeff: blob-wise coefficients
///
/// Returns:
/// - The prod variable.
VARP prod(VARP x, VARP y, {List<double> coeff = const []}) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Prod(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

/// Compute the element-wise sum
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
/// - y: A variable. Must be one of the following types: Halide_Type_Float
/// - coeff: blob-wise coefficients
///
/// Returns:
/// - The sum variable.
VARP sum(VARP x, VARP y, {List<double> coeff = const []}) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Sum(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

/// Compute the element-wise max
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
/// - y: A variable. Must be one of the following types: Halide_Type_Float
/// - coeff: blob-wise coefficients
///
/// Returns:
/// - The max variable.
VARP max(VARP x, VARP y, {List<double> coeff = const []}) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Max(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

/// Compute the element-wise sub
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Float
/// - y: A variable. Must be one of the following types: Halide_Type_Float
/// - coeff: blob-wise coefficients
///
/// Returns:
/// - The sub variable.
VARP sub(VARP x, VARP y, {List<double> coeff = const []}) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Sub(x.ptr, y.ptr, p, coeff.length));
  calloc.free(p);
  return rval;
}

VARP mod(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Mod(x.ptr, y.ptr));

// OtherOPs

/// Casts a variable to a new type.
///
/// Args:
/// - x: A variable. Must be one of the following types: Halide_Type_Int or Halide_Type_Float, Halide_Type_Int64, Halide_Type_Uint8
/// - dtype: The destination type. The list of supported dtypes is the same as x.
///
/// Returns:
/// - A variable with same shape as x and same type as dtype.
VARP cast<T extends ffi.SizedNativeType>(VARP x) =>
    VARP.fromPointer(C.mnn_expr_Cast(x.ptr, HalideType.of<T>().native.ref));

/// Multiply the matrix "a" by the matrix "b".
///
/// The inputs must be two-dimensional matrices and the inner dimension of "a" (after being transposed if transpose_a is true)
///
/// must match the outer dimension of "b" (after being transposed if transposed_b is true).
///
/// Arguments:
/// - a: a variable representing a matrix "a"
/// - b: a variable representing a matrix "b"
/// - tranposeA: If true, "a" is transposed before multiplication.
/// - tranposeB: If true, "b" is transposed before multiplication.
///
/// Returns:
/// - The product variable.
VARP matMul(VARP x, VARP y, {bool transposeA = false, bool transposeB = false}) =>
    VARP.fromPointer(C.mnn_expr_MatMul(x.ptr, y.ptr, transposeA, transposeB));
VARP normalize(VARP x, int acrossSpatial, int channelShared, double eps, List<double> scale) {
  final (p, size) = scale.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Normalize(x.ptr, acrossSpatial, channelShared, eps, p, size));
  calloc.free(p);
  return rval;
}

/// Returns the index with the largest value across axes of a tensor.
///
/// Args:
/// - input: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - axis: A int. must be in the range -rank(input), rank(input)).
///         Describes which axis of the input variable to reduce across. For vectors, use axis = 0.
///
/// Returns:
/// - A variable of type int.
VARP argMax(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ArgMax(x.ptr, axis));

/// Returns the index with the smallest value across axes of a tensor.
/// Args:
/// - input: A variable. Must be one of the following types: Halide_Type_Float, Halide_Type_Int
/// - axis: A int.
///             must be in the range -rank(input), rank(input)). Describes which axis of the input variable to reduce across.
///             For vectors, use axis = 0.
///
/// Returns:
/// - A variable of type int.
VARP argMin(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ArgMin(x.ptr, axis));

/// Multiplies slices of two variable in batches
///
/// Multiplies all slices of variable x and y (each slice can be viewed as an element of a batch),
/// and arranges the individual results in a single output variable of the same batch size.
///
/// Each of the individual slices can optionally be adjointed (to adjoint a matrix means to transpose and conjugate it)
/// before multiplication by setting the adj_x or adj_y flag to True, which are by default False.
///
/// The input variable x and y are 2-D or higher with shape [..., r_x, c_x] and [..., r_y, c_y].
///
/// The output variable is 2-D or higher with shape [..., r_o, c_o], where:
/// r_o = c_x if adj_x else r_x
/// c_o = r_y if adj_y else c_y
///
/// It is computed as:
/// output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])
///
/// Arguments:
/// - x: 2-D or higher with shape [..., r_x, c_x].
/// - y: 2-D or higher with shape [..., r_y, c_y].
///
/// Optional:
/// - adj_x: If True, adjoint the slices of x. Defaults to False.
/// - adj_y: If True, adjoint the slices of y. Defaults to False.
///
/// Returns:
/// - Output: 3-D or higher with shape [..., r_o, c_o]
VARP batchMatMul(VARP x, VARP y, {bool adjX = false, bool adjY = false}) =>
    VARP.fromPointer(C.mnn_expr_BatchMatMul(x.ptr, y.ptr, adjX, adjY));
VARP unravelIndex(VARP indices, VARP dims) =>
    VARP.fromPointer(C.mnn_expr_UnravelIndex(indices.ptr, dims.ptr));
VARP scatterND(VARP indices, VARP updates, VARP shape, {VARP? input, int? reduction}) => switch ((
  input,
  reduction,
)) {
  (null, null) => VARP.fromPointer(C.mnn_expr_ScatterNd(indices.ptr, updates.ptr, shape.ptr)),
  (VARP(), null) => VARP.fromPointer(C.mnn_expr_ScatterNd_1(indices.ptr, updates.ptr, shape.ptr, input!.ptr)),
  (null, int()) => VARP.fromPointer(C.mnn_expr_ScatterNd_2(indices.ptr, updates.ptr, shape.ptr, reduction!)),
  (VARP(), int()) => VARP.fromPointer(
    C.mnn_expr_ScatterNd_3(indices.ptr, updates.ptr, shape.ptr, input!.ptr, reduction!),
  ),
};
VARP scatterElements(VARP data, VARP indices, VARP updates, {VARP? axis, int reduction = -1}) =>
    switch (axis) {
      null => VARP.fromPointer(C.mnn_expr_ScatterElements(data.ptr, indices.ptr, updates.ptr, reduction)),
      VARP() => VARP.fromPointer(
        C.mnn_expr_ScatterElements_1(data.ptr, indices.ptr, updates.ptr, axis.ptr, reduction),
      ),
    };
VARP oneHot(VARP indices, VARP depth, VARP onValue, VARP offValue, int axis) =>
    VARP.fromPointer(C.mnn_expr_OneHot(indices.ptr, depth.ptr, onValue.ptr, offValue.ptr, axis));
VARP broadcastTo(VARP x, VARP shape) => VARP.fromPointer(C.mnn_expr_BroadcastTo(x.ptr, shape.ptr));
VARP linSpace(VARP start, VARP stop, VARP num) =>
    VARP.fromPointer(C.mnn_expr_LinSpace(start.ptr, stop.ptr, num.ptr));
VARP randomUniform(
  VARP shape,
  HalideType dtype, {
  double low = 0.0,
  double high = 1.0,
  int seed0 = 0,
  int seed1 = 0,
}) => VARP.fromPointer(C.mnn_expr_RandomUniform(shape.ptr, dtype.native.ref, low, high, seed0, seed1));
VARP cumSum(VARP x, int axis, {bool exclusive = false, bool reverse = false}) =>
    VARP.fromPointer(C.mnn_expr_CumSum(x.ptr, axis, exclusive, reverse));
VARP cumProd(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_CumProd(x.ptr, axis));
List<VARP> svd(VARP x) {
  final p = C.mnn_expr_Svd(x.ptr);
  final size = C.mnn_expr_VecVARP_size(p);
  final rval = List.generate(size, (i) => VARP.fromPointer(C.mnn_expr_VecVARP_at(p, i)));
  C.mnn_expr_VecVARP_free(p);
  return rval;
}

VARP histogram(VARP x, int bin, int min, int max, int channel) =>
    VARP.fromPointer(C.mnn_expr_Histogram(x.ptr, bin, min, max, channel));

// Neural Network Ops

/// create a input variable.
///
/// Args:
/// - shape: A vector, the shape of the variable.
/// - data_format: A enum, NCHW/NHWC/NC4HW4 is allowed.
/// - dtype: The type of the elements of the resulting variable.
///
/// Returns:
/// - output: A variable.
VARP input(
  List<int> shape, {
  DimensionFormat dataFormat = DimensionFormat.NC4HW4,
  HalideType dtype = HalideType.f32,
}) {
  final (p, size) = shape.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Input(p.cast(), size, dataFormat.value, dtype.native.ref));
  calloc.free(p);
  return rval;
}

VARP clone(VARP x, {bool deepCopy = false}) => VARP.fromPointer(C.mnn_expr_Clone(x.ptr, deepCopy));

typedef uint8 = ffi.Uint8;
typedef uint16 = ffi.Uint16;
typedef uint32 = ffi.Uint32;
typedef uint64 = ffi.Uint64;
typedef int8 = ffi.Int8;
typedef int16 = ffi.Int16;
typedef int32 = ffi.Int32;
typedef int64 = ffi.Int64;
typedef float32 = ffi.Float;
typedef float64 = ffi.Double;

ffi.Pointer<T> _pointerOf<T extends ffi.SizedNativeType>(Iterable<num> data) {
  ffi.Pointer pdata;
  if (T == uint8) {
    pdata = calloc<uint8>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == uint16) {
    pdata = calloc<uint16>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == uint32) {
    pdata = calloc<uint32>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == uint64) {
    pdata = calloc<uint64>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == int8) {
    pdata = calloc<int8>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == int16) {
    pdata = calloc<int16>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == int32) {
    pdata = calloc<int32>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == int64) {
    pdata = calloc<int64>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toInt()));
  } else if (T == float32) {
    pdata = calloc<float32>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toDouble()));
  } else if (T == float64) {
    pdata = calloc<float64>(data.length)..asTypedList(data.length).setAll(0, data.map((e) => e.toDouble()));
  } else {
    throw ArgumentError.value(T, 'T', 'Unsupported type');
  }
  return pdata.cast<T>();
}

/// create a constant variable.
///
/// Args:
/// - data: Indicates the values.
/// - shape: A vector, the shape of the variable.
/// - format: A enum, NCHW/NHWC/NC4HW4 is allowed.
/// - type: The type of the elements of the resulting variable.
///
/// Returns:
/// - output: A constant variable.
VARP constant<T extends ffi.SizedNativeType>(
  Iterable<num> data,
  Iterable<int> shape, {
  DimensionFormat format = DimensionFormat.NHWC,
}) {
  final nElem = shape.reduce((a, b) => a * b);
  MnnAssert(
    data.length == nElem,
    'data.length=${data.length} must be equal to the dot product of shape=$nElem',
  );

  final pdata = _pointerOf<T>(data);
  final pshape = _pointerOf<int32>(shape);
  final pvar = C.mnn_expr_Const(
    pdata.cast(),
    pshape.cast(),
    shape.length,
    format.value,
    HalideType.of<T>().native.ref,
  );
  // memories of pdata will be copied internally, so here we need to free them
  calloc.free(pdata);
  calloc.free(pshape);
  return VARP.fromPointer(pvar);
}

VARP scalar<T extends ffi.SizedNativeType>(num value) {
  final pdata = _pointerOf<T>([value]);
  final pvar = C.mnn_expr_Scalar(pdata.cast(), HalideType.of<T>().native.ref);
  calloc.free(pdata);
  return VARP.fromPointer(pvar);
}

VARP conv(
  VARP weight,
  VARP bias,
  VARP x, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> dilatie = const [1, 1],
  int group = 1,
  List<int> pads = const [0, 0],
}) {
  final (stridePtr, strideSize) = stride.toNativeArrayI32();
  final (dilationPtr, dilationSize) = dilatie.toNativeArrayI32();
  final (padPtr, padSize) = pads.toNativeArrayI32();
  final rval = VARP.fromPointer(
    C.mnn_expr_Conv(
      weight.ptr,
      bias.ptr,
      x.ptr,
      pad.value,
      stridePtr.cast(),
      strideSize,
      dilationPtr.cast(),
      dilationSize,
      group,
      padPtr.cast(),
      padSize,
    ),
  );
  calloc.free(stridePtr);
  calloc.free(dilationPtr);
  calloc.free(padPtr);
  return rval;
}

VARP conv2d(
  VARP weight,
  VARP bias,
  VARP x, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> dilatie = const [1, 1],
  int group = 1,
  List<int> pads = const [0, 0],
}) => conv(weight, bias, x, pad: pad, stride: stride, dilatie: dilatie, group: group, pads: pads);

VARP deconv(
  VARP weight,
  VARP bias,
  VARP x, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> dilatie = const [1, 1],
  int group = 1,
  List<int> pads = const [0, 0],
}) {
  final (stridePtr, strideSize) = stride.toNativeArrayI32();
  final (dilationPtr, dilationSize) = dilatie.toNativeArrayI32();
  final (padPtr, padSize) = pads.toNativeArrayI32();
  final rval = VARP.fromPointer(
    C.mnn_expr_Deconv(
      weight.ptr,
      bias.ptr,
      x.ptr,
      pad.value,
      stridePtr.cast(),
      strideSize,
      dilationPtr.cast(),
      dilationSize,
      group,
      padPtr.cast(),
      padSize,
    ),
  );
  calloc.free(stridePtr);
  calloc.free(dilationPtr);
  calloc.free(padPtr);
  return rval;
}

VARP conv2dTranspose(
  VARP weight,
  VARP bias,
  VARP x, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> dilatie = const [1, 1],
  int group = 1,
  List<int> pads = const [0, 0],
}) => deconv(weight, bias, x, pad: pad, stride: stride, dilatie: dilatie, group: group, pads: pads);

VARP maxPool(
  VARP x,
  List<int> kernal, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> pads = const [0, 0],
}) {
  final (kernalPtr, kernalSize) = kernal.toNativeArrayI32();
  final (stridePtr, strideSize) = stride.toNativeArrayI32();
  final (padPtr, padSize) = pads.toNativeArrayI32();
  final rval = VARP.fromPointer(
    C.mnn_expr_MaxPool(
      x.ptr,
      kernalPtr.cast(),
      kernalSize,
      stridePtr.cast(),
      strideSize,
      pad.value,
      padPtr.cast(),
      padSize,
    ),
  );
  calloc.free(kernalPtr);
  calloc.free(stridePtr);
  calloc.free(padPtr);
  return rval;
}

VARP avgPool(
  VARP x,
  List<int> kernal, {
  PaddingMode pad = PaddingMode.VALID,
  List<int> stride = const [1, 1],
  List<int> pads = const [0, 0],
}) {
  final (kernalPtr, kernalSize) = kernal.toNativeArrayI32();
  final (stridePtr, strideSize) = stride.toNativeArrayI32();
  final (padPtr, padSize) = pads.toNativeArrayI32();
  final rval = VARP.fromPointer(
    C.mnn_expr_AvePool(
      x.ptr,
      kernalPtr.cast(),
      kernalSize,
      stridePtr.cast(),
      strideSize,
      pad.value,
      padPtr.cast(),
      padSize,
    ),
  );
  calloc.free(kernalPtr);
  calloc.free(stridePtr);
  calloc.free(padPtr);
  return rval;
}

/// Reshapes a variable.
///
/// Args:
/// - x: A variable.
/// - shape: A vector, the shape of the target variable.
/// - original_format: A enum, only NCHW/NHWC is allowed, NC4HW4 is not allowed,
/// as it provides additional information(x comes from NCHW or NHWC) When x is NC4HW4.
///
/// Returns:
/// - output: A variable with the same type as `x`.
VARP reshape(VARP x, List<int> shape, {DimensionFormat format = DimensionFormat.NCHW}) {
  final (shapePtr, shapeSize) = shape.toNativeArrayI32();
  final rval = VARP.fromPointer(
    C.mnn_expr_Reshape(
      x.ptr,
      shapePtr.cast(),
      shapeSize,
      format.value,
    ),
  );
  calloc.free(shapePtr);
  return rval;
}

VARP scale(VARP x, int channels, List<double> scales, List<double> biases) {
  final (scalesPtr, scalesSize) = scales.toNativeArrayF32();
  final (biasesPtr, biasesSize) = biases.toNativeArrayF32();
  final rval = VARP.fromPointer(
    C.mnn_expr_Scale(
      x.ptr,
      channels,
      scalesPtr.cast(),
      scalesSize,
      biasesPtr.cast(),
      biasesSize,
    ),
  );
  calloc.free(scalesPtr);
  calloc.free(biasesPtr);
  return rval;
}

/// Given an input value x, it computes the output as x if x > 0 and slope * x if x <= 0.
///
/// Args:
/// - x: A variable.
/// - slope: A float, a positive float value, it leakes the negative part by multiplying with `slope` rather than setting it to 0.0f.
///
/// Returns:
/// - output: A variable with the same type as `x`.
VARP ReLU(VARP x, {double slope = 0.0}) => VARP.fromPointer(C.mnn_expr_Relu(x.ptr, slope));

/// Given an input value x, it computes Rectified Linear 6: min(max(x, 0), 6).
///
/// Args:
/// - x: A variable.
///
/// Returns:
/// - output: A variable with the same type as `x`.
VARP ReLU6(VARP x, {double slope = 0.0, double maxValue = 6.0}) =>
    VARP.fromPointer(C.mnn_expr_Relu6(x.ptr, slope, maxValue));

/// Given an input value x, it computes the output as x if x > 0 and slopes * x if x <= 0.
///
/// Args:
/// - x: A variable, must be 4-D with NC4HW4 format.
/// - slopes: A vector, has save size as x.
///
/// Returns:
/// - output: A variable with the same type as `x`.
VARP PReLU(VARP x, List<double> slopes) {
  final (slopesPtr, slopesSize) = slopes.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_PRelu(x.ptr, slopesPtr.cast(), slopesSize));
  calloc.free(slopesPtr);
  return rval;
}

/// Computes softmax activations.
///
/// Args:
/// - logits: A non-empty variable. Must be Halide_Type_Float.
/// - axis: The dimension softmax would be performed on. The default is -1 which indicates the last dimension.
///
/// Returns:
/// - output: A variable with the same type as `logits`.
VARP softMax(VARP logits, {int axis = -1}) => VARP.fromPointer(C.mnn_expr_Softmax(logits.ptr, axis));

/// Computes softplus: log(exp(features) + 1).
///
/// Args:
/// - features: A variable. Must be Halide_Type_Float.
///
/// Returns:
/// - A variable with the same type as `features`.
VARP softPlus(VARP features) => VARP.fromPointer(C.mnn_expr_Softplus(features.ptr));

/// Computes softsign: features / (abs(features) + 1).
///
/// Args:
/// - features: A variable. Must be Halide_Type_Float.
///
/// Returns:
/// - A variable with the same type as `features`.
VARP softSign(VARP features) => VARP.fromPointer(C.mnn_expr_Softsign(features.ptr));

/// Splits a variable value into a list of sub variables.
///
/// Args:
/// - value: The variable to split.
/// - size_splits: A vector, a 1-D integer containing the sizes of each output variable along axis.
/// - axis: A int, the dimension along which to split. Must be in the range [-rank(value), rank(value)). Defaults to 0
///
/// Returns:
/// - A list of variables.
List<VARP> split(VARP value, List<int> sizeSplits, {int axis = 0}) {
  final (sizeSplitsPtr, sizeSplitsSize) = sizeSplits.toNativeArrayI32();
  final p = C.mnn_expr_Split(value.ptr, sizeSplitsPtr.cast(), sizeSplitsSize, axis);
  final size = C.mnn_expr_VecVARP_size(p);
  calloc.free(sizeSplitsPtr);
  final rval = List.generate(size, (index) => VARP.fromPointer(C.mnn_expr_VecVARP_at(p, index)));
  // only free the struct, keep internal ref.ptr since they are attached
  // and will be managed by VARP object
  C.mnn_expr_VecVARP_free(p);
  return rval;
}

VARP slice(VARP x, VARP starts, VARP sizes) =>
    VARP.fromPointer(C.mnn_expr_Slice(x.ptr, starts.ptr, sizes.ptr));
VARP stridedSlice(
  VARP input,
  VARP begin,
  VARP end,
  VARP strided,
  int beginMask,
  int endMask,
  int ellipsisMask,
  int newAxisMask,
  int shrinkAxisMask,
) => VARP.fromPointer(
  C.mnn_expr_StridedSlice(
    input.ptr,
    begin.ptr,
    end.ptr,
    strided.ptr,
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask,
  ),
);
VARP stridedSliceWrite(
  VARP input,
  VARP begin,
  VARP end,
  VARP strided,
  VARP write,
  int beginMask,
  int endMask,
  int ellipsisMask,
  int newAxisMask,
  int shrinkAxisMask,
) => VARP.fromPointer(
  C.mnn_expr_StridedSliceWrite(
    input.ptr,
    begin.ptr,
    end.ptr,
    strided.ptr,
    write.ptr,
    beginMask,
    endMask,
    ellipsisMask,
    newAxisMask,
    shrinkAxisMask,
  ),
);

/// Concatenates variables along one dimension.
///
/// Args:
/// - values: A list of variables a single variable.
/// - axis: A int. Dimension along which to concatenate.
/// Must be in the range [-rank(values), rank(values)).
/// As in Python, indexing for axis is 0-based.
/// Positive axis in the rage of [0, rank(values)) refers to axis-th dimension.
/// And negative axis refers to axis + rank(values)-th dimension.
///
/// Returns:
/// - A variable resulting from concatenation of the input variables.
VARP concat(List<VARP> values, int axis) {
  final pVec = values.toNativeVec();
  final rval = VARP.fromPointer(C.mnn_expr_Concat(pVec.ptr, axis));
  pVec.dispose();
  return rval;
}

/// Convert a variable to another format(possibily added after `input`).
///
/// Args:
/// - input: A variable.
/// - format: The target format.
///
/// Returns:
/// - A variable. If `input` is already `format`, then return `input` directly, otherwize add a variable after `input` with `format`.
VARP convert(VARP input, DimensionFormat format) =>
    VARP.fromPointer(C.mnn_expr_Convert(input.ptr, format.value));

/// Transposes x.
///
/// Args:
/// - x: A variable.
/// - perm: A vector, indicating the permutation of the dimensions of x.
///
/// Returns:
/// - A transposed variable.
VARP transpose(VARP x, List<int> perm) {
  final (permPtr, permSize) = perm.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Transpose(x.ptr, permPtr.cast(), permSize));
  calloc.free(permPtr);
  return rval;
}

VARP transpose1(VARP x, VARP perm) {
  final rval = VARP.fromPointer(C.mnn_expr_Transpose_1(x.ptr, perm.ptr));
  return rval;
}

VARP channelShuffle(VARP x, int group) => VARP.fromPointer(C.mnn_expr_ChannelShuffle(x.ptr, group));

/// Convert a variable to another format(possibily added before `input`).
///
/// Args:
/// - input: A variable.
/// - format: The target format.
///
/// Returns:
/// - A variable. If `input` is already `format`, then return `input` directly, otherwize add a variable before `input` with `format`.
VARP changeInputFormat(VARP x, DimensionFormat format) =>
    VARP.fromPointer(C.mnn_expr_ChangeInputFormat(x.ptr, format.value));

VARP reverse(VARP x, VARP axis) => VARP.fromPointer(C.mnn_expr_Reverse(x.ptr, axis.ptr));
VARP reverseSequence(VARP x, VARP y, int batchDim, int seqDim) =>
    VARP.fromPointer(C.mnn_expr_ReverseSequence(x.ptr, y.ptr, batchDim, seqDim));

/// Crop images.
///
/// Args:
/// - images: 4-D variable of NC4HW4 format.
/// - size: A variable. It takes the shape of `size` as output cropped variable's shape  while omits the values/format of `size`.
/// - axis: A int indicating the dimention to crop. Must be >=2. All dimensions up to but excluding `axis` are preserved, while the dimensions including and trailing `axis` are cropped.
/// - offset: A vector of int indicating the offsets. length(`offset`) must be >=1 and <=2. If length(`offset`) is 1, then all dimensions are offset by this amount.Otherwise, the number of offsets must equal the number of cropped axes in each dimension accordingly.
///
/// Returns:
/// - The cropped 4-D variable of NC4HW4 format.
VARP crop(VARP images, VARP size, int axis, List<int> offset) {
  final (offsetPtr, offsetSize) = offset.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Crop(images.ptr, size.ptr, axis, offsetPtr.cast(), offsetSize));
  calloc.free(offsetPtr);
  return rval;
}

/// Resize images.
///
/// Args:
/// - images: 4-D variable of NC4HW4 format.
/// - xScale: A float.
/// - yScale: A float.
///
/// Returns:
/// - The resized 4-D variable of NC4HW4 format.
VARP resize(VARP images, double xScale, double yScale) =>
    VARP.fromPointer(C.mnn_expr_Resize(images.ptr, xScale, yScale));

/// Pads a variable.
///
/// Args:
/// - x: A variable.
/// - paddings: A variable of type Halide_Type_Int. The shape is [n, 2] where  n is the rank of variable.
/// - mode: A enum, One of PadValueMode_CONSTANT, PadValueMode_SYMMETRIC, or PadValueMode_REFLECT.
///
/// Returns:
/// - A variable. Has the same type as x.
VARP pad(VARP x, VARP paddings, {PadValueMode mode = PadValueMode.CONSTANT}) =>
    VARP.fromPointer(C.mnn_expr_Pad(x.ptr, paddings.ptr, mode.value));

/// Returns a variable with an additional dimension inserted at index axis.
///
/// Args:
/// - input: A variable.
/// - axis: A int, specifying the dimension index at which to expand the shape of input.
/// Given an input of D dimensions, axis must be in range [-(D+1), D] (inclusive).
///
/// Returns:
/// - A variable with the same data as input, with an additional dimension inserted at the index specified by axis.
VARP expandDims(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ExpandDims(x.ptr, axis));
VARP expandDims1(VARP x, VARP axis) => VARP.fromPointer(C.mnn_expr_ExpandDims_1(x.ptr, axis.ptr));

/// Returns the shape of a variable.
///
/// Args:
/// - input: A variable.
///
/// Returns:
/// - A variable of Halide_Type_Int.
VARP shape(VARP input, {bool nchw = false}) => VARP.fromPointer(C.mnn_expr_Shape(input.ptr, nchw));

/// Stacks a list of rank-R variables into one rank-(R+1) variable.
///
/// Packs the list of variables in `values` into a ariable with rank one higher than each variable in values,
/// by packing them along the axis dimension.
///
/// Given a list of length N of variables of shape (A, B, C);
/// if axis == 0 then the output variable will have the shape (N, A, B, C).
/// if axis == 1 then the output variable will have the shape (A, N, B, C). Etc.
///
/// Args:
/// - values: A list of variable objects with the same shape and type.
/// - axis: An int. The axis to stack along. Defaults to the first dimension. Negative values wrap around,
/// so the valid range is [-(R+1), R+1).
///
/// Returns:
/// - output: A stacked variable with the same type as `values`.
VARP stack(List<VARP> values, {int axis = 0}) {
  final pVec = values.toNativeVec();
  final rval = VARP.fromPointer(C.mnn_expr_Stack(pVec.ptr, axis));
  pVec.dispose();
  return rval;
}

/// Extracts crops from the input image variable and resizes them using bilinear sampling or nearest neighbor sampling (possibly with aspect ratio change)
/// to a common output size specified by crop_size.
///
/// Returns a variable with crops from the input image at positions defined at the bounding box locations in boxes.
///
/// The cropped boxes are all resized (with bilinear or nearest neighbor interpolation) to a fixed size = [crop_height, crop_width].
///
/// The result is a 4-D tensor [num_boxes, crop_height, crop_width, depth](supposing NHWC format).
///
/// Arguments:
/// - image: A 4-D variable of shape [batch, image_height, image_width, depth](supposing NHWC format). Both image_height and image_width need to be positive.
/// - boxes: A 2-D variable of shape [num_boxes, 4]. The i-th row of the variable specifies the coordinates of a box in the box_ind[i] image and is specified in normalized coordinates [y1, x1, y2, x2].
/// A normalized coordinate value of y is mapped to the image coordinate at y * (image_height - 1), so as the [0, 1] interval of normalized image height is mapped to [0, image_height - 1] in image height coordinates. We do allow y1 > y2, in which case the sampled crop is an up-down flipped version of the original image. The width dimension is treated similarly. Normalized coordinates outside the [0, 1] range are allowed, in which case we use extrapolation_value to extrapolate the input image values.
/// - box_ind: A 1-D variable of shape [num_boxes] with int values in [0, batch). The value of box_ind[i] specifies the image that the i-th box refers to.
/// - crop_size: A 1-D variable of 2 elements, size = [crop_height, crop_width]. All cropped image patches are resized to this size. The aspect ratio of the image content is not preserved. Both crop_height and crop_width need to be positive.
/// - method: A enum, either CropAndResizeMethod_NEAREST, or CropAndResizeMethod_BILINEAR, default to CropAndResizeMethod_BILINEAR.
/// - extrapolation_value: Value used for extrapolation, when applicable.
///
/// Returns:
/// - Output: A 4-D variable of shape [num_boxes, crop_height, crop_width, depth](supposing NHWC format).
VARP cropAndResize(
  VARP image,
  VARP boxes,
  VARP boxInd,
  VARP cropSize,
  InterpolationMethod method, {
  double extrapolationValue = 0.0,
}) => VARP.fromPointer(
  C.mnn_expr_CropAndResize(
    image.ptr,
    boxes.ptr,
    boxInd.ptr,
    cropSize.ptr,
    method.value,
    extrapolationValue,
  ),
);

/// Creates a variable filled with a scalar value.
///
/// Args:
/// - dims: A variable. Must be 1-D Halide_Type_Int. Represents the shape of the output variable.
/// - value: A variable. 0-D (scalar). Value to fill the returned variable.
///
/// Returns:
/// - A variable. Has the same type as value.
VARP fill(VARP dims, VARP value) => VARP.fromPointer(C.mnn_expr_Fill(dims.ptr, value.ptr));

/// Constructs a variable by tiling a given variable.
///
/// Args:
/// - input: A variable. 1-D or higher.
/// - multiples: A variable. Must be 1-D Halide_Type_Int.Length must be the same as the number of dimensions in input.
///
/// Returns:
/// - A variable. Has the same type as input.
VARP tile(VARP input, VARP multiples) => VARP.fromPointer(C.mnn_expr_Tile(input.ptr, multiples.ptr));

/// Gather slices from params according to indices.
///
/// Arguments:
/// - params: The variable from which to gather values.
/// - indices: Index variable. Must be Halide_Type_Int in range [0, ndims(params)-1].
///
/// Returns:
/// - Output: Values from params gathered from indices given by indices.
VARP gather(VARP input, VARP indices) => VARP.fromPointer(C.mnn_expr_Gather(input.ptr, indices.ptr));

/// Gather slices from params axis according to indices.
///
/// Arguments:
/// - params: The variable from which to gather values.
/// - indices: Index variable. Must be Halide_Type_Int in range [0, ndims(params)-1].
/// - axis: A int, the axis in params to gather indices from. Supports negative indexes.
/// If set to 0, it's same as _Gather. Currently only 0 is supported.
///
/// Returns:
/// - Output: Values from params gathered from indices given by indices.
VARP gatherV2(VARP input, VARP indices, VARP axis) =>
    VARP.fromPointer(C.mnn_expr_GatherV2(input.ptr, indices.ptr, axis.ptr));

/// Removes dimensions of size 1 from the shape of a variable.
///
/// Args:
/// - input: A variable. The input to squeeze.
/// - axis: A vector, Defaults to {}. If specified, only squeezes the dimensions listed. The dimension index starts at 0.
/// Must be in the range [-rank(input), rank(input)).
///
/// Returns:
/// - A variable. Has the same type as input. Contains the same data as input, but has one or more dimensions of size 1 removed.
VARP squeeze(VARP input, {List<int> axis = const []}) {
  final (axisPtr, axisSize) = axis.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Squeeze(input.ptr, axisPtr.cast(), axisSize));
  calloc.free(axisPtr);
  return rval;
}

/// BatchToSpace for N-D variables
///
/// This operation reshapes the "batch" dimension 0 into M + 1 dimensions of shape block_shape + [batch],
/// interleaves these blocks back into the grid defined by the spatial dimensions [1, ..., M],
/// to obtain a result with the same rank as the input.
///
/// The spatial dimensions of this intermediate result are then optionally cropped according to crops to
/// produce the output. This is the reverse of SpaceToBatch. See below for a precise description.
///
/// Arguments:
/// - input: must be 4-D with NC4HW4 format. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
/// - block_shape: 1-D with shape [M], all values must be >= 1.
/// - crops: 2-D with shape [M, 2], all values must be >= 0. crops[i] = [crop_start, crop_end] specifies the amount to crop from input dimension i + 1,
/// which corresponds to spatial dimension i. It is required that crop_start[i] + crop_end[i] <= block_shape[i] * input_shape[i + 1].
///
/// This operation is equivalent to the following steps:
/// 1. Reshape input to reshaped of shape: [block_shape[0], ..., block_shape[M-1], batch / prod(block_shape),
/// input_shape[1], ..., input_shape[N-1]]
/// 2. Permute dimensions of reshaped to produce permuted of shape
/// [batch / prod(block_shape),input_shape[1], block_shape[0], ..., input_shape[M], block_shape[M-1],input_shape[M+1], ..., input_shape[N-1]]
/// 3. Reshape permuted to produce reshaped_permuted of shape
/// [batch / prod(block_shape),input_shape[1] * block_shape[0], ..., input_shape[M] * block_shape[M-1],input_shape[M+1], ..., input_shape[N-1]]
/// 4. Crop the start and end of dimensions [1, ..., M] of reshaped_permuted according to crops to produce the output of shape:
/// [batch / prod(block_shape),input_shape[1] * block_shape[0] - crops[0,0] - crops[0,1], ..., input_shape[M] * block_shape[M-1] - crops[M-1,0] - crops[M-1,1],input_shape[M+1], ..., input_shape[N-1]]
///
/// Some examples:
/// for the following input of shape [4, 1, 1, 3], block_shape = [2, 2], and crops = [[0, 0], [0, 0]]:
/// ```
/// [[[[1, 2, 3]]], [[[4, 5, 6]]], [[[7, 8, 9]]], [[[10, 11, 12]]]]
/// ```
/// The output variable has shape [1, 2, 2, 3] and value:
/// ```
/// x = [[[[1, 2, 3], [4, 5, 6]],
///       [[7, 8, 9], [10, 11, 12]]]]
/// ```
///
/// Returns:
/// - Output: The output variable
VARP batchToSpaceND(VARP input, VARP blockShape, VARP crops) =>
    VARP.fromPointer(C.mnn_expr_BatchToSpaceND(input.ptr, crops.ptr, blockShape.ptr));

/// Gather slices from params into a variable with shape specified by indices.
///
/// Args:
/// - params: A variable. The variables from which to gather values.
/// - indices: A variable. Must be one of the following types: Halide_Type_Int.
///
/// Returns:
/// - A variable. Has the same type as params.
VARP gatherND(VARP params, VARP indices) => VARP.fromPointer(C.mnn_expr_GatherND(params.ptr, indices.ptr));
VARP gatherElements(VARP params, VARP indices, {VARP? axis}) => VARP.fromPointer(
  axis == null
      ? C.mnn_expr_GatherElements(params.ptr, indices.ptr)
      : C.mnn_expr_GatherElements_1(params.ptr, indices.ptr, axis.ptr),
);

/// Computes scaled exponential linear: scale * alpha * (exp(features) - 1) if < 0, scale * features otherwise.
///
/// Args:
/// - features: A variable of type Halide_Type_Float
/// - scale: Scaling factor (positive float)
/// - alpha: Alpha factor (positive float)
///
/// Returns:
/// - A variable. Has the same type as features.
VARP selu(VARP features, double scale, double alpha) =>
    VARP.fromPointer(C.mnn_expr_Selu(features.ptr, scale, alpha));

/// Computes the size of the variable
///
/// Args:
/// - input: A variable of type Halide_Type_Float or Halide_Type_Int
///
/// Returns:
/// - A variable. The shape is (), and type is Halide_Type_Int
VARP size(VARP input) => VARP.fromPointer(C.mnn_expr_Size(input.ptr));

/// Computes exponential linear: alpha * (exp(features) - 1) if < 0, features otherwise.
/// - features: A variable of type Halide_Type_Float
/// - alpha: Alpha factor (positive float)
///
/// Returns:
/// - A variable. Has the same type as features.
VARP elu(VARP features, {double alpha = 1.0}) => VARP.fromPointer(C.mnn_expr_Elu(features.ptr, alpha));

/// Given an input value x, it computes the output as 1.0 if x > threshold and 0.0 if x <= threshold.
/// - features: A variable of type Halide_Type_Float
/// - threshold: threshold value
///
/// Returns:
/// - A variable. Has the same type as features.
VARP threshold(VARP features, {double alpha = 1.0}) =>
    VARP.fromPointer(C.mnn_expr_Threshold(features.ptr, alpha));

/// Copies a variable setting everything outside a central band in each innermost matrix.
///
/// Arguments:
/// - input: Rank k variable.
/// - num_lower: Number of subdiagonals to keep. If negative, keep entire lower triangle.
/// - num_upper: Number of superdiagonals to keep. If negative, keep entire upper triangle.
///
/// Returns:
/// - Output: Rank k variable of the same shape as input. The extracted banded tensor.
VARP matrixBandPart(VARP input, VARP lower, VARP upper) =>
    VARP.fromPointer(C.mnn_expr_MatrixBandPart(input.ptr, lower.ptr, upper.ptr));

/// Calculates the mean and variance of x.
///
/// Args:
/// - x: A variable. must be 4-D with NC4HW4 format.
/// - axes: Array of ints. Axes along which to compute mean and variance. Ignored for this implementation: must be {2, 3}
/// - shift: Not used in the current implementation.
/// - keepdims: produce moments with the same dimensionality as the input.  Ignored for this implementation: must be true.
///
/// Returns:
/// - Two variable objects: mean and variance.
List<VARP> moments(VARP x, List<int> axis, VARP shift, bool keepDims) {
  final (axisPtr, axisSize) = axis.toNativeArrayI32();
  final vec = VecVARP.fromPointer(C.mnn_expr_Moments(x.ptr, axisPtr.cast(), axisSize, shift.ptr, keepDims));
  calloc.free(axisPtr);
  final rval = vec.toList();
  vec.dispose();
  return rval;
}

/// Computes the difference between two lists of numbers or strings.
///
/// Given a list x and a list y, this operation returns a list out that represents all values that are in x but not in y.
///
/// The returned list out is sorted in the same order that the numbers appear in x (duplicates are preserved).
///
/// This operation also returns a list idx that represents the position of each out element in x.
///
/// Arguments:
/// - x: 1-D variable of type Halide_Type_Int. Values to keep.
/// - y: 1-D variable of type Halide_Type_Int. Values to remove.
///
/// Returns:
/// - Output out: 1-D variable of type Halide_Type_Int. Values present in x but not in y.
VARP setDiff1D(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_SetDiff1D(x.ptr, y.ptr));

/// Rearranges blocks of spatial data, into depth.
///
/// More specifically, it outputs a copy of the input variable where values from the height and width dimensions are moved to the depth dimension.
///
/// The block_size indicates the input block size.
///
/// Non-overlapping blocks of size block_size x block_size are rearranged into depth at each location.
///
/// The depth of the output variable is block_size * block_size * input_depth.
///
/// The Y, X coordinates within each block of the input become the high order component of the output channel index.
///
/// The input variable's height and width must be divisible by block_size
///
/// Args:
/// - input: A variable.
/// - block_size: An int that is >= 2. The size of the spatial block.
///
/// Returns:
/// - A variable. Has the same type as input.
VARP spaceToDepth(VARP input, int blockSize) =>
    VARP.fromPointer(C.mnn_expr_SpaceToDepth(input.ptr, blockSize));

/// This operation divides "spatial" dimensions [1, ..., M] of the input into a grid of blocks of shape block_shape,
/// and interleaves these blocks with the "batch" dimension
/// such that in the output, the spatial dimensions [1, ..., M] correspond to the position within the grid,
/// and the batch dimension combines both the position within a spatial block and the original batch position.
///
/// Prior to division into blocks, the spatial dimensions of the input are optionally zero padded according to paddings.
///
/// See below for a precise description.
///
/// Args:
/// - input: A variable. must be 4-D with NC4HW4 format. N-D with shape input_shape = [batch] + spatial_shape + remaining_shape, where spatial_shape has M dimensions.
/// - block_shape: A variable. Must be one of the following types: int32, int64. 1-D with shape [M], all values must be >= 1.
/// - paddings: A variable. Must be one of the following types: int32, int64. 2-D with shape [M, 2], all values must be >= 0. paddings[i] = [pad_start, pad_end] specifies the padding for input dimension i + 1, which corresponds to spatial dimension i. It is required that block_shape[i] divides input_shape[i + 1] + pad_start + pad_end.
///
/// Returns:
/// - A variable. Has the same type as input.
VARP spaceToBatchND(VARP input, VARP blockShape, VARP paddings) =>
    VARP.fromPointer(C.mnn_expr_SpaceToBatchND(input.ptr, blockShape.ptr, paddings.ptr));

/// Creates a variable with all elements set to zero.
///
/// Args:
/// - input: A variable.
///
/// Returns:
/// - A variable with all elements set to zero.
VARP zerosLike(VARP input) => VARP.fromPointer(C.mnn_expr_ZerosLike(input.ptr));

/// Unpacks the given dimension of a rank-R tensor into rank-(R-1) variable.
///
/// For example, given a variable of shape (A, B, C, D);
///
/// If axis == 0 then the i'th variable in output is the slice value[i, :, :, :] and each variable in output will have shape (B, C, D).
/// (Note that the dimension unpacked along is gone, unlike split).
///
/// If axis == 1 then the i'th variable in output is the slice value[:, i, :, :] and each variable in output will have shape (A, C, D).
///
/// Args:
/// - value: A rank R > 0 variable to be unstacked.
/// - num: An int. The length of the dimension axis. Automatically inferred if None (the default).
/// - axis: An int. The axis to unstack along. Defaults to the first dimension. Negative values wrap around, so the valid range is [-R, R).
///
/// Returns:
/// - The list of variable objects unstacked from value.
List<VARP> unstack(VARP input, {int axis = 0}) {
  final vec = VecVARP.fromPointer(C.mnn_expr_Unstack(input.ptr, axis));
  final rval = vec.toList();
  vec.dispose();
  return rval;
}

/// Returns the rank of a variable.
///
/// Returns a 0-D int32 variable representing the rank of input.
///
/// Note: The rank of a variable is not the same as the rank of a matrix.
///
/// It's the number of indices required to uniquely select each element of the variable.
///
/// It's also known as "order", "degree", or "ndims."
///
/// Args:
/// - input: A variable.
///
/// Returns:
/// - A 0-D variable of type Halide_Type_Int
VARP rank(VARP input) => VARP.fromPointer(C.mnn_expr_Rank(input.ptr));

/// Creates a sequence of numbers.
///
/// Args:
/// - start: A 0-D variable (scalar).
/// - limit: A 0-D variable (scalar).
/// - delta: A 0-D variable (scalar).
VARP range(VARP start, VARP limit, VARP delta) =>
    VARP.fromPointer(C.mnn_expr_Range(start.ptr, limit.ptr, delta.ptr));

/// Rearranges data from depth into blocks of spatial data.
///
/// It is the reverse transformation of SpaceToDepth. More specifically,
/// it outputs a copy of the input variable where values from the depth dimension are moved in spatial blocks to the height and width dimensions.
///
/// Args:
/// - input: A variable.
/// - block_size: An int that is >= 2. The size of the spatial block, same as in Space2Depth.
///
/// Returns:
/// - A variable. Has the same type as input.
VARP depthToSpace(VARP input, int blockSize) =>
    VARP.fromPointer(C.mnn_expr_DepthToSpace(input.ptr, blockSize));

/// SSD network's permute layer.
///
/// Args:
/// - input: A variable. Contains the feature map. Namely bottom[0] in caffe.
/// - dims:  A vector. Contains the order.
///
/// Returns:
/// - A variable.
VARP Permute(VARP input, List<int> dims) {
  final (permPtr, permSize) = dims.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Permute(input.ptr, permPtr.cast(), permSize));
  calloc.free(permPtr);
  return rval;
}

VARP interp(
  List<VARP> xs,
  double widthScale,
  double heightScale,
  int outputWidth,
  int outputHeight,
  int resizeType,
  bool alignCorners,
) {
  final xsPtr = xs.toNativeVec();
  final rval = VARP.fromPointer(
    C.mnn_expr_Interp(
      xsPtr.ptr,
      widthScale,
      heightScale,
      outputWidth,
      outputHeight,
      resizeType,
      alignCorners,
    ),
  );
  xsPtr.dispose();
  return rval;
}

VARP zeroGrad(VARP input) => VARP.fromPointer(C.mnn_expr_ZeroGrad(input.ptr));

VARP CosineSimilarity(VARP input0, VARP input1, VARP inputDim) =>
    VARP.fromPointer(C.mnn_expr_CosineSimilarity(input0.ptr, input1.ptr, inputDim.ptr));

VARP GridSample(
  VARP input,
  VARP grid, {
  InterpolationMethod mode = InterpolationMethod.BILINEAR,
  GridSamplePaddingMode paddingMode = GridSamplePaddingMode.GRID_SAMPLE_PADDING_ZEROS,
  bool alignCorners = false,
}) =>
    VARP.fromPointer(C.mnn_expr_GridSample(input.ptr, grid.ptr, mode.value, paddingMode.value, alignCorners));

VARP floatToInt8(VARP x, VARP scale, int minvalue, int maxValue, {int? zeroPoint}) => VARP.fromPointer(
  zeroPoint == null
      ? C.mnn_expr_FloatToInt8(x.ptr, scale.ptr, minvalue, maxValue)
      : C.mnn_expr_FloatToInt8_1(x.ptr, scale.ptr, minvalue, maxValue, zeroPoint),
);

VARP int8ToFloat(VARP x, VARP scale, {int? zeroPoint}) => VARP.fromPointer(
  zeroPoint == null
      ? C.mnn_expr_Int8ToFloat(x.ptr, scale.ptr)
      : C.mnn_expr_Int8ToFloat_1(x.ptr, scale.ptr, zeroPoint),
);

VARP select(VARP select, VARP input0, VARP input1) =>
    VARP.fromPointer(C.mnn_expr_Select(select.ptr, input0.ptr, input1.ptr));

VARP where(VARP x) => VARP.fromPointer(C.mnn_expr_Where(x.ptr));

VARP sort(VARP x, {int axis = -1, bool arg = false, bool descend = false}) =>
    VARP.fromPointer(C.mnn_expr_Sort(x.ptr, axis, arg, descend));
