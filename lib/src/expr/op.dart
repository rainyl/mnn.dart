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
  SAME(2);

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
  AVEPOOL(1);

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
  EDGE(3);

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
  NEAREST(1);

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
  GRID_SAMPLE_PADDING_REFLECTION(2);

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
VARP add(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Add(x.ptr, y.ptr));
VARP subtract(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Subtract(x.ptr, y.ptr));
VARP multiply(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Multiply(x.ptr, y.ptr));
VARP divide(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Divide(x.ptr, y.ptr));
VARP pow(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Pow(x.ptr, y.ptr));
VARP minimum(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Minimum(x.ptr, y.ptr));
VARP maximum(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Maximum(x.ptr, y.ptr));
VARP biasAdd(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BiasAdd(x.ptr, y.ptr));
VARP greater(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Greater(x.ptr, y.ptr));
VARP greaterEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_GreaterEqual(x.ptr, y.ptr));
VARP less(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Less(x.ptr, y.ptr));
VARP floorDiv(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_FloorDiv(x.ptr, y.ptr));
VARP squaredDifference(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_SquaredDifference(x.ptr, y.ptr));
VARP equal(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Equal(x.ptr, y.ptr));
VARP lessEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_LessEqual(x.ptr, y.ptr));
VARP floorMod(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_FloorMod(x.ptr, y.ptr));
VARP atan2(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Atan2(x.ptr, y.ptr));
VARP logicalOr(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_LogicalOr(x.ptr, y.ptr));
VARP notEqual(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_NotEqual(x.ptr, y.ptr));
VARP bitwiseAnd(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseAnd(x.ptr, y.ptr));
VARP bitwiseOr(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseOr(x.ptr, y.ptr));
VARP bitwiseXor(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_BitwiseXor(x.ptr, y.ptr));

// UnaryOPs
VARP sign(VARP x) => VARP.fromPointer(C.mnn_expr_Sign(x.ptr));
VARP abs(VARP x) => VARP.fromPointer(C.mnn_expr_Abs(x.ptr));
VARP negative(VARP x) => VARP.fromPointer(C.mnn_expr_Negative(x.ptr));
VARP floor(VARP x) => VARP.fromPointer(C.mnn_expr_Floor(x.ptr));
VARP round(VARP x) => VARP.fromPointer(C.mnn_expr_Round(x.ptr));
VARP ceil(VARP x) => VARP.fromPointer(C.mnn_expr_Ceil(x.ptr));
VARP square(VARP x) => VARP.fromPointer(C.mnn_expr_Square(x.ptr));
VARP sqrt(VARP x) => VARP.fromPointer(C.mnn_expr_Sqrt(x.ptr));
VARP rsqrt(VARP x) => VARP.fromPointer(C.mnn_expr_Rsqrt(x.ptr));
VARP exp(VARP x) => VARP.fromPointer(C.mnn_expr_Exp(x.ptr));
VARP log(VARP x) => VARP.fromPointer(C.mnn_expr_Log(x.ptr));
VARP sin(VARP x) => VARP.fromPointer(C.mnn_expr_Sin(x.ptr));
VARP sinh(VARP x) => VARP.fromPointer(C.mnn_expr_Sinh(x.ptr));
VARP cos(VARP x) => VARP.fromPointer(C.mnn_expr_Cos(x.ptr));
VARP cosh(VARP x) => VARP.fromPointer(C.mnn_expr_Cosh(x.ptr));
VARP tan(VARP x) => VARP.fromPointer(C.mnn_expr_Tan(x.ptr));
VARP tanh(VARP x) => VARP.fromPointer(C.mnn_expr_Tanh(x.ptr));
VARP asin(VARP x) => VARP.fromPointer(C.mnn_expr_Asin(x.ptr));
VARP asinh(VARP x) => VARP.fromPointer(C.mnn_expr_Asinh(x.ptr));
VARP acos(VARP x) => VARP.fromPointer(C.mnn_expr_Acos(x.ptr));
VARP acosh(VARP x) => VARP.fromPointer(C.mnn_expr_Acosh(x.ptr));
VARP atan(VARP x) => VARP.fromPointer(C.mnn_expr_Atan(x.ptr));
VARP atanh(VARP x) => VARP.fromPointer(C.mnn_expr_Atanh(x.ptr));
VARP reciprocal(VARP x) => VARP.fromPointer(C.mnn_expr_Reciprocal(x.ptr));
VARP log1p(VARP x) => VARP.fromPointer(C.mnn_expr_Log1p(x.ptr));
VARP gelu(VARP x) => VARP.fromPointer(C.mnn_expr_Gelu(x.ptr));
VARP sigmoid(VARP x) => VARP.fromPointer(C.mnn_expr_Sigmoid(x.ptr));
VARP erf(VARP x) => VARP.fromPointer(C.mnn_expr_Erf(x.ptr));
VARP erfc(VARP x) => VARP.fromPointer(C.mnn_expr_Erfc(x.ptr));
VARP erfinv(VARP x) => VARP.fromPointer(C.mnn_expr_Erfinv(x.ptr));
VARP expm1(VARP x) => VARP.fromPointer(C.mnn_expr_Expm1(x.ptr));
VARP hardswish(VARP x) => VARP.fromPointer(C.mnn_expr_Hardswish(x.ptr));
VARP silu(VARP x) => VARP.fromPointer(C.mnn_expr_Silu(x.ptr));

// ReduceOPs
VARP reduceSum(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceSum(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMean(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMean(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMax(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMax(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMin(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMin(x.ptr, axis.i32.ptr, keepDims));
VARP reduceProd(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceProd(x.ptr, axis.i32.ptr, keepDims));
VARP reduceAny(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAny(x.ptr, axis.i32.ptr, keepDims));
VARP reduceAll(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAll(x.ptr, axis.i32.ptr, keepDims));

VARP reduceSumMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceSumMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMeanMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMeanMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMaxMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMaxMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceMinMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceMinMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceProdMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceProdMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceAnyMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAnyMutable(x.ptr, axis.i32.ptr, keepDims));
VARP reduceAllMutable(VARP x, List<int> axis, {bool keepDims = false}) =>
    VARP.fromPointer(C.mnn_expr_ReduceAllMutable(x.ptr, axis.i32.ptr, keepDims));

// EltwiseOPs
VARP prod(VARP x, VARP y, List<double> coeff) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Prod(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

VARP sum(VARP x, VARP y, List<double> coeff) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Sum(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

VARP max(VARP x, VARP y, List<double> coeff) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Max(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

VARP sub(VARP x, VARP y, List<double> coeff) {
  final (p, size) = coeff.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Sub(x.ptr, y.ptr, p, size));
  calloc.free(p);
  return rval;
}

VARP mod(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_Mod(x.ptr, y.ptr));

// OtherOPs
VARP cast<T extends ffi.SizedNativeType>(VARP x) =>
    VARP.fromPointer(C.mnn_expr_Cast(x.ptr, HalideType.of<T>().native.ref));
VARP matMul(VARP x, VARP y, {bool transposeA = false, bool transposeB = false}) =>
    VARP.fromPointer(C.mnn_expr_MatMul(x.ptr, y.ptr, transposeA, transposeB));
VARP normalize(VARP x, int acrossSpatial, int channelShared, double eps, List<double> scale) {
  final (p, size) = scale.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_Normalize(x.ptr, acrossSpatial, channelShared, eps, p, size));
  calloc.free(p);
  return rval;
}

VARP argMax(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ArgMax(x.ptr, axis));
VARP argMin(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ArgMin(x.ptr, axis));
VARP batchMatMul(VARP x, VARP y, {bool adjX = false, bool adjY = false}) =>
    VARP.fromPointer(C.mnn_expr_BatchMatMul(x.ptr, y.ptr, adjX, adjY));
VARP unravelIndex(VARP indices, VARP dims) =>
    VARP.fromPointer(C.mnn_expr_UnravelIndex(indices.ptr, dims.ptr));
VARP scatterND(VARP indices, VARP updates, VARP shape, {VARP? input, int? reduction}) =>
    switch ((input, reduction)) {
      (null, null) => VARP.fromPointer(C.mnn_expr_ScatterNd(indices.ptr, updates.ptr, shape.ptr)),
      (VARP(), null) =>
        VARP.fromPointer(C.mnn_expr_ScatterNd_1(indices.ptr, updates.ptr, shape.ptr, input!.ptr)),
      (null, int()) =>
        VARP.fromPointer(C.mnn_expr_ScatterNd_2(indices.ptr, updates.ptr, shape.ptr, reduction!)),
      (VARP(), int()) =>
        VARP.fromPointer(C.mnn_expr_ScatterNd_3(indices.ptr, updates.ptr, shape.ptr, input!.ptr, reduction!)),
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
}) =>
    VARP.fromPointer(C.mnn_expr_RandomUniform(shape.ptr, dtype.native.ref, low, high, seed0, seed1));
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
}) =>
    conv(weight, bias, x, pad: pad, stride: stride, dilatie: dilatie, group: group, pads: pads);

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
}) =>
    deconv(weight, bias, x, pad: pad, stride: stride, dilatie: dilatie, group: group, pads: pads);

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

VARP ReLU(VARP x, {double slope = 0.0}) => VARP.fromPointer(C.mnn_expr_Relu(x.ptr, slope));
VARP ReLU6(VARP x, {double slope = 0.0, double maxValue = 6.0}) =>
    VARP.fromPointer(C.mnn_expr_Relu6(x.ptr, slope, maxValue));
VARP PReLU(VARP x, List<double> slopes) {
  final (slopesPtr, slopesSize) = slopes.toNativeArrayF32();
  final rval = VARP.fromPointer(C.mnn_expr_PRelu(x.ptr, slopesPtr.cast(), slopesSize));
  calloc.free(slopesPtr);
  return rval;
}

VARP softMax(VARP logits, {int axis = -1}) => VARP.fromPointer(C.mnn_expr_Softmax(logits.ptr, axis));
VARP softPlus(VARP features) => VARP.fromPointer(C.mnn_expr_Softplus(features.ptr));
VARP softSign(VARP features) => VARP.fromPointer(C.mnn_expr_Softsign(features.ptr));
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
) =>
    VARP.fromPointer(
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
) =>
    VARP.fromPointer(
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

VARP concat(List<VARP> values, int axis) {
  final pVec = values.toNativeVec();
  final rval = VARP.fromPointer(C.mnn_expr_Concat(pVec.ptr, axis));
  pVec.dispose();
  return rval;
}

VARP convert(VARP input, DimensionFormat format) =>
    VARP.fromPointer(C.mnn_expr_Convert(input.ptr, format.value));

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
VARP changeInputFormat(VARP x, DimensionFormat format) =>
    VARP.fromPointer(C.mnn_expr_ChangeInputFormat(x.ptr, format.value));

VARP reverse(VARP x, VARP axis) => VARP.fromPointer(C.mnn_expr_Reverse(x.ptr, axis.ptr));
VARP reverseSequence(VARP x, VARP y, int batchDim, int seqDim) =>
    VARP.fromPointer(C.mnn_expr_ReverseSequence(x.ptr, y.ptr, batchDim, seqDim));
VARP crop(VARP images, VARP size, int axis, List<int> offset) {
  final (offsetPtr, offsetSize) = offset.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Crop(images.ptr, size.ptr, axis, offsetPtr.cast(), offsetSize));
  calloc.free(offsetPtr);
  return rval;
}

VARP resize(VARP images, double xScale, double yScale) =>
    VARP.fromPointer(C.mnn_expr_Resize(images.ptr, xScale, yScale));

VARP pad(VARP x, VARP paddings, {PadValueMode mode = PadValueMode.CONSTANT}) =>
    VARP.fromPointer(C.mnn_expr_Pad(x.ptr, paddings.ptr, mode.value));

VARP expandDims(VARP x, int axis) => VARP.fromPointer(C.mnn_expr_ExpandDims(x.ptr, axis));
VARP expandDims1(VARP x, VARP axis) => VARP.fromPointer(C.mnn_expr_ExpandDims_1(x.ptr, axis.ptr));

VARP shape(VARP input, {bool nchw = false}) => VARP.fromPointer(C.mnn_expr_Shape(input.ptr, nchw));

VARP stack(List<VARP> values, {int axis = 0}) {
  final pVec = values.toNativeVec();
  final rval = VARP.fromPointer(C.mnn_expr_Stack(pVec.ptr, axis));
  pVec.dispose();
  return rval;
}

VARP cropAndResize(
  VARP image,
  VARP boxes,
  VARP boxInd,
  VARP cropSize,
  InterpolationMethod method, {
  double extrapolationValue = 0.0,
}) =>
    VARP.fromPointer(
      C.mnn_expr_CropAndResize(
        image.ptr,
        boxes.ptr,
        boxInd.ptr,
        cropSize.ptr,
        method.value,
        extrapolationValue,
      ),
    );

VARP fill(VARP dims, VARP value) => VARP.fromPointer(C.mnn_expr_Fill(dims.ptr, value.ptr));
VARP tile(VARP input, VARP multiples) => VARP.fromPointer(C.mnn_expr_Tile(input.ptr, multiples.ptr));
VARP gather(VARP input, VARP indices) => VARP.fromPointer(C.mnn_expr_Gather(input.ptr, indices.ptr));
VARP squeeze(VARP input, {List<int> axis = const []}) {
  final (axisPtr, axisSize) = axis.toNativeArrayI32();
  final rval = VARP.fromPointer(C.mnn_expr_Squeeze(input.ptr, axisPtr.cast(), axisSize));
  calloc.free(axisPtr);
  return rval;
}

VARP batchToSpaceND(VARP input, VARP blockShape, VARP crops) =>
    VARP.fromPointer(C.mnn_expr_BatchToSpaceND(input.ptr, crops.ptr, blockShape.ptr));
VARP gatherND(VARP params, VARP indices) => VARP.fromPointer(C.mnn_expr_GatherND(params.ptr, indices.ptr));
VARP gatherElements(VARP params, VARP indices, {VARP? axis}) => VARP.fromPointer(
      axis == null
          ? C.mnn_expr_GatherElements(params.ptr, indices.ptr)
          : C.mnn_expr_GatherElements_1(params.ptr, indices.ptr, axis.ptr),
    );
VARP selu(VARP features, double scale, double alpha) =>
    VARP.fromPointer(C.mnn_expr_Selu(features.ptr, scale, alpha));
VARP size(VARP input) => VARP.fromPointer(C.mnn_expr_Size(input.ptr));
VARP elu(VARP features, {double alpha = 1.0}) => VARP.fromPointer(C.mnn_expr_Elu(features.ptr, alpha));
VARP threshold(VARP features, {double alpha = 1.0}) =>
    VARP.fromPointer(C.mnn_expr_Threshold(features.ptr, alpha));
VARP matrixBandPart(VARP input, VARP lower, VARP upper) =>
    VARP.fromPointer(C.mnn_expr_MatrixBandPart(input.ptr, lower.ptr, upper.ptr));
List<VARP> moments(VARP x, List<int> axis, VARP shift, bool keepDims) {
  final (axisPtr, axisSize) = axis.toNativeArrayI32();
  final vec = VecVARP.fromPointer(C.mnn_expr_Moments(x.ptr, axisPtr.cast(), axisSize, shift.ptr, keepDims));
  calloc.free(axisPtr);
  final rval = vec.toList();
  vec.dispose();
  return rval;
}

VARP setDiff1D(VARP x, VARP y) => VARP.fromPointer(C.mnn_expr_SetDiff1D(x.ptr, y.ptr));
VARP spaceToDepth(VARP input, int blockSize) =>
    VARP.fromPointer(C.mnn_expr_SpaceToDepth(input.ptr, blockSize));
VARP spaceToBatchND(VARP input, VARP blockShape, VARP paddings) =>
    VARP.fromPointer(C.mnn_expr_SpaceToBatchND(input.ptr, blockShape.ptr, paddings.ptr));
VARP zerosLike(VARP input) => VARP.fromPointer(C.mnn_expr_ZerosLike(input.ptr));
List<VARP> unstack(VARP input, {int axis = 0}) {
  final vec = VecVARP.fromPointer(C.mnn_expr_Unstack(input.ptr, axis));
  final rval = vec.toList();
  vec.dispose();
  return rval;
}

VARP rank(VARP input) => VARP.fromPointer(C.mnn_expr_Rank(input.ptr));
VARP range(VARP start, VARP limit, VARP delta) =>
    VARP.fromPointer(C.mnn_expr_Range(start.ptr, limit.ptr, delta.ptr));
VARP depthToSpace(VARP input, int blockSize) =>
    VARP.fromPointer(C.mnn_expr_DepthToSpace(input.ptr, blockSize));
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
