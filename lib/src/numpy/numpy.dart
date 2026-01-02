import 'dart:ffi' as ffi;
import 'dart:math' as math;

import '../core/base.dart';
import '../core/halide_runtime.dart';
import '../expr/expr.dart';
import '../expr/op.dart' as F;

void _orderAssert(String order) {
  if (order.isEmpty || !["C", "K"].contains(order)) {
    throw ArgumentError.value(order, 'order', 'Invalid order value');
  }
}

(HalideType dtype, List<int> shape) _arrayLikeType(
  VARP prototype, {
  HalideType? dtype,
  String? order,
  List<int>? shape,
}) {
  if (prototype.dtype == null || prototype.shape == null) {
    throw ArgumentError.value(prototype, 'prototype', 'prototype dtype or shape is null');
  }
  final dstDtype = dtype ?? prototype.dtype!;
  final dstShape = (order != "K" && shape != null) ? shape : prototype.shape!;
  return (dstDtype, dstShape);
}

List<int> _getShape(Iterable x) {
  final shape = <int>[];
  dynamic current = x;
  while (current is Iterable) {
    shape.add(current.length);
    if (current.isEmpty) break;
    current = current.first;
  }
  return shape;
}

Iterable<num> _flatten(Iterable x) sync* {
  for (final e in x) {
    if (e is Iterable) {
      yield* _flatten(e);
    } else if (e is num) {
      yield e;
    } else {
      throw ArgumentError.value(e, 'element', 'Invalid element type in Iterable');
    }
  }
}

VARP _toVARP<T extends ffi.SizedNativeType>(dynamic x) {
  if (x is VARP) {
    return x;
  }
  if (x is num) {
    return F.scalar<T>(x);
  }
  if (x is Iterable) {
    final shape = _getShape(x);
    final flat = _flatten(x).toList(growable: false);
    return F.constant<T>(flat, shape, format: DimensionFormat.NCHW);
  }
  throw ArgumentError.value(x, 'x', 'Invalid x value');
}

int _normalAxis(int axis, int ndim) {
  final axis_ = axis < 0 ? axis + ndim : axis;
  MnnAssert(
    axis_ >= 0 && axis_ < ndim,
    "AxisError: axes_arg: axis $axis_ is out of bounds for array of dimension $ndim",
  );
  return axis_;
}

List<int> _normalAxes(List<int> axes, int ndim) {
  return axes.map((e) => _normalAxis(e, ndim)).toList();
}

bool _canBroadcast(List<int> shape1, List<int> shape2) {
  if (shape1.length > shape2.length) return false;
  for (int i = 0; i < shape2.length - shape1.length; i++) {
    shape1.insert(i, 1);
  }
  for (int i = 0; i < shape2.length; i++) {
    if (shape2[i] % shape1[i] != 0) return false;
  }
  return true;
}

VARP scalar<T extends ffi.SizedNativeType>(num value, {HalideType? dtype}) {
  return F.scalar<T>(value, dtype: dtype);
}

/// empty(shape, dtype=float32)
/// Return a new var of given shape and type, without initializing entries.
///
/// Parameters
/// ----------
/// shape : int or tuple of int
///     Shape of the empty var, e.g., (2, 3) or 2.
/// dtype : data-type, optional
///     Desired output data-type for the array, e.g, np.int8.
///     Default is np.float32.
/// order : {'C', 'F', 'A', or 'K'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     Var of uninitialized (arbitrary) data of the given shape,
///     dtype, and order. Object arrays will be initialized to None.
///
/// Example:
/// -------
/// >>> np.empty([2, 2])
VARP empty<T extends ffi.SizedNativeType>(List<int> shape, {String order = "C"}) {
  MnnAssert(T != ffi.SizedNativeType, "You must specify the generic type T. e.g., mnn.float32");
  _orderAssert(order);
  return F.input<T>(shape, dataFormat: DimensionFormat.NCHW);
}

/// empty_like(prototype, dtype=None, order='K', subok=True, shape=None)
/// Return a new var with the same shape and type as a given var.
///
/// Parameters
/// ----------
/// prototype : var_like
///     The shape and data-type of prototype define these same
///     attributes of the returned array.
/// dtype : data-type, optional
///     Overrides the data type of the result.
/// order : {'C', 'F', 'A', or 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional.
///     Compatible with numpy.
/// shape : int or sequence of ints, optional.
///     Overrides the shape of the result.
///
/// Returns
/// -------
/// out : var
///     Var of uninitialized (arbitrary) data with the same shape
///     and type as prototype.
///
/// Example:
/// -------
/// >>> a = ([1,2,3], [4,5,6])
/// >>> np.empty_like(a)
VARP emptyLike<T extends ffi.SizedNativeType>(
  VARP prototype, {
  String order = "K",
  bool subOk = true,
  List<int>? shape,
}) {
  MnnAssert(T != ffi.SizedNativeType, "You must specify the generic type T. e.g., mnn.float32");
  _orderAssert(order);
  final (dstDtype, dstShape) = _arrayLikeType(
    prototype,
    dtype: HalideType.of<T>(),
    order: order,
    shape: shape,
  );
  return F.input<T>(dstShape, dataFormat: prototype.dataFormat ?? DimensionFormat.NCHW);
}

/// eye(N, M=None, k=0, dtype=float32, order='C')
/// Return a 2-D var with ones on the diagonal and zeros elsewhere.
///
/// Parameters
/// ----------
/// N : int
///     Number of rows in the output.
/// M : int, optional
///     Number of columns in the output. If None, defaults to `N`.
/// k : int, optional
///     Index of the diagonal: 0 (the default) refers to the main diagonal,
///     a positive value refers to an upper diagonal, and a negative value
///     to a lower diagonal.
/// dtype : data-type, optional
///     Data-type of the returned array.
/// order : {'C', 'F'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// I : var of shape (N,M)
///   An var where all elements are equal to zero, except for the `k`-th
///   diagonal, whose values are equal to one.
///
/// Examples
/// --------
/// >>> np.eye(2, dtype=int)
/// var([[1, 0],
///       [0, 1]])
/// >>> np.eye(3, k=1)
/// var([[0.,  1.,  0.],
///       [0.,  0.,  1.],
///       [0.,  0.,  0.]])
VARP eye<T extends ffi.SizedNativeType>(int N, {int? M, int k = 0, String order = "C"}) {
  MnnAssert(T != ffi.SizedNativeType, "You must specify the generic type T. e.g., mnn.float32");
  _orderAssert(order);
  M ??= N;
  final x = F.oneHot(
    arange<int32>(start: (0 + k).toDouble(), stop: (N + k).toDouble(), step: 1),
    F.scalar<int32>(M),
  );
  return F.cast<T>(x);
}

/// identity(n, dtype=float32)
/// Return the identity var. The identity var is a
/// square array with ones on the main diagonal.
///
/// Parameters
/// ----------
/// n : int
///     Number of rows (and columns) in `n` x `n` output.
/// dtype : data-type, optional
///     Data-type of the output.  Defaults to ``float``.
///
/// Returns
/// -------
/// out : var
///     `n` x `n` array with its main diagonal set to one,
///     and all other elements 0.
///
/// Examples
/// --------
/// >>> np.identity(3)
/// var([[1.,  0.,  0.],
///      [0.,  1.,  0.],
///      [0.,  0.,  1.]])
VARP identity<T extends ffi.SizedNativeType>(int n) {
  return eye<T>(n);
}

/// full(shape, fill_value, dtype=None, order='C')
/// Return a new var of given shape and type, filled with `fill_value`.
///
/// Parameters
/// ----------
/// shape : int or sequence of ints
///     Shape of the new var, e.g., ``(2, 3)`` or ``2``.
/// fill_value : scalar or var_like
///     Fill value.
/// dtype : data-type, optional
///     The desired data-type for the var  The default, None, means
///       ``np.array(fill_value).dtype``.
/// order : {'C', 'F'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     Var of `fill_value` with the given shape, dtype, and order.
///
/// Examples
/// --------
/// >>> np.full((2, 2), 10)
/// var([[10, 10],
///       [10, 10]])
VARP full<T extends ffi.SizedNativeType>(
  List<int> shape,
  dynamic fillValue, {
  String order = "C",
}) {
  _orderAssert(order);
  return F.fill(_toVARP<int32>(shape), _toVARP<T>(fillValue));
}

/// full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None)
/// Return a full var with the same shape and type as a given var.
///
/// Parameters
/// ----------
/// a : var_like
///     The shape and data-type of `a` define these same attributes of
///     the returned var.
/// fill_value : scalar
///     Fill value.
/// dtype : data-type, optional
///     Overrides the data type of the result.
/// order : {'C', 'F', 'A', or 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional.
///     Compatible with numpy.
/// shape : int or sequence of ints, optional.
///     Overrides the shape of the result.
///
/// Returns
/// -------
/// out : var
///     Var of `fill_value` with the same shape and type as `a`.
///
/// Examples
/// --------
/// >>> x = np.arange(6, dtype=np.int32)
/// >>> np.full_like(x, 1)
/// var([1, 1, 1, 1, 1, 1])
/// >>> np.full_like(x, 0.1)
/// var([0, 0, 0, 0, 0, 0])
/// >>> np.full_like(x, 0.1, dtype=np.float32)
/// var([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
/// >>> y = np.arange(6, dtype=np.float32)
/// >>> np.full_like(y, 0.1)
/// var([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
VARP fullLike<T extends ffi.SizedNativeType>(
  VARP a,
  num fillValue, {
  String order = "K",
  List<int>? shape,
}) {
  final (dstDtype, dstShape) = _arrayLikeType(a, order: order, shape: shape);
  return full<T>(dstShape, F.scalar<T>(fillValue), order: order);
}

/// ones(shape, dtype=None, order='C')
/// Return a new array of given shape and type, filled with ones.
///
/// Parameters
/// ----------
/// shape : int or sequence of ints
///     Shape of the new array, e.g., ``(2, 3)`` or ``2``.
/// dtype : data-type, optional
///     The desired data-type for the array, e.g., `np.int8`.  Default is
///     `np.float32`.
/// order : {'C', 'F'}, optional, default: C
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : Var
///     Var of ones with the given shape, dtype, and order.
///
/// Examples
/// --------
/// >>> np.ones(5)
/// var([1., 1., 1., 1., 1.])
/// >>> np.ones((5,), dtype=int)
/// var([1, 1, 1, 1, 1])
/// >>> np.ones((2, 1))
/// var([[1.],
///       [1.]])
/// >>> s = (2,2)
/// >>> np.ones(s)
/// var([[1.,  1.],
///       [1.,  1.]])
VARP ones<T extends ffi.SizedNativeType>(
  List<int> shape, {
  String order = "C",
}) {
  _orderAssert(order);
  return full<T>(shape, F.scalar<T>(1), order: order);
}

/// ones_like(a, dtype=None, order='K', subok=True, shape=None)
/// Return an array of ones with the same shape and type as a given array.
///
/// Parameters
/// ----------
/// a : var_like
///     The shape and data-type of `a` define these same attributes of
///     the returned array.
/// dtype : data-type, optional
///     Overrides the data type of the result.
/// order : {'C', 'F', 'A', or 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional.
///     Compatible with numpy.
/// shape : int or sequence of ints, optional.
///     Overrides the shape of the result.
///
/// Returns
/// -------
/// out : Var
///     Var of ones with the same shape and type as `a`.
///
/// Examples
/// --------
/// >>> x = np.arange(6)
/// >>> x = x.reshape((2, 3))
/// >>> x
/// var([[0, 1, 2],
///       [3, 4, 5]])
/// >>> np.ones_like(x)
/// var([[1, 1, 1],
///       [1, 1, 1]])
/// >>> y = np.arange(3, dtype=float)
/// >>> y
/// var([0., 1., 2.])
/// >>> np.ones_like(y)
/// var([1.,  1.,  1.])
VARP onesLike<T extends ffi.SizedNativeType>(
  VARP a, {
  String order = "K",
  List<int>? shape,
}) {
  return fullLike<T>(a, 1, order: order, shape: shape);
}

/// zeros(shape, dtype=None, order='C')
/// Return a new array of given shape and type, filled with zeros.
///
/// Parameters
/// ----------
/// shape : int or sequence of ints
///     Shape of the new array, e.g., ``(2, 3)`` or ``2``.
/// dtype : data-type, optional
///     The desired data-type for the array, e.g., `np.int8`.  Default is
///     `np.float32`.
/// order : {'C', 'F'}, optional, default: C
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : Var
///     Var of zeros with the given shape, dtype, and order.
///
/// Examples
/// --------
/// >>> np.zeros(5)
/// var([ 0.,  0.,  0.,  0.,  0.])
/// >>> np.ones((5,), dtype=int)
/// var([0, 0, 0, 0, 0])
/// >>> np.ones((2, 1))
/// var([[ 0.],
///       [ 0.]])
/// >>> s = (2,2)
/// >>> np.ones(s)
/// var([[ 0.,  0.],
///       [ 0.,  0.]])
VARP zeros<T extends ffi.SizedNativeType>(
  List<int> shape, {
  String order = "C",
}) {
  _orderAssert(order);
  return full<T>(shape, F.scalar<T>(0), order: order);
}

/// zeros_like(a, dtype=None, order='K', subok=True, shape=None)
/// Return an array of zeros with the same shape and type as a given array.
///
/// Parameters
/// ----------
/// a : var_like
///     The shape and data-type of `a` define these same attributes of
///     the returned array.
/// dtype : data-type, optional
///     Overrides the data type of the result.
/// order : {'C', 'F', 'A', or 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional.
///     Compatible with numpy.
/// shape : int or sequence of ints, optional.
///     Overrides the shape of the result.
///
/// Returns
/// -------
/// out : Var
///     Var of zeros with the same shape and type as `a`.
///
/// Examples
/// --------
/// >>> x = np.arange(6)
/// >>> x = x.reshape((2, 3))
/// >>> x
/// var([[0, 1, 2],
///       [3, 4, 5]])
/// >>> np.zeros_like(x)
/// var([[0, 0, 0],
///       [0, 0, 0]])
/// >>> y = np.arange(3, dtype=float)
/// >>> y
/// var([0., 1., 2.])
/// >>> np.zeros_like(y)
/// var([0.,  0.,  0.])
VARP zerosLike<T extends ffi.SizedNativeType>(
  VARP a, {
  String order = "K",
  List<int>? shape,
}) {
  return fullLike<T>(a, 0, order: order, shape: shape);
}

/// copy(a, order='K', subok=False)
/// Return an array copy of the given object.
///
/// Parameters
/// ----------
/// a : var_like
///     Input data.
/// order : {'C', 'F', 'A', 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// arr : var
///     Var interpretation of `a`.
///
/// Examples
/// --------
/// >>> x = np.array([1, 2, 3])
/// >>> np.copy(x)
/// var([1, 2, 3])
VARP copy<T extends ffi.SizedNativeType>(
  VARP a, {
  String order = "K",
}) {
  _orderAssert(order);
  return F.clone(a, deepCopy: true);
}

/// array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0, like=None)
/// Create an array.
///
/// Parameters
/// ----------
/// object : var_like
///     An var, any object exposing the var interface, an object
///     whose __array__ method returns an array, or any (nested)
///     sequence. If object is a scalar, a 0-dimensional array
///     containing object is returned.
/// dtype : data-type, optional
///     The desired data-type for the array. If not given, then the
///     type will be determined as the minimum type required to
///     hold the objects in the sequence.
/// copy : bool, optional
///     If true (default), then the object is copied. Otherwise, a copy
///     will only be made if __array__ returns a copy, if obj is a nested
///     sequence, or if a copy is needed to satisfy any of the other
///     requirements (dtype, order, etc.).
/// order : {'C', 'F', 'A', 'K'}, optional
///     Compatible with numpy.
/// subok : bool, optional
///     Compatible with numpy.
/// ndmin : int, optional
///     Specifies the minimum number of dimensions that the
///     resulting array should have. Ones will be pre-pended to the
///     shape as needed to meet this requirement.
/// like : var_like
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     An var object satisfying the specified requirements.
///
/// Examples
/// --------
/// >>> np.array([1, 2, 3])
/// var([1, 2, 3])
/// >>> np.array([[1, 2], [3, 4]])
/// var([[1, 2],
///       [3, 4]])
VARP array<T extends ffi.SizedNativeType>(
  dynamic a, {
  String order = "K",
  bool copy = true,
  int ndim = 0,
}) {
  _orderAssert(order);
  VARP x;
  if (a is VARP) {
    x = F.clone(a, deepCopy: copy);
  } else {
    x = _toVARP<T>(a);
  }
  x = F.cast<T>(x);
  if (ndim > 0 && x.ndim != null) {
    final dim = x.ndim!;
    if (ndim > dim) {
      x = F.unsqueeze(x, axis: List.generate(ndim - dim, (index) => index));
    }
  }
  return x;
}

/// asarray(a, dtype=None, order=None)
/// Convert the input to an array.
///
/// Parameters
/// ----------
/// a : var_like
///     Input data, in any form that can be converted to an array.
///     This includes lists, lists of tuples, tuples, tuples of tuples,
///     tuples of lists and ndarrays.
/// dtype : data-type, optional
///     By default, the data-type is inferred from the input data.
/// order : {'C', 'F', 'A', 'K'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     Array interpretation of a. No copy is performed if the input is
///     already an ndarray with matching dtype and order.
///
/// Examples
/// --------
/// >>> a = [1, 2]
/// >>> np.asarray(a)
/// var([1, 2])
VARP asarray<T extends ffi.SizedNativeType>(dynamic a, {String order = "K"}) {
  return array<T>(a, order: order);
}

/// asanyarray(a, dtype=None, order=None)
/// Convert the input to an array.
///
/// Parameters
/// ----------
/// a : var_like
///     Input data, in any form that can be converted to an array.
///     This includes lists, lists of tuples, tuples, tuples of tuples,
///     tuples of lists and ndarrays.
/// dtype : data-type, optional
///     By default, the data-type is inferred from the input data.
/// order : {'C', 'F', 'A', 'K'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     Array interpretation of a. No copy is performed if the input is
///     already an ndarray with matching dtype and order.
///
/// Examples
/// --------
/// >>> a = [1, 2]
/// >>> np.asanyarray(a)
/// var([1, 2])
VARP asanyarray<T extends ffi.SizedNativeType>(dynamic a, {String order = "K"}) {
  return array<T>(a, order: order);
}

/// ascontiguousarray(a, dtype=None, order=None)
/// Return a contiguous array (ndim >= 1) in memory (C order).
///
/// Parameters
/// ----------
/// a : var_like
///     Input data, in any form that can be converted to an array.
///     This includes lists, lists of tuples, tuples, tuples of tuples,
///     tuples of lists and ndarrays.
/// dtype : data-type, optional
///     By default, the data-type is inferred from the input data.
/// order : {'C', 'F', 'A', 'K'}, optional
///     Compatible with numpy.
///
/// Returns
/// -------
/// out : var
///     Array interpretation of a. No copy is performed if the input is
///     already an ndarray with matching dtype and order.
///
/// Examples
/// --------
/// >>> a = [1, 2]
/// >>> np.ascontiguousarray(a)
/// var([1, 2])
VARP ascontiguousarray<T extends ffi.SizedNativeType>(dynamic a, {String order = "K"}) {
  return array<T>(a, order: order);
}

/// asmatrix(a, dtype=None)
/// Interpret the input as a matrix.
///
/// Parameters
/// ----------
/// a : var_like
///     Input data.
/// dtype : data-type, optional
///     Data-type of the output matrix.
///
/// Returns
/// -------
/// out : var
///     data interpreted as a matrix.
///
/// Examples
/// --------
/// >>> a = [[1, 2], [3, 4]]
/// >>> np.asmatrix(a)
/// var([[1, 2],
///       [3, 4]])
VARP asmatrix<T extends ffi.SizedNativeType>(dynamic a, {String order = "K"}) {
  return array<T>(a, order: order);
}

/// arange([start, ]stop, [step, ]dtype=None)
/// Return evenly spaced values within a given interval.
///
/// Parameters
/// ----------
/// start : integer or real, optional
///     Start of interval. The interval includes this value. The default
///     start value is 0.
/// stop : integer or real
///     End of interval. The interval does not include this value,
///     except in some cases where step is not an integer and
///     floating point round-off affects the length of out.
/// step : integer or real, optional
///     Spacing between values. For any output out, this is the
///     distance between two adjacent values, ``out[i+1] - out[i]``.
///     The default step size is 1. If step is specified as a
///     position argument, start must also be given.
/// dtype : dtype
///     The type of the output array. If dtype is not given, infer the
///     data type from the other input arguments.
///
/// Returns
/// -------
/// arange : var
///     Var of evenly spaced values.
///
/// Examples
/// --------
/// >>> np.arange(0, 5, 1)
/// var([0, 1, 2, 3, 4])
/// >>> np.arange(5.)
/// var([0., 1., 2., 3., 4.])
VARP arange<T extends ffi.SizedNativeType>({required num stop, num? start, num? step}) {
  final x = F.range(F.scalar<T>(start ?? 0), F.scalar<T>(stop), F.scalar<T>(step ?? 1));
  return F.cast<T>(x);
}

VARP linspace<T extends ffi.SizedNativeType>(num start, num stop, {int count = 50}) {
  final x = F.linSpace(F.scalar<T>(start), F.scalar<T>(stop), F.scalar<int32>(count));
  return F.cast<T>(x);
}

VARP logspace<T extends ffi.SizedNativeType>(num start, num stop, {int count = 50, double base = 10.0}) {
  final pow = F.cast<T>(linspace<T>(start, stop, count: count));
  return F.pow(pow, F.scalar<T>(base));
}

VARP geomspace<T extends ffi.SizedNativeType>(num start, num stop, {int count = 50}) {
  throw UnimplementedError();
}

VARP meshgrid<T extends ffi.SizedNativeType>(VARP x, VARP y) {
  throw UnimplementedError();
}

VARP diag(VARP v, {int k = 0}) => throw UnimplementedError();
VARP diagflat(VARP v, {int k = 0}) => throw UnimplementedError();
VARP tri(VARP N, {int? M, int k = 0}) => throw UnimplementedError();
VARP tril(VARP v, {int k = 0}) => throw UnimplementedError();
VARP triu(VARP v, {int k = 0}) => throw UnimplementedError();
VARP vander(VARP x, {int? n}) => throw UnimplementedError();

VARP mat<T extends ffi.SizedNativeType>(dynamic data) => asmatrix<T>(data);
VARP matrix<T extends ffi.SizedNativeType>(dynamic data) => asmatrix<T>(data);

// Array manipulation routines
List<int> shape(VARP a) => a.shape ?? [];
VARP reshape(VARP a, List<int> newShape, {DimensionFormat format = DimensionFormat.NCHW}) =>
    F.reshape(a, newShape, format: format);
VARP ravel(VARP a) => F.reshape(a, [-1]);

VARP moveaxis(VARP a, List<int> source, List<int> destination) {
  final ndim = a.ndim;
  MnnAssert(ndim != null, "a must have a shape");
  final source_ = _normalAxes(source, ndim!);
  final destination_ = _normalAxes(destination, ndim);
  MnnAssert(source_.length == destination_.length, "source and destination must have the same length");
  final axes = List.generate(ndim, (index) => index).where((e) => !source_.contains(e)).toList();
  final sorted = List.generate(source_.length, (i) => i).map((i) => (destination_[i], source_[i])).toList();
  sorted.sort((a, b) => a.$1.compareTo(b.$1));
  for (final (dst, src) in sorted) {
    axes.insert(dst, src);
  }
  return F.transpose(a, axes);
}

VARP rollaxis(VARP a, int axis, {int start = 0}) {
  final ndim = a.ndim;
  MnnAssert(ndim != null, "a must have a shape");
  final axis_ = _normalAxis(axis, ndim!);
  var start_ = _normalAxis(start, ndim);
  MnnAssert(start_ >= 0 && start_ < ndim + 1, "start must be in the range [0, ${ndim + 1})");
  if (axis_ < start_) {
    start_ -= 1;
  }
  if (axis_ == start_) {
    return a;
  }

  final axes = List.generate(ndim, (index) => index);
  axes.remove(axis_);
  axes.insert(start_, axis_);
  return F.transpose(a, axes);
}

VARP swapaxes(VARP a, int axis1, int axis2) {
  final ndim = a.ndim;
  MnnAssert(ndim != null, "a must have a shape");
  final axes = List.generate(ndim!, (index) => index);
  final tmp = axes[axis1];
  axes[axis1] = axes[axis2];
  axes[axis2] = tmp;
  return F.transpose(a, axes);
}

VARP transpose(VARP a, {List<int>? axes}) {
  final ndim = a.ndim;
  MnnAssert(ndim != null, "a must have a shape");
  axes ??= List.generate(ndim!, (index) => index).reversed.toList();
  return F.transpose(a, axes);
}

VARP broadcast(VARP x, VARP y) => throw UnimplementedError();
VARP broadcastTo(VARP x, List<int> shape) {
  final srcShape = x.shape;
  MnnAssert(srcShape != null, "x must have a shape");
  MnnAssert(_canBroadcast(srcShape!, shape), "can't broadcast from $srcShape to $shape");
  return F.broadcastTo(x, F.constant<int32>(shape, [shape.length], format: DimensionFormat.NCHW));
}

VARP expandDims(VARP x, List<int> axis) => F.unsqueeze(x, axis: axis);

VARP squeeze(VARP x, {List<int> axis = const []}) => F.squeeze(x, axis: axis);

VARP concatenate(List<VARP> vars, {int axis = 0}) => F.concat(vars, axis);

List<VARP> split(VARP arr, List<int> indicesOrSections, {int axis = 0}) {
  MnnAssert(arr.shape != null, "arr must have a shape");
  final sizeSplits = <int>[indicesOrSections[0]];
  var idxExceeds = false;

  final axisLength = arr.shape![axis];
  for (var i = 1; i < indicesOrSections.length; i++) {
    var nowIdx = indicesOrSections[i];
    if (indicesOrSections[i] > axisLength) {
      idxExceeds = true;
      nowIdx = axisLength;
    }
    sizeSplits.add(nowIdx - indicesOrSections[i - 1]);
  }
  final res = F.split(arr, sizeSplits, axis: axis);
  if (idxExceeds) {
    res.add(array<int32>([0])); // TODO: MNN not support empty Var
  }
  return res;
}

VARP tile(VARP x, List<int> reps) =>
    F.tile(x, F.constant<int32>(reps, [reps.length], format: DimensionFormat.NCHW));

VARP repeat(VARP x, int reps) {
  final xx = expandDims(ravel(x), [-1]);
  final repeats = [1, reps];
  return ravel(tile(xx, repeats));
}

VARP bitwiseAnd(VARP x, VARP y) => F.bitwiseAnd(x, y);
VARP bitwiseOr(VARP x, VARP y) => F.bitwiseOr(x, y);
VARP bitwiseXor(VARP x, VARP y) => F.bitwiseXor(x, y);
VARP where(VARP condition, {VARP? x, VARP? y}) {
  if (x == null && y == null) {
    return nonzero(condition).$1;
  }
  MnnAssert(x != null && y != null, "x and y must be provided");
  return F.select(condition, x!, y!);
}

VARP dot<T extends ffi.SizedNativeType>(VARP a, VARP b) {
  final ad = a.ndim!;
  final bd = b.ndim!;

  if (ad == 0 || bd == 0) {
    return F.multiply(a, b);
  }

  if (ad == 1 && bd == 1) {
    MnnAssert(a.shape![0] == b.shape![0], 'shapes not aligned');
    return F.reduceSum(F.multiply(a, b));
  }

  if (ad > 1 && bd == 1) {
    MnnAssert(a.shape!.last == b.shape![0], 'shapes not aligned');
    return F.reduceSum(F.multiply(a, b), axis: [-1]);
  }

  if (ad == 2 && bd == 2) {
    MnnAssert(a.shape![1] == b.shape![0], 'shapes not aligned');
    return F.matMul(a, b);
  }

  if (ad > 2 && bd > 1) {
    final reduceDim = a.shape!.last;
    MnnAssert(reduceDim == b.shape![bd - 2], 'shapes not aligned');

    final aShape = List.of(a.shape!);
    final bShape = List.of(b.shape!);
    aShape.removeLast();
    bShape.removeAt(bd - 2);
    final dstShape = [...aShape, ...bShape];

    final newA = reshape(a, [-1, reduceDim]);
    var newB = moveaxis(b, [bd - 2], [0]);
    newB = reshape(newB, [reduceDim, -1]);

    final res = F.matMul(newA, newB);
    return reshape(res, dstShape);
  }

  throw ArgumentError('dot not implemented for dimensions $ad and $bd');
}

VARP vdot(VARP a, VARP b) => dot(a, b);
VARP inner(VARP a, VARP b) => dot(a, b);
VARP outer(VARP a, VARP b) => throw UnimplementedError();
VARP matmul(VARP a, VARP b) => dot(a, b);

VARP all(VARP a, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceAll(a, axis: axis, keepDims: keepDims);
}

VARP any(VARP a, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceAny(a, axis: axis, keepDims: keepDims);
}

VARP arrayEqual(VARP a1, VARP a2, {bool equalNan = false}) {
  return all(F.equal(a1, a2));
}

VARP arrayEquiv(VARP a1, VARP a2) {
  return arrayEqual(a1, a2);
}

VARP greater(VARP a, VARP b) => F.greater(a, b);
VARP greaterEqual(VARP a, VARP b) => F.greaterEqual(a, b);
VARP less(VARP a, VARP b) => F.less(a, b);
VARP lessEqual(VARP a, VARP b) => F.lessEqual(a, b);
VARP notEqual(VARP a, VARP b) => F.notEqual(a, b);
VARP equal(VARP a, VARP b) => F.equal(a, b);
VARP sin(VARP x) => F.sin(x);
VARP cos(VARP x) => F.cos(x);
VARP tan(VARP x) => F.tan(x);
VARP arcsin(VARP x) => F.asin(x);
VARP arccos(VARP x) => F.acos(x);
VARP arctan(VARP x) => F.atan(x);
VARP arctan2(VARP x, VARP y) => F.atan2(x, y);
VARP sinh(VARP x) => F.sinh(x);
VARP cosh(VARP x) => F.cosh(x);
VARP tanh(VARP x) => F.tanh(x);
VARP arcsinh(VARP x) => F.asinh(x);
VARP arccosh(VARP x) => F.acosh(x);
VARP arctanh(VARP x) => F.atanh(x);
VARP around(VARP x) => F.round(x);
VARP round(VARP x) => F.round(x);
VARP rint(VARP x) => F.round(x);
VARP fix(VARP x) {
  final zero = F.subtract(x, x);
  return where(greaterEqual(x, zero), x: floor(x), y: ceil(x));
}

VARP floor(VARP x) => F.floor(x);
VARP ceil(VARP x) => F.ceil(x);
VARP trunc(VARP x) => fix(x);
VARP prod(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceProd(x, axis: axis, keepDims: keepDims);
}

VARP sum(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceSum(x, axis: axis, keepDims: keepDims);
}

VARP sqrt(VARP x) => F.sqrt(x);
VARP exp(VARP x) => F.exp(x);
VARP expm1(VARP x) => F.expm1(x);
VARP log(VARP x) => F.log(x);
VARP log1p(VARP x) => F.log1p(x);
VARP sign(VARP x) => F.sign(x);
VARP reciprocal(VARP x) => F.reciprocal(x);
VARP positive(VARP x) => F.clone(x);
VARP negative(VARP x) => F.negative(x);
VARP multiply(VARP x, VARP y) => F.multiply(x, y);
VARP add(VARP x, VARP y) => F.add(x, y);
VARP divide(VARP x, VARP y) => F.divide(x, y);
VARP power(VARP x, VARP y) => F.pow(x, y);
VARP subtract(VARP x, VARP y) => F.subtract(x, y);
VARP trueDiv(VARP x, VARP y) => divide(x, y);
VARP floorDiv(VARP x, VARP y) => F.floorDiv(x, y);
VARP mod(VARP x, VARP y) => F.mod(x, y);
VARP square(VARP x) => F.square(x);
VARP abs(VARP x) => F.abs(x);
VARP absolute(VARP x) => F.abs(x);
VARP fabs(VARP x) => F.abs(x);
VARP maximum(VARP x, VARP y) => F.maximum(x, y);
VARP minimum(VARP x, VARP y) => F.minimum(x, y);

VARP hypot(VARP x, VARP y) => F.sqrt(x * x + y * y);
VARP exp2(VARP x) => F.pow(F.scalar<float32>(2, dtype: x.dtype), x);
VARP log2(VARP x) => F.log(x) / F.log(F.scalar<float32>(2, dtype: x.dtype));
VARP log10(VARP x) => F.log(x) / F.log(F.scalar<float32>(10, dtype: x.dtype));
VARP logaddexp(VARP x, VARP y) => F.log(F.exp(x) + F.exp(y));
VARP logaddexp2(VARP x, VARP y) => log2(exp2(x) + exp2(y));
VARP sinc(VARP x) {
  final pi = F.scalar(math.pi, dtype: x.dtype);
  final pix = pi * x;
  final zero = F.scalar(0, dtype: x.dtype);
  final one = F.scalar(1, dtype: x.dtype);
  final mask = equal(x, zero);
  final safePix = where(mask, x: one, y: pix);
  return where(mask, x: one, y: sin(pix) / safePix);
}

VARP signbit(VARP x) => equal(sign(x), F.scalar(-1, dtype: x.dtype));
VARP copysign(VARP x, VARP y) => x * sign(y);
VARP ldexp(VARP x, VARP y) => x * exp2(y);
(VARP, VARP) divmod(VARP x, VARP y) => (F.floorDiv(x, y), F.mod(x, y));
VARP clip(VARP x, double aMin, double aMax) {
  return F.cast(
    F.ReLU6(F.cast<float32>(x), minValue: aMin, maxValue: aMax),
    dtype: x.dtype,
  );
}

VARP cbrt(VARP x) => power(x, F.scalar(1.0 / 3.0, dtype: x.dtype));

VARP cumprod(VARP x, int axis) => F.cumProd(x, axis);
VARP cumsum(VARP x, int axis) => F.cumSum(x, axis);

VARP pad(VARP x, List<int> padWidth, {F.PadValueMode mode = F.PadValueMode.CONSTANT}) {
  final ndim = x.ndim;
  MnnAssert(ndim != null, "x must have a shape");
  var padWidth_ = asarray<int32>(padWidth);
  MnnAssert(padWidth_.dtype == HalideType.i32, "padWidth must be int32");
  padWidth_ = broadcastTo(padWidth_, [ndim!, 2]);
  return F.pad(x, padWidth_, mode: mode);
}

VARP sort(VARP x, {int axis = -1, bool descend = false}) =>
    F.sort(x, axis: axis, arg: false, descend: descend);

// VARP lexsort(List<VARP> vars, {int axis = -1}) => sort(vars, axis: axis);
VARP argsort(VARP x, {int axis = -1, bool descend = false}) =>
    F.sort(x, axis: axis, arg: true, descend: descend);

VARP msort(VARP x) => sort(x, axis: 0);

VARP argmax(VARP x, {int? axis}) {
  if (axis == null) {
    return F.argMax(ravel(x), 0);
  }
  return F.argMax(x, axis);
}

VARP argmin(VARP x, {int? axis}) {
  if (axis == null) {
    return F.argMin(ravel(x), 0);
  }
  return F.argMin(x, axis);
}

VARP argwhere(VARP x) {
  final mask = F.notEqual(x, F.scalar(0, dtype: x.dtype));
  return F.where(mask);
}

(VARP, VARP?) nonzero(VARP x) {
  final res = F.where(x);
  if (x.ndim == 1) {
    return (res, null);
  }
  final splitted = split(x, [2]);
  return (ravel(splitted[0]), ravel(splitted[1]));
}

VARP flatnonzero(VARP x) => nonzero(x).$1;

VARP countNonZero(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  final mask = F.notEqual(x, F.scalar(0, dtype: x.dtype));
  return F.reduceSum(mask, axis: axis, keepDims: keepDims);
}

VARP max(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceMax(x, axis: axis, keepDims: keepDims);
}

VARP min(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  return F.reduceMin(x, axis: axis, keepDims: keepDims);
}

VARP ptp(VARP x, {List<int> axis = const []}) => max(x, axis: axis) - min(x, axis: axis);

VARP median(VARP x, {int axis = -1, bool keepDims = false}) => throw UnimplementedError();
VARP mean(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    F.reduceMean(x, axis: axis, keepDims: keepDims);
VARP variance(VARP x, {List<int> axis = const [], bool keepDims = false}) =>
    F.reduceVariance(x, axis: axis, keepDims: keepDims);
VARP std(VARP x, {List<int> axis = const [], bool keepDims = false}) {
  final var_ = variance(x, axis: axis, keepDims: keepDims);
  return F.sqrt(var_);
}

(VARP, VARP) histogram(VARP x, {int bins = 10, (int, int)? range}) {
  final (min, max) = range ?? (F.reduceMin(x).value.toInt(), F.reduceMax(x).value.toInt());
  final hist = F.histogram(x, bins, min, max);
  final binEdges = linspace<float32>(min, max, count: bins + 1);
  return (hist, binEdges);
}
