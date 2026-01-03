/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../core/base.dart';
import '../g/mnn.g.dart' as c;

class Point extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  Point.fromPointer(ffi.Pointer<c.mnn_cv_point_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());
  factory Point.fromXY(double x, double y) {
    final ptr = calloc<c.mnn_cv_point_t>()
      ..ref.x = x
      ..ref.y = y;
    return Point.fromPointer(ptr);
  }

  factory Point.fromTuple((double, double) tuple) {
    return Point.fromXY(tuple.$1, tuple.$2);
  }

  factory Point.fromNative(c.mnn_cv_point_t p) {
    final ptr = calloc<c.mnn_cv_point_t>()
      ..ref.x = p.x
      ..ref.y = p.y;
    return Point.fromPointer(ptr);
  }

  c.mnn_cv_point_t get ref => ptr.cast<c.mnn_cv_point_t>().ref;
  double get x => ref.x;
  double get y => ref.y;
  @override
  ffi.NativeFinalizer get finalizer => _finalizer;
  @override
  List<Object?> get props => [x, y];
  @override
  void release() {
    calloc.free(ptr);
  }
}

class Size extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  Size.fromPointer(ffi.Pointer<c.mnn_cv_size2i_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory Size.fromXY(int x, int y) {
    final ptr = calloc<c.mnn_cv_size2i_t>()
      ..ref.width = x
      ..ref.height = y;
    return Size.fromPointer(ptr);
  }

  factory Size.fromTuple((int, int) tuple) {
    return Size.fromXY(tuple.$1, tuple.$2);
  }

  factory Size.fromNative(c.mnn_cv_size2i_t p) {
    final ptr = calloc<c.mnn_cv_size2i_t>()
      ..ref.width = p.width
      ..ref.height = p.height;
    return Size.fromPointer(ptr);
  }

  c.mnn_cv_size2i_t get ref => ptr.cast<c.mnn_cv_size2i_t>().ref;
  int get width => ref.width;
  int get height => ref.height;
  @override
  ffi.NativeFinalizer get finalizer => _finalizer;
  @override
  List<Object?> get props => [width, height];
  @override
  void release() {
    calloc.free(ptr.cast<c.mnn_cv_size2i_t>());
  }
}

class Scalar extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  Scalar.fromPointer(ffi.Pointer<c.mnn_cv_scalar_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory Scalar.fromVal(List<double> values) {
    MnnAssert(values.length == 4, 'Scalar must have 4 values');
    final ptr = calloc<c.mnn_cv_scalar_t>()
      ..ref.val[0] = values[0]
      ..ref.val[1] = values[1]
      ..ref.val[2] = values[2]
      ..ref.val[3] = values[3];
    return Scalar.fromPointer(ptr);
  }

  factory Scalar.fromNative(c.mnn_cv_scalar_t p) {
    final ptr = calloc<c.mnn_cv_scalar_t>()..ref.val = p.val;
    return Scalar.fromPointer(ptr);
  }

  c.mnn_cv_scalar_t get ref => ptr.cast<c.mnn_cv_scalar_t>().ref;
  List<double> get val => ref.val.elements;
  @override
  ffi.NativeFinalizer get finalizer => _finalizer;
  @override
  List<Object?> get props => val;
  @override
  void release() {
    calloc.free(ptr.cast<c.mnn_cv_scalar_t>());
  }
}

class Rect extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  Rect.fromPointer(ffi.Pointer<c.mnn_cv_rect_t> ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory Rect.fromLTRB(double left, double top, double right, double bottom) {
    final ptr = calloc<c.mnn_cv_rect_t>()
      ..ref.left = left
      ..ref.top = top
      ..ref.right = right
      ..ref.bottom = bottom;
    return Rect.fromPointer(ptr);
  }

  c.mnn_cv_rect_t get ref => ptr.cast<c.mnn_cv_rect_t>().ref;

  double get left => ref.left;
  double get top => ref.top;
  double get right => ref.right;
  double get bottom => ref.bottom;
  double get width => right - left;
  double get height => bottom - top;

  double get x => ref.left;
  double get y => ref.top;
  double get w => width;
  double get h => height;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [x, y, w, h];

  @override
  void release() {
    calloc.free(ptr);
  }
}

class RotatedRect extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  RotatedRect.fromPointer(ffi.Pointer<c.mnn_cv_rotated_rect_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory RotatedRect.create((double, double) center, double width, double height, {double angle = 0.0}) {
    final ptr = calloc<c.mnn_cv_rotated_rect_t>()
      ..ref.center_x = center.$1
      ..ref.center_y = center.$2
      ..ref.width = width
      ..ref.height = height
      ..ref.angle = angle;
    return RotatedRect.fromPointer(ptr);
  }

  c.mnn_cv_rotated_rect_t get ref => ptr.cast<c.mnn_cv_rotated_rect_t>().ref;

  (double, double) get center => (ref.center_x, ref.center_y);
  double get centerX => ref.center_x;
  double get centerY => ref.center_y;
  double get width => ref.width;
  double get height => ref.height;
  double get angle => ref.angle;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [centerX, centerY, width, height, angle];

  @override
  void release() {
    calloc.free(ptr.cast<c.mnn_cv_rotated_rect_t>());
  }
}
