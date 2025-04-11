/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;

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
