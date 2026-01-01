// Copyright (c) 2024, rainyl and all contributors. All rights reserved.
// Use of this source code is governed by a Apache-2.0 license
// that can be found in the LICENSE file.

import 'dart:collection';
import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../g/mnn.g.dart' as C;
import 'base.dart';

abstract class Vec<N extends ffi.Pointer, T> extends NativeObject with IterableMixin<T> {
  Vec.fromPointer(N ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  @override
  int get length;

  T operator [](int idx);
  void operator []=(int idx, T value);

  Vec clone();
  int size();
  void add(T element);
  void resize(int newSize);
  void reserve(int newCapacity);
  void clear();
  void shrinkToFit();
  void extend(Vec other);

  ffi.Pointer<ffi.Void> asVoid();

  @override
  List<Object?> get props => [ptr.address, length];
}

abstract class VecIterator<T> implements Iterator<T> {
  int currentIndex = -1;
  int get length;
  T operator [](int idx);

  @override
  T get current {
    if (currentIndex >= 0 && currentIndex < length) {
      return this[currentIndex];
    }
    throw IndexError.withLength(currentIndex, length);
  }

  @override
  bool moveNext() {
    if (currentIndex < length - 1) {
      currentIndex++;
      return true;
    }
    return false;
  }
}

abstract class VecUnmodifible<N extends ffi.Pointer, T> extends Vec<N, T> {
  VecUnmodifible.fromPointer(super.ptr) : super.fromPointer();

  @override
  void operator []=(int idx, T value) => throw UnsupportedError("Unmodifiable Vec");

  @override
  void add(T element) => throw UnsupportedError("Unmodifiable Vec");
  @override
  void resize(int newSize) => throw UnsupportedError("Unmodifiable Vec");
  @override
  void reserve(int newCapacity) => throw UnsupportedError("Unmodifiable Vec");
  @override
  void clear() => throw UnsupportedError("Unmodifiable Vec");
  @override
  void shrinkToFit() => throw UnsupportedError("Unmodifiable Vec");
  @override
  void extend(Vec other) => throw UnsupportedError("Unmodifiable Vec");
}

class VecU8 extends Vec<C.VecU8, int> {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecU8_free);

  VecU8.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Uint8>());

  factory VecU8([int length = 0, int value = 0]) =>
      VecU8.fromPointer(C.std_VecU8_new_1(length, value), length: length);

  factory VecU8.fromList(List<int> pts) {
    final length = pts.length;
    final p = C.std_VecU8_new(length);
    final pdata = C.std_VecU8_data(p);
    pdata.cast<ffi.Uint8>().asTypedList(length).setAll(0, pts);
    return VecU8.fromPointer(p, length: pts.length);
  }

  factory VecU8.generate(int length, int Function(int i) generator) {
    final p = C.std_VecU8_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecU8_set(p, i, generator(i));
    }
    return VecU8.fromPointer(p, length: length);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  int get length => C.std_VecU8_length(ptr);

  ffi.Pointer<ffi.Uint8> get dataPtr => C.std_VecU8_data(ptr);

  /// Returns a view of native pointer
  Uint8List get data => dataPtr.cast<ffi.Uint8>().asTypedList(length);
  Uint8List get dataView => dataPtr.cast<ffi.Uint8>().asTypedList(length);

  /// ~~alias of data~~
  ///
  /// This method will return a full copy of [data]
  ///
  /// https://github.com/rainyl/opencv_dart/issues/85
  Uint8List toU8List() => Uint8List.fromList(data);

  @override
  VecU8 clone() => VecU8.fromPointer(C.std_VecU8_clone(ptr));

  @override
  Iterator<int> get iterator => VecU8Iterator(dataView);

  @override
  void release() {
    C.std_VecU8_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, int value) => C.std_VecU8_set(ptr, idx, value);

  @override
  int operator [](int idx) => C.std_VecU8_get(ptr, idx);

  @override
  void add(int element) => C.std_VecU8_push_back(ptr, element);

  @override
  void clear() => C.std_VecU8_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecU8_extend(ptr, (other as VecU8).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecU8_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecU8_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecU8_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecU8_length(ptr);
}

class VecU8Iterator extends VecIterator<int> {
  VecU8Iterator(this.dataView);
  Uint8List dataView;

  @override
  int get length => dataView.length;

  @override
  int operator [](int idx) => dataView[idx];
}

class VecI8 extends Vec<C.VecI8, int> {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecI8_free);

  VecI8.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Int8>());

  factory VecI8([int length = 0, int value = 0]) =>
      VecI8.fromPointer(C.std_VecI8_new_1(length, value), length: length);
  factory VecI8.fromList(List<int> pts) {
    final length = pts.length;
    final p = C.std_VecI8_new(length);
    final pdata = C.std_VecI8_data(p);
    pdata.cast<ffi.Int8>().asTypedList(length).setAll(0, pts);
    return VecI8.fromPointer(p, length: pts.length);
  }

  factory VecI8.generate(int length, int Function(int i) generator) {
    final p = C.std_VecI8_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecI8_set(p, i, generator(i));
    }
    return VecI8.fromPointer(p, length: length);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  int get length => C.std_VecI8_length(ptr);

  ffi.Pointer<ffi.Int8> get dataPtr => C.std_VecI8_data(ptr);

  Uint8List get data => dataPtr.cast<ffi.Uint8>().asTypedList(length);
  Int8List get dataView => dataPtr.cast<ffi.Int8>().asTypedList(length);

  String asString() => utf8.decode(data);

  @override
  VecI8 clone() => VecI8.fromPointer(C.std_VecI8_clone(ptr));

  @override
  Iterator<int> get iterator => VecI8Iterator(dataView);

  @override
  void release() {
    C.std_VecI8_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, int value) => C.std_VecI8_set(ptr, idx, value);

  @override
  int operator [](int idx) => C.std_VecI8_get(ptr, idx);

  @override
  void add(int element) => C.std_VecI8_push_back(ptr, element);

  @override
  void clear() => C.std_VecI8_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecI8_extend(ptr, (other as VecI8).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecI8_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecI8_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecI8_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecI8_length(ptr);
}

class VecI8Iterator extends VecIterator<int> {
  VecI8Iterator(this.dataView);
  Int8List dataView;

  @override
  int get length => dataView.length;

  @override
  int operator [](int idx) => dataView[idx];
}

class VecU16 extends Vec<C.VecU16, int> {
  VecU16.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Uint16>());

  factory VecU16([int length = 0, int value = 0]) =>
      VecU16.fromPointer(C.std_VecU16_new_1(length, value), length: length);

  factory VecU16.fromList(List<int> pts) {
    final length = pts.length;
    final p = C.std_VecU16_new(length);
    final pdata = C.std_VecU16_data(p);
    pdata.asTypedList(length).setAll(0, pts);
    return VecU16.fromPointer(p, length: pts.length);
  }

  factory VecU16.generate(int length, int Function(int i) generator) {
    final p = C.std_VecU16_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecU16_set(p, i, generator(i));
    }
    return VecU16.fromPointer(p, length: length);
  }

  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecU16_free);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  VecU16 clone() => VecU16.fromPointer(C.std_VecU16_clone(ptr));

  @override
  int get length => C.std_VecU16_length(ptr);

  ffi.Pointer<ffi.Uint16> get dataPtr => C.std_VecU16_data(ptr);

  Uint16List get data => dataPtr.cast<ffi.Uint16>().asTypedList(length);

  @override
  Iterator<int> get iterator => VecU16Iterator(data);

  @override
  void release() {
    C.std_VecU16_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, int value) => data[idx] = value;

  @override
  int operator [](int idx) => data[idx];

  @override
  void add(int element) => C.std_VecU16_push_back(ptr, element);

  @override
  void clear() => C.std_VecU16_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecU16_extend(ptr, (other as VecU16).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecU16_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecU16_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecU16_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecU16_length(ptr);
}

class VecU16Iterator extends VecIterator<int> {
  VecU16Iterator(this.data);
  Uint16List data;

  @override
  int get length => data.length;

  @override
  int operator [](int idx) => data[idx];
}

class VecI16 extends Vec<C.VecI16, int> {
  VecI16.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Int16>());

  factory VecI16([int length = 0, int value = 0]) =>
      VecI16.fromPointer(C.std_VecI16_new_1(length, value), length: length);

  factory VecI16.fromList(List<int> pts) {
    final length = pts.length;
    final p = C.std_VecI16_new(length);
    final pdata = C.std_VecI16_data(p);
    pdata.asTypedList(length).setAll(0, pts);
    return VecI16.fromPointer(p, length: pts.length);
  }

  factory VecI16.generate(int length, int Function(int i) generator) {
    final p = C.std_VecI16_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecI16_set(p, i, generator(i));
    }
    return VecI16.fromPointer(p, length: length);
  }

  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecI16_free);
  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  VecI16 clone() => VecI16.fromPointer(C.std_VecI16_clone(ptr));

  @override
  int get length => C.std_VecI16_length(ptr);

  ffi.Pointer<ffi.Int16> get dataPtr => C.std_VecI16_data(ptr);

  Int16List get data => dataPtr.cast<ffi.Int16>().asTypedList(length);
  @override
  Iterator<int> get iterator => VecI16Iterator(data);

  @override
  void release() {
    C.std_VecI16_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, int value) => data[idx] = value;

  @override
  int operator [](int idx) => data[idx];

  @override
  void add(int element) => C.std_VecI16_push_back(ptr, element);

  @override
  void clear() => C.std_VecI16_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecI16_extend(ptr, (other as VecI16).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecI16_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecI16_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecI16_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecI16_length(ptr);
}

class VecI16Iterator extends VecIterator<int> {
  VecI16Iterator(this.data);
  Int16List data;

  @override
  int get length => data.length;

  @override
  int operator [](int idx) => data[idx];
}

class VecI32 extends Vec<C.VecI32, int> {
  VecI32.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Int32>());

  factory VecI32([int length = 0, int value = 0]) =>
      VecI32.fromPointer(C.std_VecI32_new_1(length, value), length: length);

  factory VecI32.fromList(List<int> pts) {
    final length = pts.length;
    final p = C.std_VecI32_new(length);
    final pdata = C.std_VecI32_data(p);
    pdata.asTypedList(length).setAll(0, pts);
    return VecI32.fromPointer(p, length: pts.length);
  }

  factory VecI32.generate(int length, int Function(int i) generator) {
    final p = C.std_VecI32_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecI32_set(p, i, generator(i));
    }
    return VecI32.fromPointer(p, length: length);
  }

  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecI32_free);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  VecI32 clone() => VecI32.fromPointer(C.std_VecI32_clone(ptr));

  @override
  int get length => C.std_VecI32_length(ptr);

  ffi.Pointer<ffi.Int32> get dataPtr => C.std_VecI32_data(ptr);

  Int32List get data => dataPtr.cast<ffi.Int32>().asTypedList(length);

  @override
  Iterator<int> get iterator => VecI32Iterator(data);

  @override
  void release() {
    C.std_VecI32_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, int value) => C.std_VecI32_set(ptr, idx, value);

  @override
  int operator [](int idx) => C.std_VecI32_get(ptr, idx);

  @override
  void add(int element) => C.std_VecI32_push_back(ptr, element);

  @override
  void clear() => C.std_VecI32_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecI32_extend(ptr, (other as VecI32).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecI32_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecI32_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecI32_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecI32_length(ptr);
}

class VecI32Iterator extends VecIterator<int> {
  VecI32Iterator(this.data);
  Int32List data;

  @override
  int get length => data.length;

  @override
  int operator [](int idx) => data[idx];
}

class VecF32 extends Vec<C.VecF32, double> {
  VecF32.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Float>());

  factory VecF32([int length = 0, double value = 0.0]) =>
      VecF32.fromPointer(C.std_VecF32_new_1(length, value), length: length);
  factory VecF32.fromList(List<double> pts) {
    final length = pts.length;
    final p = C.std_VecF32_new(length);
    final pdata = C.std_VecF32_data(p);
    pdata.asTypedList(length).setAll(0, pts);
    return VecF32.fromPointer(p, length: pts.length);
  }

  factory VecF32.generate(int length, double Function(int i) generator) {
    final p = C.std_VecF32_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecF32_set(p, i, generator(i));
    }
    return VecF32.fromPointer(p, length: length);
  }

  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecF32_free);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  VecF32 clone() => VecF32.fromPointer(C.std_VecF32_clone(ptr));

  @override
  int get length => C.std_VecF32_length(ptr);

  ffi.Pointer<ffi.Float> get dataPtr => C.std_VecF32_data(ptr);

  Float32List get data => dataPtr.cast<ffi.Float>().asTypedList(length);
  @override
  Iterator<double> get iterator => VecF32Iterator(data);

  @override
  void release() {
    C.std_VecF32_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, double value) => C.std_VecF32_set(ptr, idx, value);

  @override
  double operator [](int idx) => C.std_VecF32_get(ptr, idx);

  @override
  void add(double element) => C.std_VecF32_push_back(ptr, element);

  @override
  void clear() => C.std_VecF32_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecF32_extend(ptr, (other as VecF32).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecF32_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecF32_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecF32_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecF32_length(ptr);
}

class VecF32Iterator extends VecIterator<double> {
  VecF32Iterator(this.data);
  Float32List data;

  @override
  int get length => data.length;

  @override
  double operator [](int idx) => data[idx];
}

class VecF64 extends Vec<C.VecF64, double> {
  VecF64.fromPointer(super.ptr, {super.attach = true, int? length})
    : super.fromPointer(externalSize: length == null ? null : length * ffi.sizeOf<ffi.Double>());

  factory VecF64([int length = 0, double value = 0.0]) =>
      VecF64.fromPointer(C.std_VecF64_new_1(length, value), length: length);
  factory VecF64.fromList(List<double> pts) {
    final length = pts.length;
    final p = C.std_VecF64_new(length);
    final pdata = C.std_VecF64_data(p);
    pdata.asTypedList(length).setAll(0, pts);
    return VecF64.fromPointer(p, length: pts.length);
  }

  factory VecF64.generate(int length, double Function(int i) generator) {
    final p = C.std_VecF64_new(length);
    for (var i = 0; i < length; i++) {
      C.std_VecF64_set(p, i, generator(i));
    }
    return VecF64.fromPointer(p, length: length);
  }

  static final _finalizer = ffi.NativeFinalizer(C.addresses.std_VecF64_free);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  VecF64 clone() => VecF64.fromPointer(C.std_VecF64_clone(ptr));

  @override
  int get length => C.std_VecF64_length(ptr);

  ffi.Pointer<ffi.Double> get dataPtr => C.std_VecF64_data(ptr);

  Float64List get data => dataPtr.cast<ffi.Double>().asTypedList(length);

  @override
  Iterator<double> get iterator => VecF64Iterator(data);

  @override
  void release() {
    C.std_VecF64_free(ptr);
  }

  @override
  ffi.Pointer<ffi.Void> asVoid() => dataPtr.cast<ffi.Void>();

  @override
  void operator []=(int idx, double value) => data[idx] = value;

  @override
  double operator [](int idx) => data[idx];

  @override
  void add(double element) => C.std_VecF64_push_back(ptr, element);

  @override
  void clear() => C.std_VecF64_clear(ptr);

  @override
  void extend(Vec other) => C.std_VecF64_extend(ptr, (other as VecF64).ptr);

  @override
  void reserve(int newCapacity) => C.std_VecF64_reserve(ptr, newCapacity);

  @override
  void resize(int newSize) => C.std_VecF64_resize(ptr, newSize);

  @override
  void shrinkToFit() => C.std_VecF64_shrink_to_fit(ptr);

  @override
  int size() => C.std_VecF64_length(ptr);
}

class VecF64Iterator extends VecIterator<double> {
  VecF64Iterator(this.data);
  Float64List data;

  @override
  int get length => data.length;

  @override
  double operator [](int idx) => data[idx];
}

extension StringVecExtension on String {
  VecU8 get u8 {
    final p = toNativeUtf8();
    final pp = p.cast<ffi.Uint8>().asTypedList(p.length);
    final v = VecU8.fromList(pp);
    calloc.free(p);
    return v;
  }

  VecI8 get i8 {
    final p = toNativeUtf8();
    final pp = p.cast<ffi.Int8>().asTypedList(p.length);
    final v = VecI8.fromList(pp);
    calloc.free(p);
    return v;
  }
}

extension ListUCharExtension on List<int> {
  VecU8 get vecUChar => VecU8.fromList(this);
  VecI8 get vecChar => VecI8.fromList(this);
  VecU8 get u8 => VecU8.fromList(this);
  VecI8 get i8 => VecI8.fromList(this);
  VecU16 get u16 => VecU16.fromList(this);
  VecI16 get i16 => VecI16.fromList(this);
  VecI32 get i32 => VecI32.fromList(this);
  VecF32 get f32 => VecF32.fromList(map((e) => e.toDouble()).toList());
  VecF64 get f64 => VecF64.fromList(map((e) => e.toDouble()).toList());
}

extension ListFloatExtension on List<double> {
  VecF32 get f32 => VecF32.fromList(this);
  VecF64 get f64 => VecF64.fromList(this);
}
