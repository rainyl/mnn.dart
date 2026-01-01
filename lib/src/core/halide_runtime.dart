/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import '../g/mnn.g.dart' as c;
import 'base.dart';

class HalideType with ComparableMixin {
  final c.HalideTypeCode code;
  final int bits;
  final int lanes;

  const HalideType({this.code = c.HalideTypeCode.halide_type_int, this.bits = 0, this.lanes = 1});

  factory HalideType.fromNative(c.halide_type_c_t type) =>
      HalideType(code: c.HalideTypeCode.fromValue(type.code), bits: type.bits, lanes: type.lanes);

  static const bf16 = HalideType(code: c.HalideTypeCode.halide_type_bfloat, bits: 16);
  static const f32 = HalideType(code: c.HalideTypeCode.halide_type_float, bits: 32);
  static const f64 = HalideType(code: c.HalideTypeCode.halide_type_float, bits: 64);
  static const bool_ = HalideType(code: c.HalideTypeCode.halide_type_uint, bits: 1);
  static const u8 = HalideType(code: c.HalideTypeCode.halide_type_uint, bits: 8);
  static const u16 = HalideType(code: c.HalideTypeCode.halide_type_uint, bits: 16);
  static const u32 = HalideType(code: c.HalideTypeCode.halide_type_uint, bits: 32);
  static const u64 = HalideType(code: c.HalideTypeCode.halide_type_uint, bits: 64);
  static const i8 = HalideType(code: c.HalideTypeCode.halide_type_int, bits: 8);
  static const i16 = HalideType(code: c.HalideTypeCode.halide_type_int, bits: 16);
  static const i32 = HalideType(code: c.HalideTypeCode.halide_type_int, bits: 32);
  static const i64 = HalideType(code: c.HalideTypeCode.halide_type_int, bits: 64);

  static HalideType of<T extends ffi.SizedNativeType>() {
    if (T == ffi.Float) {
      return HalideType.f32;
    } else if (T == ffi.Double) {
      return HalideType.f64;
    } else if (T == ffi.Bool) {
      return HalideType.bool_;
    } else if (T == ffi.Uint8) {
      return HalideType.u8;
    } else if (T == ffi.Uint16) {
      return HalideType.u16;
    } else if (T == ffi.Uint32) {
      return HalideType.u32;
    } else if (T == ffi.Uint64) {
      return HalideType.u64;
    } else if (T == ffi.Int8) {
      return HalideType.i8;
    } else if (T == ffi.Int16) {
      return HalideType.i16;
    } else if (T == ffi.Int32) {
      return HalideType.i32;
    } else if (T == ffi.Int64) {
      return HalideType.i64;
    } else {
      throw ArgumentError.value(T, 'T', 'Unsupported type');
    }
  }

  ffi.Pointer<ffi.Void> pointerOf(Iterable<num> data) {
    final ffi.Pointer pdata = switch ((code, bits)) {
      (c.HalideTypeCode.halide_type_uint, 8) => calloc<uint8>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_uint, 16) => calloc<uint16>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_uint, 32) => calloc<uint32>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_uint, 64) => calloc<uint64>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_int, 8) => calloc<int8>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_int, 16) => calloc<int16>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_int, 32) => calloc<int32>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_int, 64) => calloc<int64>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toInt())),
      (c.HalideTypeCode.halide_type_float, 32) => calloc<float32>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toDouble())),
      (c.HalideTypeCode.halide_type_float, 64) => calloc<float64>(
        data.length,
      )..asTypedList(data.length).setAll(0, data.map((e) => e.toDouble())),
      _ => throw ArgumentError.value(this, 'T', 'Unsupported type'),
    };
    return pdata.cast<ffi.Void>();
  }

  /// Size in bytes for a single element, even if width is not 1, of this type.
  int get bytes => (bits + 7) ~/ 8;

  bool get isInt => [c.HalideTypeCode.halide_type_int, c.HalideTypeCode.halide_type_uint].contains(code);
  bool get isFloat =>
      [c.HalideTypeCode.halide_type_float, c.HalideTypeCode.halide_type_bfloat].contains(code);

  _HalideTypeNative get native => _HalideTypeNative(this);

  @override
  List<Object?> get props => [bits, lanes, code];

  @override
  String toString() {
    return switch (this) {
      HalideType.bf16 => 'bfloat16',
      HalideType.f32 => 'float32',
      HalideType.f64 => 'float64',
      HalideType.bool_ => 'bool',
      HalideType.u8 => 'uint8',
      HalideType.u16 => 'uint16',
      HalideType.u32 => 'uint32',
      HalideType.u64 => 'uint64',
      HalideType.i8 => 'int8',
      HalideType.i16 => 'int16',
      HalideType.i32 => 'int32',
      HalideType.i64 => 'int64',
      _ => 'HalideType(code=$code, bits=$bits, lanes=$lanes)',
    };
  }
}

class _HalideTypeNative extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  _HalideTypeNative.fromPointer(ffi.Pointer<c.halide_type_c_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory _HalideTypeNative(HalideType type) {
    final ptr = calloc<c.halide_type_c_t>()
      ..ref.code = type.code.value
      ..ref.bits = type.bits
      ..ref.lanes = type.lanes;
    return _HalideTypeNative.fromPointer(ptr, attach: true, externalSize: ffi.sizeOf<c.halide_type_c_t>());
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  c.halide_type_c_t get ref => ptr.cast<c.halide_type_c_t>().ref;

  @override
  List<Object?> get props => [ref.code, ref.bits, ref.lanes];

  @override
  void release() {
    calloc.free(ptr);
  }
}

class HalideBuffer extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  HalideBuffer.fromPointer(ffi.Pointer<c.halide_buffer_c_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  int get devide => ref.device;

  // The interface used to interpret the above handle.
  // Not used
  // external ffi.Pointer<halide_device_interface_t> device_interface;

  ffi.Pointer<ffi.Uint8> get host => ref.host;
  int get flags => ref.flags;
  HalideType get type {
    return HalideType.fromNative(ref.type);
  }

  int get dimensions => ref.dimensions;

  List<HalideDimension> get dim => List<HalideDimension>.generate(
    dimensions,
    (index) => HalideDimension.fromPointer(ref.dim + index, attach: false),
  );

  // Not used
  // ffi.Pointer<ffi.Void> get padding => ref.padding;

  c.halide_buffer_c_t get ref => ptr.cast<c.halide_buffer_c_t>().ref;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    calloc.free(ptr);
  }

  @override
  List<Object?> get props => [ptr.address];
  @override
  String toString() {
    return 'HalideBuffer('
        'devide=$devide, '
        'host=0x${host.address.toRadixString(16)}, '
        'flags=$flags, '
        'type=$type, '
        'dimensions=$dimensions, '
        ')';
  }
}

class HalideDimension extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  HalideDimension.fromPointer(ffi.Pointer<c.halide_dimension_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  c.halide_dimension_t get ref => ptr.cast<c.halide_dimension_t>().ref;

  int get min => ref.min;
  int get extent => ref.extent;
  int get stride => ref.stride;
  int get flags => ref.flags;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    calloc.free(ptr);
  }

  @override
  List<Object?> get props => [min, extent, stride, flags];
  @override
  String toString() {
    return 'HalideDimension(min=$min, extent=$extent, stride=$stride, flags=$flags)';
  }
}
