import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;

class HalideType extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  HalideType.fromPointer(ffi.Pointer<c.halide_type_t> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  factory HalideType.create({
    c.HalideTypeCode code = c.HalideTypeCode.halide_type_int,
    int bits = 0,
    int lanes = 1,
  }) {
    final p = c.mnn_halide_type_create_1(code, bits, lanes);
    return HalideType.fromPointer(p.cast());
  }

  factory HalideType.fromNative(c.halide_type_t type) {
    final p = calloc<c.halide_type_t>();
    p.cast<c.halide_type_t>().ref = type;
    return HalideType.fromPointer(p.cast());
  }

  factory HalideType.bf16() => HalideType.create(code: c.HalideTypeCode.halide_type_bfloat, bits: 16);
  factory HalideType.f32() => HalideType.create(code: c.HalideTypeCode.halide_type_float, bits: 32);
  factory HalideType.f64() => HalideType.create(code: c.HalideTypeCode.halide_type_float, bits: 64);
  factory HalideType.bool() => HalideType.create(code: c.HalideTypeCode.halide_type_uint, bits: 1);
  factory HalideType.u8() => HalideType.create(code: c.HalideTypeCode.halide_type_uint, bits: 8);
  factory HalideType.u16() => HalideType.create(code: c.HalideTypeCode.halide_type_uint, bits: 16);
  factory HalideType.u32() => HalideType.create(code: c.HalideTypeCode.halide_type_uint, bits: 32);
  factory HalideType.u64() => HalideType.create(code: c.HalideTypeCode.halide_type_uint, bits: 64);
  factory HalideType.i8() => HalideType.create(code: c.HalideTypeCode.halide_type_int, bits: 8);
  factory HalideType.i16() => HalideType.create(code: c.HalideTypeCode.halide_type_int, bits: 16);
  factory HalideType.i32() => HalideType.create(code: c.HalideTypeCode.halide_type_int, bits: 32);
  factory HalideType.i64() => HalideType.create(code: c.HalideTypeCode.halide_type_int, bits: 64);

  int get bits => ref.bits;
  int get lanes => ref.lanes;
  int get code => ref.code;

  /// Size in bytes for a single element, even if width is not 1, of this type.
  int get bytes => (bits + 7) ~/ 8;

  c.halide_type_t get ref => ptr.cast<c.halide_type_t>().ref;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    c.mnn_halide_type_destroy(ptr.cast());
  }

  @override
  List<Object?> get props => [bits, lanes, code];

  @override
  String toString() {
    return 'HalideType(code=$code, bits=$bits, lanes=$lanes)';
  }
}

class HalideBuffer extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  HalideBuffer.fromPointer(ffi.Pointer<c.halide_buffer_t> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  int get devide => ref.device;

  /// The interface used to interpret the above handle.
  // TODO: external ffi.Pointer<halide_device_interface_t> device_interface;

  ffi.Pointer<ffi.Uint8> get host => ref.host;
  int get flags => ref.flags;
  HalideType get type => HalideType.fromNative(ref.type);
  int get dimensions => ref.dimensions;

  List<HalideDimension> get dim => List<HalideDimension>.generate(
        dimensions,
        (index) => HalideDimension.fromPointer(ref.dim + index, attach: false),
      );

  // Not used
  // ffi.Pointer<ffi.Void> get padding => ref.padding;

  c.halide_buffer_t get ref => ptr.cast<c.halide_buffer_t>().ref;

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
