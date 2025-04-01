import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;

class HalideType extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  HalideType.fromPointer(super.ptr, {super.attach, super.externalSize});

  factory HalideType.create({
    c.HalideTypeCode code = c.HalideTypeCode.halide_type_int,
    int bits = 0,
    int lanes = 0,
  }) {
    final p = c.mnn_halide_type_create_1(code, bits, lanes);
    return HalideType.fromPointer(p.cast());
  }

  int get bits => ptr.cast<c.halide_type_t>().ref.bits;
  int get lanes => ptr.cast<c.halide_type_t>().ref.lanes;
  c.HalideTypeCode get code => ptr.cast<c.halide_type_t>().ref.code;

  c.halide_type_t get ref => ptr.cast<c.halide_type_t>().ref;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    if (attach) {
      finalizer.detach(this);
    }
    c.mnn_halide_type_destroy(ptr.cast());
  }
}
