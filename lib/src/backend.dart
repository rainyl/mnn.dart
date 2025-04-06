import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'constant.dart';
import 'g/mnn.g.dart' as c;

class BackendConfig extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  BackendConfig.fromPointer(ffi.Pointer<c.mnn_backend_config_t> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  factory BackendConfig.create({
    int memory = MNN_MEMORY_NORMAL,
    int power = MNN_POWER_NORMAL,
    int precision = MNN_PRECISION_NORMAL,
  }) {
    final p = calloc<c.mnn_backend_config_t>()
      ..ref.memory = memory
      ..ref.power = power
      ..ref.precision = precision;
    return BackendConfig.fromPointer(p);
  }

  c.mnn_backend_config_t get ref => ptr.cast<c.mnn_backend_config_t>().ref;

  int get memory => ref.memory;
  set memory(int value) => ref.memory = value;

  int get power => ref.power;
  set power(int value) => ref.power = value;

  int get precision => ref.precision;
  set precision(int value) => ref.precision = value;

  @override
  void release() {
    if (ptr != ffi.nullptr) {
      calloc.free(ptr);
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'BackendConfig(address=0x${ptr.address})';
  }
}
