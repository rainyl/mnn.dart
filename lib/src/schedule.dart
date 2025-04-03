import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'backend.dart';
import 'base.dart';
import 'g/mnn.g.dart' as c;

class ScheduleConfig extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ScheduleConfig.fromPointer(super.ptr, {super.attach, super.externalSize});

  factory ScheduleConfig.create({
    ForwardType type = ForwardType.MNN_FORWARD_CPU,
    int numThread = 4,
    int mode = 0,
    BackendConfig? backendConfig,
  }) {
    final p = calloc<c.mnn_schedule_config_t>()
      ..ref.type = type.value
      ..ref.unnamed.num_thread = numThread
      ..ref.unnamed.mode = mode
      ..ref.backend_config = backendConfig == null ? ffi.nullptr : backendConfig.ptr.cast();
    return ScheduleConfig.fromPointer(p.cast());
  }

  c.mnn_schedule_config_t get ref => ptr.cast<c.mnn_schedule_config_t>().ref;

  int get type => ref.type;

  int get numThread => ref.unnamed.num_thread;
  set numThread(int value) {
    ref.unnamed.num_thread = value;
  }

  int get mode => ref.unnamed.mode;
  set mode(int value) {
    ref.unnamed.mode = value;
  }

  BackendConfig get backendConfig => BackendConfig.fromPointer(ref.backend_config, attach: false);

  set backendConfig(BackendConfig value) {
    ref.backend_config = value.ptr.cast();
  }

  @override
  void release() {
    if (ptr != ffi.nullptr) {
      if (ref.backend_config != ffi.nullptr) {
        calloc.free(ref.backend_config);
      }
      calloc.free(ptr);
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'ScheduleConfig(address=0x${ptr.address})';
  }
}

enum ForwardType {
  MNN_FORWARD_CPU(0),

  /// Firtly find the first available backends not equal to CPU
  /// If no other backends, use cpu
  MNN_FORWARD_AUTO(4),

  /// Hand write metal
  MNN_FORWARD_METAL(1),

  /// NVIDIA GPU API
  MNN_FORWARD_CUDA(2),

  /// Android / Common Device GPU API
  MNN_FORWARD_OPENCL(3),
  MNN_FORWARD_OPENGL(6),
  MNN_FORWARD_VULKAN(7),

  /// Android 8.1's NNAPI or CoreML for ios
  MNN_FORWARD_NN(5),

  /// User can use API from Backend.hpp to add or search Backend
  MNN_FORWARD_USER_0(8),
  MNN_FORWARD_USER_1(9),
  MNN_FORWARD_USER_2(10),
  MNN_FORWARD_USER_3(11),

  MNN_FORWARD_ALL(12),

  /// Apply arm extension instruction set to accelerate some Ops, this forward
  /// type is only used in MNN internal, and will be active automatically when
  /// user set forward type to be MNN_FORWARD_CPU and extension instruction set
  /// is valid on hardware.
  MNN_FORWARD_CPU_EXTENSION(13),

  /// use for shared memory on android device
  MNN_MEMORY_AHARDWAREBUFFER(14);

  final int value;
  const ForwardType(this.value);

  static ForwardType fromValue(int value) => switch (value) {
        0 => MNN_FORWARD_CPU,
        4 => MNN_FORWARD_AUTO,
        1 => MNN_FORWARD_METAL,
        2 => MNN_FORWARD_CUDA,
        3 => MNN_FORWARD_OPENCL,
        6 => MNN_FORWARD_OPENGL,
        7 => MNN_FORWARD_VULKAN,
        5 => MNN_FORWARD_NN,
        8 => MNN_FORWARD_USER_0,
        9 => MNN_FORWARD_USER_1,
        10 => MNN_FORWARD_USER_2,
        11 => MNN_FORWARD_USER_3,
        12 => MNN_FORWARD_ALL,
        13 => MNN_FORWARD_CPU_EXTENSION,
        14 => MNN_MEMORY_AHARDWAREBUFFER,
        _ => throw ArgumentError('Unknown value for ForwardType: $value'),
      };
}

// mnn_forward_type;
