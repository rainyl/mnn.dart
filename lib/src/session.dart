import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;
import 'tensor.dart';

class Session with ComparableMixin {
  // Managed by Interpreter
  final c.mnn_session_t ptr;
  final c.mnn_interpreter_t interpreterPtr;

  Session.fromPointer(this.ptr, this.interpreterPtr);

  /// @brief Get input tensor by name
  ///
  /// @param name Tensor name (NULL for first input)
  ///
  /// @return Tensor instance or NULL if failed
  Tensor? getInput({String? name}) {
    final p = name != null ? name.toNativeUtf8().cast<ffi.Char>() : ffi.nullptr;
    final tensor = c.mnn_interpreter_get_session_input(interpreterPtr, ptr, p);
    if (p != ffi.nullptr) calloc.free(p);
    return tensor == ffi.nullptr ? null : Tensor.fromPointer(tensor, attach: false);
  }

  Tensor? getOutput({String? name}) {
    final p = name != null ? name.toNativeUtf8().cast<ffi.Char>() : ffi.nullptr;
    final tensor = c.mnn_interpreter_get_session_output(interpreterPtr, ptr, p);
    if (p != ffi.nullptr) calloc.free(p);
    return tensor == ffi.nullptr ? null : Tensor.fromPointer(tensor, attach: false);
  }

  Map<String, Tensor> getInputAll() {
    final pCount = calloc<ffi.Size>();
    final pNames = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pTensors = calloc<ffi.Pointer<c.mnn_tensor_t>>();
    final res = c.mnn_interpreter_get_session_input_all(interpreterPtr, ptr, pTensors, pNames, pCount);
    final rval = <String, Tensor>{};
    if (res == c.ErrorCode.NO_ERROR) {
      final count = pCount.value;
      for (var i = 0; i < count; i++) {
        final name = pNames.value[i].cast<Utf8>().toDartString();
        final tensor = pTensors.value[i];
        rval[name] = Tensor.fromPointer(tensor, attach: false);
      }
    }
    calloc.free(pCount);
    calloc.free(pNames);
    calloc.free(pTensors);
    return rval;
  }

  Map<String, Tensor> getOutputAll() {
    final pCount = calloc<ffi.Size>();
    final pNames = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pTensors = calloc<ffi.Pointer<c.mnn_tensor_t>>();
    final res = c.mnn_interpreter_get_session_output_all(interpreterPtr, ptr, pTensors, pNames, pCount);
    final rval = <String, Tensor>{};
    if (res == c.ErrorCode.NO_ERROR) {
      final count = pCount.value;
      for (var i = 0; i < count; i++) {
        final name = pNames.value[i].cast<Utf8>().toDartString();
        final tensor = pTensors.value[i];
        rval[name] = Tensor.fromPointer(tensor, attach: false);
      }
    }
    calloc.free(pCount);
    calloc.free(pNames);
    calloc.free(pTensors);
    return rval;
  }

  c.ErrorCode run() {
    return c.mnn_interpreter_run_session(interpreterPtr, ptr, ffi.nullptr);
  }

  void setHint(HintMode mode, int value) {
    c.mnn_interpreter_set_session_hint(ptr, mode.value, value);
  }

  c.ErrorCode updateSessionToModel() {
    return c.mnn_interpreter_update_session_to_model(interpreterPtr, ptr);
  }

  bool get isEmpty => ptr == ffi.nullptr;

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'Session(address=0x${ptr.address})';
  }
}

/// Hint mode enum
enum HintMode {
  /// Max Op number for async tuning
  MAX_TUNING_NUMBER(0),

  /// Strictly check model file or not, default 1. if set 0, will not check model file valid/invalid
  STRICT_CHECK_MODEL(1),

  /// Memory allocator type
  MEM_ALLOCATOR_TYPE(2),

  /// Winograd unit candidates count, default 3. if set 0, will use less unit candidates for less memory at the expense of performance.
  WINOGRAD_MEMORY_LEVEL(3),

  /// Geometry Compute option, default is 0xFFFF
  GEOMETRY_COMPUTE_MASK(4),

  /// 0: Close dynamic quant;
  /// 1: For general convolution, use one scale&zeropoint to quant.
  DYNAMIC_QUANT_OPTIONS(5),

  /// For Mobile CPU with big-litter core, set decrease rate to let MNN divide task differential by CPU's performance
  /// 0-100, 50 means litter core has 50% capacity of large core
  /// Default is 50
  CPU_LITTLECORE_DECREASE_RATE(6),

  /// 0: Do not quantize
  /// 1: Only quantize key, use int8 asymmetric quantization
  /// 2: Only quantize value, use fp8 quantization
  /// 3: quantize both key and value
  /// 4: quantize query, key and value, and use gemm int8 kernel to compute K*V
  QKV_QUANT_OPTIONS(7),

  /// size limit of kvcache in memory (for a single layer)
  /// if the size of kvcache exceeds the limit, it will be moved to disk
  KVCACHE_SIZE_LIMIT(8),

  /// Op encoder number for commit
  OP_ENCODER_NUMBER_FOR_COMMIT(9),

  /// KVCache Info
  KVCACHE_INFO(10),

  /// mmap allocate file size, KB
  MMAP_FILE_SIZE(11),

  /// Use cached mmap
  USE_CACHED_MMAP(12),

  /// Multi-Thread Load module, default is 0 (don't use other Thread)
  INIT_THREAD_NUMBER(13);

  final int value;
  const HintMode(this.value);

  static HintMode fromValue(int value) => switch (value) {
        0 => MAX_TUNING_NUMBER,
        1 => STRICT_CHECK_MODEL,
        2 => MEM_ALLOCATOR_TYPE,
        3 => WINOGRAD_MEMORY_LEVEL,
        4 => GEOMETRY_COMPUTE_MASK,
        5 => DYNAMIC_QUANT_OPTIONS,
        6 => CPU_LITTLECORE_DECREASE_RATE,
        7 => QKV_QUANT_OPTIONS,
        8 => KVCACHE_SIZE_LIMIT,
        9 => OP_ENCODER_NUMBER_FOR_COMMIT,
        10 => KVCACHE_INFO,
        11 => MMAP_FILE_SIZE,
        12 => USE_CACHED_MMAP,
        13 => INIT_THREAD_NUMBER,
        _ => throw ArgumentError('Unknown value for HintMode: $value'),
      };
}
