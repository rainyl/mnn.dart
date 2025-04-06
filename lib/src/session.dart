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
    try {
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
      return rval;
    } finally {
      calloc.free(pCount);
      calloc.free(pNames);
      calloc.free(pTensors);
    }
  }

  Map<String, Tensor> getOutputAll() {
    final pCount = calloc<ffi.Size>();
    final pNames = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pTensors = calloc<ffi.Pointer<c.mnn_tensor_t>>();
    try {
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
      return rval;
    } finally {
      calloc.free(pCount);
      calloc.free(pNames);
      calloc.free(pTensors);
    }
  }

  void run() {
    final code = c.mnn_interpreter_run_session(interpreterPtr, ptr, ffi.nullptr);
    if (code != c.ErrorCode.NO_ERROR) {
      throw Exception("runSession failed: $code");
    }
  }

  Future<void> runAsync() async {
    return mnnRunAsync0(
      (callback) => c.mnn_interpreter_run_session(interpreterPtr, ptr, callback),
      (c) => c.complete(),
    );
  }

  void setHint(HintMode mode, int value) {
    c.mnn_interpreter_set_session_hint(ptr, mode.value, value);
  }

  c.ErrorCode updateSessionToModel() {
    return c.mnn_interpreter_update_session_to_model(interpreterPtr, ptr);
  }

  double get memoryInfo {
    final p = calloc<f32>();
    try {
      final code =
          c.mnn_interpreter_get_session_info(interpreterPtr, ptr, SessionInfoCode.MEMORY.value, p.cast());
      final rval = code == c.ErrorCode.NO_ERROR ? p.value : -1.0;
      return rval;
    } finally {
      calloc.free(p);
    }
  }

  double get flopsInfo {
    final p = calloc<f32>();
    try {
      final code =
          c.mnn_interpreter_get_session_info(interpreterPtr, ptr, SessionInfoCode.FLOPS.value, p.cast());
      final rval = code == c.ErrorCode.NO_ERROR ? p.value : -1.0;
      return rval;
    } finally {
      calloc.free(p);
    }
  }

  List<int> get backendsInfo {
    final p = calloc<i32>(2);
    try {
      final code =
          c.mnn_interpreter_get_session_info(interpreterPtr, ptr, SessionInfoCode.BACKENDS.value, p.cast());
      final rval = code == c.ErrorCode.NO_ERROR ? [p[0], p[1]] : <int>[];
      return rval;
    } finally {
      calloc.free(p);
    }
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

enum SessionMode {
  /// About CallBack, Default Session_Debug
  /// runSessionWithCallBack is allowed and can get internal op info
  Session_Debug(0),

  /// runSessionWithCallBack is not valid and can't get any info of op in session
  Session_Release(1),

  /// About input tenosr, Default Session_Input_Inside
  /// The input tensor is alloced by session, input data after session resized
  Session_Input_Inside(2),

  /// The input tensor is alloced by user, set input data before session resize
  Session_Input_User(3),

  /// The output tensor depends on session, and can't be separate used
  Session_Output_Inside(4),

  /// The output tensor can be separated from session
  Session_Output_User(5),

  /// Try Resize Session when create Session or not, default direct:
  Session_Resize_Direct(6),
  Session_Resize_Defer(7),

  /// Determine the Execution's forward type is determine by user or auto determine
  Session_Backend_Fix(8), // Use the backend user set, when not support use default backend
  Session_Backend_Auto(9), // Auto Determine the Op type by MNN

  /// Determine static memory whether recyle in resizeSession or just cache the memory
  /// Recycle static memory when session resize in case memory explosion
  Session_Memory_Collect(10),

  /// Cache the static memory for next forward usage
  Session_Memory_Cache(11),

  /// Determine whether use codegen function
  /// Disable codegen in case extra build codegen cost
  Session_Codegen_Disable(12),

  /// Enable codegen
  Session_Codegen_Enable(13),

  /// Dynamic Reisze Optimization
  /// Open Trace for resize
  Session_Resize_Check(14),

  /// Apply Resize Optimization
  Session_Resize_Fix(15),

  /// Set for Module's traceOrOptimize API.
  ///  Module_Forward_Seperate:
  ///  when inputs is not empty , Module's onForward will only infer shape and alloc memory.
  ///  when inputs is empty , Module's onForward will only runSession to compute content.
  ///  Default is Module_Forward_Combine

  Module_Forward_Separate(16),
  Module_Forward_Combine(17);

  final int value;
  const SessionMode(this.value);

  static SessionMode fromValue(int value) => switch (value) {
        0 => Session_Debug,
        1 => Session_Release,
        2 => Session_Input_Inside,
        3 => Session_Input_User,
        4 => Session_Output_Inside,
        5 => Session_Output_User,
        6 => Session_Resize_Direct,
        7 => Session_Resize_Defer,
        8 => Session_Backend_Fix,
        9 => Session_Backend_Auto,
        10 => Session_Memory_Collect,
        11 => Session_Memory_Cache,
        12 => Session_Codegen_Disable,
        13 => Session_Codegen_Enable,
        14 => Session_Resize_Check,
        15 => Session_Resize_Fix,
        16 => Module_Forward_Separate,
        17 => Module_Forward_Combine,
        _ => throw ArgumentError('Unknown value for SessionMode: $value'),
      };
}

enum SessionInfoCode {
  /// memory session used in MB, float*
  MEMORY(0),

  /// float operation needed in session in M, float* */
  FLOPS(1),

  /// Backends in session in M, int*, length >= 1 + number of configs when create session */
  BACKENDS(2),

  /// Resize Info, int* , the mean different from API
  /// Interpreter::getSessionInfo: 0: ready to execute, 1: need malloc, 2: need resize
  /// RuntimeManager::getInfo: 0: no resize, 1: re-malloc, 2: resize
  RESIZE_STATUS(3),

  /// Mode / NumberThread, int* */
  THREAD_NUMBER(4);

  // ALL

  final int value;
  const SessionInfoCode(this.value);

  static SessionInfoCode fromValue(int value) => switch (value) {
        0 => MEMORY,
        1 => FLOPS,
        2 => BACKENDS,
        3 => RESIZE_STATUS,
        4 => THREAD_NUMBER,
        _ => throw ArgumentError('Unknown value for SessionInfoCode: $value'),
      };
}
