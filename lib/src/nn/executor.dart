import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:mnn/src/core/backend.dart';
import 'package:mnn/src/core/runtime_info.dart';

import '../core/base.dart';
import '../core/schedule.dart';
import '../core/session.dart';
import '../expr/expr.dart';
import '../g/mnn.g.dart' as C;

enum ExternalPathType {
  /// Path of the kvcache directory
  EXTERNAL_PATH_KVCACHE_DIR(0),

  /// Mid Buffer Cache File
  EXTERNAL_FEATUREMAP_DIR(1),

  /// Weight Buffer Cache File
  EXTERNAL_WEIGHT_DIR(2),

  /// Path of the NPU Model directory
  EXTERNAL_NPU_FILE_DIR(3)
  ;

  final int value;
  const ExternalPathType(this.value);

  factory ExternalPathType.fromValue(int value) => switch (value) {
    0 => EXTERNAL_PATH_KVCACHE_DIR,
    1 => EXTERNAL_FEATUREMAP_DIR,
    2 => EXTERNAL_WEIGHT_DIR,
    3 => EXTERNAL_NPU_FILE_DIR,
    _ => throw ArgumentError.value(value, 'value', 'Invalid ExternalPathType value'),
  };
}

enum LazyMode {
  LAZY_FULL(0),

  /// Don't compute content until user needed.
  LAZY_CONTENT(1 << 0),

  /// Expr can only compute once, it can reduce the create cost of expr
  LAZY_COMPUTE_ONCE(1 << 1)
  ;

  final int value;
  const LazyMode(this.value);

  factory LazyMode.fromValue(int value) => switch (value) {
    0 => LAZY_FULL,
    1 => LAZY_CONTENT,
    2 => LAZY_COMPUTE_ONCE,
    _ => throw ArgumentError.value(value, 'value', 'Invalid LazyMode value'),
  };
}

enum GCFlag {
  FULL(0),
  PART(1)
  ;

  final int value;
  const GCFlag(this.value);

  factory GCFlag.fromValue(int value) => switch (value) {
    0 => FULL,
    1 => PART,
    _ => throw ArgumentError.value(value, 'value', 'Invalid GCFlag value'),
  };
}

enum RuntimeStatus {
  ///get status whether this runtime support 16-bits float point arithmetic
  STATUS_SUPPORT_FP16(0),

  ///get status whether this runtime support dot-product arithmetic
  STATUS_SUPPORT_DOT_PRODUCT(1),

  ///get status whether this runtime support power-low (means low priority for opencl)
  STATUS_SUPPORT_POWER_LOW(2),

  ///emum total number
  STATUS_COUNT(3)
  ;

  final int value;
  const RuntimeStatus(this.value);

  factory RuntimeStatus.fromValue(int value) => switch (value) {
    0 => STATUS_SUPPORT_FP16,
    1 => STATUS_SUPPORT_DOT_PRODUCT,
    2 => STATUS_SUPPORT_POWER_LOW,
    3 => STATUS_COUNT,
    _ => throw ArgumentError.value(value, 'value', 'Invalid RuntimeStatus value'),
  };
}

// Classes
class Executor extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_executor_destroy);
  Executor.fromPointer(C.mnn_executor_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory Executor.create(ForwardType type, BackendConfig config, int numThreads) {
    final p = C.mnn_executor_static_new_executor(type.value, config.ref, numThreads);
    return Executor.fromPointer(p);
  }

  // ignore: prefer_constructors_over_static_methods
  static Executor get global => Executor.fromPointer(C.mnn_executor_static_get_global_executor());

  LazyMode get lazyMode => LazyMode.fromValue(C.mnn_executor_get_lazy_mode(ptr));
  set lazyMode(LazyMode value) => C.mnn_executor_set_lazy_mode(ptr, value.value);

  bool get lazyEval => C.mnn_executor_get_lazyEval(ptr);
  set lazyEval(bool value) => C.mnn_executor_set_lazyEval(ptr, value);

  void setGlobalExecutorConfig(ForwardType type, BackendConfig config, int numThreads) {
    C.mnn_executor_set_global_executor_config(ptr, type.value, config.ref, numThreads);
  }

  int getCurrentRuntimeStatus(RuntimeStatus status) =>
      C.mnn_executor_get_current_runtime_status(ptr, status.value);

  void gc(GCFlag flag) => C.mnn_executor_gc(ptr, flag.value);

  /// Get runtime info
  static RuntimeInfo getRuntimeInfo() => RuntimeInfo.fromPointer(C.mnn_executor_static_get_runtime());

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    C.mnn_executor_destroy(ptr);
  }

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return "Executor(address=0x${ptr.address.toRadixString(16)})";
  }
}

class ExecutorScope extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_executor_scope_destroy);
  ExecutorScope.fromPointer(C.mnn_executor_scope_t ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory ExecutorScope.create(Executor executor, {String? name}) {
    if (name == null) {
      return ExecutorScope.fromPointer(C.mnn_executor_scope_create(executor.ptr));
    }
    final cName = name.toNativeUtf8().cast<ffi.Char>();
    try {
      return ExecutorScope.fromPointer(C.mnn_executor_scope_create_with_name(cName, executor.ptr));
    } finally {
      calloc.free(cName);
    }
  }

  static Executor current() => Executor.fromPointer(C.mnn_executor_scope_static_current_executor());

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    C.mnn_executor_scope_destroy(ptr);
  }

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return "ExecutorScope(address=0x${ptr.address.toRadixString(16)})";
  }
}

class RuntimeManager extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_runtime_manager_destroy);
  RuntimeManager.fromPointer(C.mnn_runtime_manager_t ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory RuntimeManager.create([ScheduleConfig? config]) {
    config ??= ScheduleConfig.create();
    return RuntimeManager.fromPointer(C.mnn_runtime_manager_create(config.ref));
  }

  void setCache(String path) {
    final cPath = path.toNativeUtf8();
    try {
      C.mnn_runtime_manager_set_cache(ptr, cPath.cast());
    } finally {
      calloc.free(cPath);
    }
  }

  void setExternalPath(String path, ExternalPathType type) {
    final cPath = path.toNativeUtf8();
    try {
      C.mnn_runtime_manager_set_external_path(ptr, cPath.cast(), type.value);
    } finally {
      calloc.free(cPath);
    }
  }

  void setExternalFile(String path) {
    final cPath = path.toNativeUtf8();
    try {
      C.mnn_runtime_manager_set_external_file(ptr, cPath.cast());
    } finally {
      calloc.free(cPath);
    }
  }

  void updateCache() {
    C.mnn_runtime_manager_update_cache(ptr);
  }

  bool isBackendSupport(ForwardType type) {
    return C.mnn_runtime_manager_is_backend_support(ptr, type.value);
  }

  void setMode(SessionMode mode) {
    C.mnn_runtime_manager_set_mode(ptr, mode.value);
  }

  void setHint(HintMode mode, int value) {
    C.mnn_runtime_manager_set_hint(ptr, mode.value, value);
  }

  double get infoMemory {
    final p = calloc<ffi.Float>();
    final success = C.mnn_runtime_manager_get_info(ptr, SessionInfoCode.MEMORY.value, p.cast());
    final memory = success ? p.value : 0.0;
    calloc.free(p);
    return memory;
  }

  double get infoFLOPs {
    final p = calloc<ffi.Double>();
    final success = C.mnn_runtime_manager_get_info(ptr, SessionInfoCode.FLOPS.value, p.cast());
    final flops = success ? p.value : 0.0;
    calloc.free(p);
    return flops;
  }

  static (bool success, String info) getDeviceInfo(String deviceKey, ForwardType type) {
    final cDeviceKey = deviceKey.toNativeUtf8();
    final pInfo = calloc<ffi.Pointer<ffi.Char>>(1);
    try {
      final success = C.mnn_runtime_manager_static_get_device_info(cDeviceKey.cast(), type.value, pInfo);
      final info = pInfo.value != ffi.nullptr ? pInfo.value.cast<Utf8>().toDartString() : "";
      return (success, info);
    } finally {
      calloc.free(cDeviceKey);
      calloc.free(pInfo);
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    C.mnn_runtime_manager_destroy(ptr);
  }

  @override
  List<Object?> get props => [ptr.address];
}

class ModuleInfo extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_module_info_destroy);

  ModuleInfo.fromPointer(C.mnn_module_info_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory ModuleInfo.create() {
    return ModuleInfo.fromPointer(C.mnn_module_info_create());
  }

  int get inputsLength => C.mnn_module_info_get_inputs_length(ptr);

  VariableInfo getInput(int index) {
    MnnAssert(index >= 0 && index < inputsLength, "index out of range");
    return VariableInfo.fromPointer(C.mnn_module_info_get_inputs_at(ptr, index));
  }

  DimensionFormat defaultFormat() {
    return DimensionFormat.fromValue(C.mnn_module_info_get_default_format(ptr));
  }

  List<String> getInputNames() {
    final p = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final length = C.mnn_module_info_get_input_names(ptr, p);
    final rval = List.generate(length, (i) {
      final s = p.value[i];
      return s == ffi.nullptr ? "" : s.cast<Utf8>().toDartString();
    });
    calloc.free(p);
    return rval;
  }

  List<String> getOutputNames() {
    final p = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final length = C.mnn_module_info_get_output_names(ptr, p);
    final rval = List.generate(length, (i) {
      final s = p.value[i];
      return s == ffi.nullptr ? "" : s.cast<Utf8>().toDartString();
    });
    calloc.free(p);
    return rval;
  }

  String get version {
    final p = C.mnn_module_info_get_version(ptr);
    return p == ffi.nullptr ? "" : p.cast<Utf8>().toDartString();
  }

  String get bizCode {
    final p = C.mnn_module_info_get_bizCode(ptr);
    return p == ffi.nullptr ? "" : p.cast<Utf8>().toDartString();
  }

  Map<String, String> get metadata {
    final pKeys = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pValues = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    try {
      final length = C.mnn_module_info_get_metadata(ptr, pKeys, pValues);
      final rval = Map<String, String>.fromEntries(
        List.generate(
          length,
          (i) {
            final k = pKeys.value[i];
            final v = pValues.value[i];
            return MapEntry(
              k == ffi.nullptr ? "" : k.cast<Utf8>().toDartString(),
              v == ffi.nullptr ? "" : v.cast<Utf8>().toDartString(),
            );
          },
        ),
      );
      return rval;
    } finally {
      calloc.free(pKeys);
      calloc.free(pValues);
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    C.mnn_module_info_destroy(ptr);
  }
}

/// Execute a computation using an executor
///
/// TODO: support this
/// NOTE: DO NOT USE!!!
///
/// Note: remember to dispose the module in [computation]
///
/// NOTE: Our implementation uses `attach` to attach a native object to a Dart object,
/// which means in most cases, native objects will be disposed automatically when Dart objects
/// are garbage collected. **However**, when they will be disposed is not guaranteed,
/// if they are freed before the lazy computation is executed, a use-after-free may occur.
///
/// - [computation] The computation to execute, which takes an executor as input and returns a result
///
/// - [type] The forward type of the executor, default is MNN_FORWARD_CPU
///
/// - [config] The backend config of the executor, default is null
///
/// - [numThreads] The number of threads to use for the executor, default is the number of processors
///
/// return the result of the computation
T usingExecutor<T>(
  T Function(Executor executor) computation, {
  ForwardType type = ForwardType.MNN_FORWARD_CPU,
  BackendConfig? config,
  int? numThreads,
}) {
  config ??= BackendConfig.create();
  numThreads ??= Platform.numberOfProcessors;
  // var isAsync = false;
  final executor = Executor.create(type, config, numThreads);
  executor.lazyMode = LazyMode.LAZY_FULL;
  final scope = ExecutorScope.create(executor);

  try {
    final result = computation(executor);
    if (result is Future) {
      // isAsync = true;
      return result.whenComplete(scope.dispose) as T;
    }
    return result;
  } finally {
    scope.dispose();
  }
}
