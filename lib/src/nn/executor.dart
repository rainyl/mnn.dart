import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

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
}

// Classes
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

  // TODO
  // bool getInfo(Interpreter::SessionInfoCode code, void* ptr);

  (bool success, String info) getDeviceInfo(String deviceKey, ForwardType type) {
    final cDeviceKey = deviceKey.toNativeUtf8();
    try {
      final pInfo = calloc<ffi.Pointer<ffi.Char>>(1);
      final success = C.mnn_runtime_manager_static_get_device_info(cDeviceKey.cast(), type.value, pInfo);
      final info = pInfo.value.cast<Utf8>().toDartString();
      return (success, info);
    } finally {
      calloc.free(cDeviceKey);
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
    final p = calloc<ffi.Pointer<ffi.Char>>();
    final length = C.mnn_module_info_get_input_names(ptr, p);
    final rval = List.generate(length, (i) => p[i].cast<Utf8>().toDartString());
    calloc.free(p);
    return rval;
  }

  List<String> getOutputNames() {
    final p = calloc<ffi.Pointer<ffi.Char>>();
    final length = C.mnn_module_info_get_output_names(ptr, p);
    final rval = List.generate(length, (i) => p[i].cast<Utf8>().toDartString());
    calloc.free(p);
    return rval;
  }

  String get version => C.mnn_module_info_get_version(ptr).cast<Utf8>().toDartString();

  String get bizCode => C.mnn_module_info_get_bizCode(ptr).cast<Utf8>().toDartString();

  Map<String, String> get metadata {
    final pKeys = calloc<ffi.Pointer<ffi.Char>>();
    final pValues = calloc<ffi.Pointer<ffi.Char>>();
    try {
      final length = C.mnn_module_info_get_metadata(ptr, pKeys, pValues);
      final rval = Map<String, String>.fromEntries(
        List.generate(
          length,
          (i) => MapEntry(
            pKeys[i].cast<Utf8>().toDartString(),
            pValues[i].cast<Utf8>().toDartString(),
          ),
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
