import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../core/backend.dart';
import '../core/base.dart';
import '../core/schedule.dart';
import '../expr/expr.dart';
import '../g/mnn.g.dart' as C;
import 'executor.dart';

class ModuleConfig extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);

  ModuleConfig.fromPointer(ffi.Pointer<C.mnn_module_config_t> ptr, {super.attach, super.externalSize})
    : super(ptr.cast());

  factory ModuleConfig.create({
    bool dynamic = false,
    bool? shapeMutable,
    bool rearrange = false,
    ForwardType backendType = ForwardType.MNN_FORWARD_CPU,
    BackendConfig? backendConfig,
  }) {
    final ptr = calloc<C.mnn_module_config_t>()
      ..ref.dynamic = dynamic
      ..ref.shape_mutable = shapeMutable ?? dynamic
      ..ref.rearrange = rearrange
      ..ref.backend_info_type = backendType.value
      ..ref.backend_info_config = backendConfig?.ptr.cast() ?? ffi.nullptr;
    return ModuleConfig.fromPointer(ptr);
  }

  C.mnn_module_config_t get ref => ptr.cast<C.mnn_module_config_t>().ref;

  bool get dynamic => ref.dynamic;
  set dynamic(bool value) => ref.dynamic = value;

  bool get shapeMutable => ref.shape_mutable;
  set shapeMutable(bool value) => ref.shape_mutable = value;

  bool get rearrange => ref.rearrange;
  set rearrange(bool value) => ref.rearrange = value;

  ForwardType get backendType => ForwardType.fromValue(ref.backend_info_type);
  set backendType(ForwardType value) => ref.backend_info_type = value.value;

  BackendConfig? get backendConfig => ref.backend_info_config != ffi.nullptr
      ? BackendConfig.fromPointer(ref.backend_info_config.cast(), attach: false)
      : null;
  set backendConfig(BackendConfig? value) => ref.backend_info_config = value?.ptr.cast() ?? ffi.nullptr;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    calloc.free(ptr);
  }
}

class Module extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_module_destroy);

  Module.fromPointer(C.mnn_module_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory Module.loadFromFile(
    String path, {
    List<String>? inputs,
    List<String>? outputs,
    RuntimeManager? runtimeManager,
    ModuleConfig? config,
  }) {
    final cPath = path.toNativeUtf8().cast<ffi.Char>();
    final cInputs = inputs == null ? ffi.nullptr : _toCStringList(inputs);
    final cOutputs = outputs == null ? ffi.nullptr : _toCStringList(outputs);
    try {
      final ptr = C.mnn_module_load_from_file(
        cPath,
        cInputs,
        inputs?.length ?? 0,
        cOutputs,
        outputs?.length ?? 0,
        runtimeManager?.ptr ?? ffi.nullptr,
        config?.ptr.cast<C.mnn_module_config_t>() ?? ffi.nullptr,
      );
      return Module.fromPointer(ptr);
    } finally {
      calloc.free(cPath);
      _freeCStringList(cInputs, inputs?.length ?? 0);
      _freeCStringList(cOutputs, outputs?.length ?? 0);
    }
  }

  factory Module.loadFromBuffer(
    Uint8List buffer, {
    List<String>? inputs,
    List<String>? outputs,
    RuntimeManager? runtimeManager,
    ModuleConfig? config,
  }) {
    final cBuffer = calloc<ffi.Uint8>(buffer.length)..asTypedList(buffer.length).setAll(0, buffer);
    final cInputs = inputs == null ? ffi.nullptr : _toCStringList(inputs);
    final cOutputs = outputs == null ? ffi.nullptr : _toCStringList(outputs);
    try {
      final ptr = C.mnn_module_load_from_bytes(
        cBuffer,
        buffer.length,
        cInputs,
        inputs?.length ?? 0,
        cOutputs,
        outputs?.length ?? 0,
        runtimeManager?.ptr ?? ffi.nullptr,
        config?.ptr.cast<C.mnn_module_config_t>() ?? ffi.nullptr,
      );
      return Module.fromPointer(ptr);
    } finally {
      calloc.free(cBuffer);
      _freeCStringList(cInputs, inputs?.length ?? 0);
      _freeCStringList(cOutputs, outputs?.length ?? 0);
    }
  }

  factory Module.extract(VecVARP inputs, VecVARP outputs, {bool forTrain = false}) {
    return Module.fromPointer(C.mnn_module_extract(inputs.ptr, outputs.ptr, forTrain));
  }

  Module clone({bool shareParams = false}) => Module.fromPointer(C.mnn_module_clone(ptr, shareParams));

  bool loadParams(VecVARP parameters) => C.mnn_module_load_parameters(ptr, parameters.ptr);

  bool get isTraining => C.mnn_module_get_is_training(ptr);
  set isTraining(bool value) => C.mnn_module_set_is_training(ptr, value);

  void clearCache() => C.mnn_module_clear_cache(ptr);

  String get name => C.mnn_module_get_name(ptr).cast<Utf8>().toDartString();
  set name(String value) {
    final cVal = value.toNativeUtf8().cast<ffi.Char>();
    C.mnn_module_set_name(ptr, cVal);
    calloc.free(cVal);
  }

  String get type => C.mnn_module_get_type(ptr).cast<Utf8>().toDartString();
  set type(String value) {
    final cVal = value.toNativeUtf8().cast<ffi.Char>();
    C.mnn_module_set_type(ptr, cVal);
    calloc.free(cVal);
  }

  int addParameter(VARP parameter) => C.mnn_module_add_parameter(ptr, parameter.ptr);
  void setParameter(VARP parameter, int index) => C.mnn_module_set_parameter(ptr, parameter.ptr, index);

  ModuleInfo get info => ModuleInfo.fromPointer(C.mnn_module_get_info(ptr));

  VARP forward(VARP input) {
    final p = calloc<C.VARP_t>();
    mnnRun(() => C.mnn_module_forward(ptr, input.ptr, p, ffi.nullptr));
    return VARP.fromPointer(p.value);
  }

  Future<VARP> forwardAsync(VARP input) async {
    final p = calloc<C.VARP_t>();
    return mnnRunAsync0(
      (callback) => C.mnn_module_forward(ptr, input.ptr, p, callback),
      (c) => c.complete(VARP.fromPointer(p.value)),
    );
  }

  VecVARP onForward(VecVARP inputs) {
    final p = calloc<C.VecVARP_t>();
    mnnRun(() => C.mnn_module_on_forward(ptr, inputs.ptr, p, ffi.nullptr));
    return VecVARP.fromPointer(p.value);
  }

  Future<VecVARP> onForwardAsync(VecVARP inputs) async {
    final p = calloc<C.VecVARP_t>();
    return mnnRunAsync0(
      (callback) => C.mnn_module_on_forward(ptr, inputs.ptr, p, callback),
      (c) => c.complete(VecVARP.fromPointer(p.value)),
    );
  }

  static ffi.Pointer<ffi.Pointer<ffi.Char>> _toCStringList(List<String> list) {
    final cList = calloc<ffi.Pointer<ffi.Char>>(list.length);
    for (int i = 0; i < (list.length); i++) {
      cList[i] = list[i].toNativeUtf8().cast<ffi.Char>();
    }
    return cList;
  }

  static void _freeCStringList(ffi.Pointer<ffi.Pointer<ffi.Char>> list, int length) {
    if (list != ffi.nullptr) {
      for (int i = 0; i < length; i++) {
        calloc.free(list[i]);
      }
      calloc.free(list);
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    C.mnn_module_destroy(ptr);
  }
}
