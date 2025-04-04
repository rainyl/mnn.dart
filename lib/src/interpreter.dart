import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;
import 'runtime_info.dart';
import 'schedule.dart';
import 'session.dart';
import 'tensor.dart';

class Interpreter extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(c.addresses.mnn_interpreter_destroy);

  Interpreter.fromPointer(super.ptr, {super.attach, super.externalSize});

  factory Interpreter.fromFile(String filePath) {
    final p = filePath.toNativeUtf8();
    final interpreter = c.mnn_interpreter_create_from_file(p.cast(), ffi.nullptr);
    calloc.free(p);
    return Interpreter.fromPointer(interpreter);
  }

  factory Interpreter.fromBuffer(Uint8List buffer) {
    final p = calloc<ffi.Uint8>(buffer.length);
    for (var i = 0; i < buffer.length; i++) {
      p[i] = buffer[i];
    }
    final interpreter = c.mnn_interpreter_create_from_buffer(p.cast(), buffer.length, ffi.nullptr);
    calloc.free(p);
    return Interpreter.fromPointer(interpreter);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    c.mnn_interpreter_destroy(ptr);
  }

  static RuntimeInfo createRuntime(List<ScheduleConfig> configs) {
    final p = calloc<c.mnn_schedule_config_t>(configs.length);
    for (var i = 0; i < configs.length; i++) {
      p[i] = configs[i].ref;
    }
    final pRuntime = c.mnn_interpreter_create_runtime(p, configs.length);
    return RuntimeInfo.fromPointer(pRuntime);
  }

  Session createSession({ScheduleConfig? config}) {
    config ??= ScheduleConfig.create();
    final p = c.mnn_interpreter_create_session(ptr, config.ptr.cast(), ffi.nullptr);
    return Session.fromPointer(p, ptr.cast());
  }

  Session createSessionWithRuntime(ScheduleConfig config, RuntimeInfo runtime) {
    final p = c.mnn_interpreter_create_session_with_runtime(ptr, config.ptr.cast(), runtime.ptr, ffi.nullptr);
    return Session.fromPointer(p, ptr.cast());
  }

  c.ErrorCode releaseSession(Session session) {
    return c.mnn_interpreter_release_session(ptr, session.ptr, ffi.nullptr);
  }

  c.ErrorCode resizeSession(Session session) {
    return c.mnn_interpreter_resize_session(ptr, session.ptr, ffi.nullptr);
  }

  Tensor? getSessionInput(Session session, {String? name}) {
    final p = name != null ? name.toNativeUtf8().cast<ffi.Char>() : ffi.nullptr;
    final tensor = c.mnn_interpreter_get_session_input(ptr, session.ptr, p);
    if (p != ffi.nullptr) calloc.free(p);
    return tensor == ffi.nullptr ? null : Tensor.fromPointer(tensor, attach: false);
  }

  /// @brief Get all input tensors from session
  ///
  /// @param session Session
  Map<String, Tensor> getSessionInputAll(Session session) {
    final pCount = calloc<ffi.Size>();
    final pNames = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pTensors = calloc<ffi.Pointer<c.mnn_tensor_t>>();
    final res = c.mnn_interpreter_get_session_input_all(ptr, session.ptr, pTensors, pNames, pCount);
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

  Tensor getSessionOutput(Session session, {String? name}) {
    final p = name != null ? name.toNativeUtf8().cast<ffi.Char>() : ffi.nullptr;
    final tensor = c.mnn_interpreter_get_session_output(ptr, session.ptr, p);
    if (p != ffi.nullptr) calloc.free(p);
    return Tensor.fromPointer(tensor, attach: false);
  }

  Map<String, Tensor> getSessionOutputAll(Session session) {
    final pCount = calloc<ffi.Size>();
    final pNames = calloc<ffi.Pointer<ffi.Pointer<ffi.Char>>>();
    final pTensors = calloc<ffi.Pointer<c.mnn_tensor_t>>();
    final res = c.mnn_interpreter_get_session_output_all(ptr, session.ptr, pTensors, pNames, pCount);
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

  void setSessionMode(SessionMode mode) {
    c.mnn_interpreter_set_session_mode(ptr, mode.value);
  }

  c.ErrorCode runSession(Session session) {
    return c.mnn_interpreter_run_session(ptr, session.ptr, ffi.nullptr);
  }

  String get bizCode => c.mnn_interpreter_biz_code(ptr).cast<Utf8>().toDartString();

  String get uuid => c.mnn_interpreter_uuid(ptr).cast<Utf8>().toDartString();

  void setCacheFile(String cacheFile, {int keySize = 128}) {
    final p = cacheFile.toNativeUtf8();
    c.mnn_interpreter_set_cache_file(ptr, p.cast(), keySize);
    calloc.free(p);
  }

  void setExternalFile(String file, {int flag = 128}) {
    final p = file.toNativeUtf8();
    c.mnn_interpreter_set_external_file(ptr, p.cast(), flag);
    calloc.free(p);
  }

  void setSessionHint(HintMode mode, int value) {
    c.mnn_interpreter_set_session_hint(ptr, mode.value, value);
  }

  void releaseModel() {
    c.mnn_interpreter_release_model(ptr);
  }

  /// @brief Get model buffer
  ///
  /// @param self Interpreter instance
  ///
  /// @param buffer Output parameter to receive pointer to model data
  ///
  /// @return Size of model data in bytes, or 0 if failed
  Uint8List get modelBuffer {
    final p = calloc<ffi.Pointer<ffi.Void>>();
    final size = c.mnn_interpreter_get_model_buffer(ptr, p);
    final buf = p.value.cast<ffi.Uint8>().asTypedList(size);
    calloc.free(p);
    return buf;
  }

  String get modelVersion => c.mnn_interpreter_get_model_version(ptr).cast<Utf8>().toDartString();

  /// @brief The API shoud be called after last resize session.
  ///
  /// If resize session generate new cache info, try to rewrite cache file.
  /// If resize session do not generate any new cache info, just do nothing.
  ///
  /// @param session    given session
  ///
  /// @param flag   Protected param, not used now
  c.ErrorCode updateCacheFile(Session session, {int flag = 0}) {
    return c.mnn_interpreter_update_cache_file(ptr, session.ptr, flag);
  }

  c.ErrorCode updateSessionToModel(Session session) {
    return c.mnn_interpreter_update_session_to_model(ptr, session.ptr);
  }

  void resizeTensor(Tensor tensor, List<int> dims) {
    final pDims = calloc<ffi.Int>(dims.length);
    pDims.cast<i32>().asTypedList(dims.length).setAll(0, dims);
    try {
      final code = c.mnn_interpreter_resize_tensor(ptr, tensor.ptr, pDims, dims.length);
      if (code != c.ErrorCode.NO_ERROR) {
        throw Exception('resizeTensor failed, code=$code');
      }
    } finally {
      calloc.free(pDims);
    }
  }

  @override
  List<int> get props => [ptr.address];

  @override
  String toString() {
    return 'Interpreter(address=0x${ptr.address.toRadixString(16)})';
  }
}

String getVersion() => c.mnn_get_version().cast<Utf8>().toDartString();
