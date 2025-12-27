/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'exception.dart';
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
    p.asTypedList(buffer.length).setAll(0, buffer);
    final interpreter = c.mnn_interpreter_create_from_buffer(p.cast(), buffer.length, ffi.nullptr);
    calloc.free(p); // MNN::Interpreter::createFromBuffer will copy buffer, so we can free it here
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
    // pRuntime is not managed by interpreter, so we should free it manually
    // here, attach it to RuntimeInfo
    final pRuntime = c.mnn_interpreter_create_runtime(p, configs.length);
    return RuntimeInfo.fromPointer(pRuntime);
  }

  Session createSession({ScheduleConfig? config}) {
    config ??= ScheduleConfig.create();
    final p = c.mnn_interpreter_create_session(ptr, config.ptr.cast(), ffi.nullptr);
    return Session.fromPointer(p, this);
  }

  Session createSessionWithRuntime(ScheduleConfig config, RuntimeInfo runtime) {
    final p = c.mnn_interpreter_create_session_with_runtime(ptr, config.ptr.cast(), runtime.ptr, ffi.nullptr);
    return Session.fromPointer(p, this);
  }

  /// @brief release session.
  ///
  /// @param session   given session.
  ///
  /// @return true if given session is held by net and is freed.
  bool releaseSession(Session session) {
    final code = c.mnn_interpreter_release_session(ptr, session.ptr, ffi.nullptr);
    return switch (code) {
      c.ErrorCode.MNNC_BOOL_TRUE => true,
      c.ErrorCode.MNNC_BOOL_FALSE => false,
      _ => throw MNNException('releaseSession failed: $code'),
    };
  }

  void resizeSession(Session session) => session.resize();

  Tensor? getSessionInput(Session session, {String? name}) => session.getInput(name: name);

  /// @brief Get all input tensors from session
  ///
  /// @param session Session
  Map<String, Tensor> getSessionInputAll(Session session) => session.getInputAll();

  Tensor? getSessionOutput(Session session, {String? name}) => session.getOutput(name: name);

  Map<String, Tensor> getSessionOutputAll(Session session) => session.getOutputAll();

  void setSessionMode(SessionMode mode) {
    c.mnn_interpreter_set_session_mode(ptr, mode.value);
  }

  void runSession(Session session) => session.run();

  Future<void> runSessionAsync(Session session) async => session.runAsync();

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
      if (code != c.ErrorCode.MNNC_NO_ERROR) {
        throw MNNException('resizeTensor failed, code=$code');
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
