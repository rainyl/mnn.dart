/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import '../g/mnn.g.dart' as c;
import 'base.dart';

class RuntimeInfo extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(c.addresses.mnn_runtime_info_destroy);

  RuntimeInfo.fromPointer(super.ptr, {super.attach, super.externalSize});

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    c.mnn_runtime_info_destroy(ptr);
  }

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'RuntimeInfo(address=0x${ptr.address.toRadixString(16)})';
  }
}
