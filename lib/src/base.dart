// ignore_for_file: camel_case_types

import 'dart:ffi' as ffi;
import 'package:meta/meta.dart';

mixin ComparableMixin {
  List<Object?> get props;

  @override
  // ignore: hash_and_equals
  bool operator ==(Object other) {
    if (other is! ComparableMixin) return false;
    if (props.length != other.props.length) return false;
    return props.indexed.every((e) => other.props[e.$1] == e.$2);
  }
}

/// Base class for wrapping C++ objects in Dart
abstract class NativeObject with ComparableMixin implements ffi.Finalizable {
  /// Pointer to the underlying C++ object
  final ffi.Pointer<ffi.Void> _ptr;

  ffi.NativeFinalizer get finalizer;

  @protected
  final bool attach;

  /// Creates a NativeObject instance
  ///
  /// @param ptr Pointer to the underlying C++ object
  /// @param attach Whether to automatically release the C++ object when Dart object is destroyed
  NativeObject(this._ptr, {this.attach = true, int? externalSize}) {
    if (_ptr == ffi.nullptr) {
      throw Exception("ptr is null");
    }
    if (attach) {
      finalizer.attach(this, _ptr, detach: this, externalSize: externalSize);
    }
  }

  bool get isEmpty => _ptr == ffi.nullptr;

  /// Gets the pointer to the underlying C++ object
  ffi.Pointer<ffi.Void> get ptr => _ptr;

  /// Releases the underlying C++ object
  /// Subclasses must implement specific release logic
  @protected
  void release();

  @mustCallSuper
  void dispose() {
    if (attach) {
      finalizer.detach(this);
      release();
    }
  }
}

typedef u8 = ffi.Uint8;
typedef u16 = ffi.Uint16;
typedef u32 = ffi.Uint32;
typedef u64 = ffi.Uint64;
typedef i8 = ffi.Int8;
typedef i16 = ffi.Int16;
typedef i32 = ffi.Int32;
typedef i64 = ffi.Int64;
typedef f32 = ffi.Float;
typedef f64 = ffi.Double;
