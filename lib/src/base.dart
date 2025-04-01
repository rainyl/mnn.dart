import 'dart:ffi' as ffi;
import 'package:meta/meta.dart';

/// Base class for wrapping C++ objects in Dart
abstract class NativeObject implements ffi.Finalizable {
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

  /// Gets the pointer to the underlying C++ object
  ffi.Pointer<ffi.Void> get ptr => _ptr;

  /// Releases the underlying C++ object
  /// Subclasses must implement specific release logic
  @protected
  void release();

  /// Destructor, called when the object is garbage collected
  @mustCallSuper
  void dispose() {
    print("dispose, _attach: $attach, ptr: ${_ptr.address.toRadixString(16)}");
    if (attach) {
      finalizer.detach(this);
      release();
    }
  }
}
