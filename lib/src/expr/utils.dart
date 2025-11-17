import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

extension ListIntExtension on List<int> {
  (ffi.Pointer<ffi.Int8> ptr, int size) toNativeArrayI8({ffi.Pointer<ffi.Int8>? ptr}) {
    ptr ??= calloc.allocate<ffi.Int8>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Uint8> ptr, int size) toNativeArrayU8({ffi.Pointer<ffi.Uint8>? ptr}) {
    ptr ??= calloc.allocate<ffi.Uint8>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Int16> ptr, int size) toNativeArrayI16({ffi.Pointer<ffi.Int16>? ptr}) {
    ptr ??= calloc.allocate<ffi.Int16>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Uint16> ptr, int size) toNativeArrayU16({ffi.Pointer<ffi.Uint16>? ptr}) {
    ptr ??= calloc.allocate<ffi.Uint16>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Uint32> ptr, int size) toNativeArrayU32({ffi.Pointer<ffi.Uint32>? ptr}) {
    ptr ??= calloc.allocate<ffi.Uint32>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Int32> ptr, int size) toNativeArrayI32({ffi.Pointer<ffi.Int32>? ptr}) {
    ptr ??= calloc.allocate<ffi.Int32>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Uint64> ptr, int size) toNativeArrayU64({ffi.Pointer<ffi.Uint64>? ptr}) {
    ptr ??= calloc.allocate<ffi.Uint64>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Int64> ptr, int size) toNativeArrayI64({ffi.Pointer<ffi.Int64>? ptr}) {
    ptr ??= calloc.allocate<ffi.Int64>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Float> ptr, int size) toNativeArrayF32({ffi.Pointer<ffi.Float>? ptr}) {
    ptr ??= calloc.allocate<ffi.Float>(length);
    ptr.asTypedList(length).setAll(0, map((e) => e.toDouble()));
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Double> ptr, int size) toNativeArrayF64({ffi.Pointer<ffi.Double>? ptr}) {
    ptr ??= calloc.allocate<ffi.Double>(length);
    ptr.asTypedList(length).setAll(0, map((e) => e.toDouble()));
    return (ptr, length);
  }
}

extension ListDoubleExtension on List<double> {
  (ffi.Pointer<ffi.Float> ptr, int size) toNativeArrayF32({ffi.Pointer<ffi.Float>? ptr}) {
    ptr ??= calloc.allocate<ffi.Float>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }

  (ffi.Pointer<ffi.Double> ptr, int size) toNativeArrayF64({ffi.Pointer<ffi.Double>? ptr}) {
    ptr ??= calloc.allocate<ffi.Double>(length);
    ptr.asTypedList(length).setAll(0, this);
    return (ptr, length);
  }
}
