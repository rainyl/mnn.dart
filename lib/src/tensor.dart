import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;
import 'halide_runtime.dart';

class Tensor extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(c.addresses.mnn_tensor_destroy);

  Tensor.fromPointer(super.ptr, {super.attach, super.externalSize});

  factory Tensor.create(int dimSize, c.DimensionType dimType) {
    final p = c.mnn_tensor_create(dimSize, dimType);
    return Tensor.fromPointer(p);
  }

  factory Tensor.fromTensor(
    Tensor tensor,
    c.DimensionType dimType, {
    bool allocMemory = false,
  }) {
    final p = c.mnn_tensor_create_from_tensor(tensor.ptr, dimType, allocMemory);
    return Tensor.fromPointer(p);
  }

  factory Tensor.withData(
    ffi.Pointer<ffi.Int> shape,
    int shapeSize,
    HalideType type,
    ffi.Pointer<ffi.Void> data,
    c.DimensionType dimType,
  ) {
    final p = c.mnn_tensor_create_with_data(shape, shapeSize, type.ref, data, dimType);
    return Tensor.fromPointer(p);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    if (attach) {
      _finalizer.detach(this);
    }
    c.mnn_tensor_destroy(ptr);
  }

  Tensor clone(bool deepCopy) {
    final p = c.mnn_tensor_clone(ptr, deepCopy);
    return Tensor.fromPointer(p);
  }

  c.ErrorCode copyFromHost(Tensor hostTensor) {
    return c.mnn_tensor_copy_from_host(ptr, hostTensor.ptr);
  }

  c.ErrorCode copyToHost(Tensor hostTensor) {
    return c.mnn_tensor_copy_to_host(ptr, hostTensor.ptr);
  }

  int get dimensions => c.mnn_tensor_dimensions(ptr);

  List<int> get shape {
    final dims = dimensions;
    final shape = calloc<ffi.Int>(dims);
    final code = c.mnn_tensor_shape(ptr, shape, dims);
    if (code != c.ErrorCode.NO_ERROR) {
      throw Exception('mnn_tensor_shape failed: $code');
    }
    final result = List<int>.generate(dims, (i) => shape[i]);
    calloc.free(shape);
    return result;
  }

  int get size => c.mnn_tensor_size(ptr);
  int get elementSize => c.mnn_tensor_element_size(ptr);
  int get usize => c.mnn_tensor_usize(ptr);
  int get width => c.mnn_tensor_width(ptr);
  int get height => c.mnn_tensor_height(ptr);
  int get channel => c.mnn_tensor_channel(ptr);
  int get batch => c.mnn_tensor_batch(ptr);

  int getStride(int index) => c.mnn_tensor_stride(ptr, index);
  int getLength(int index) => c.mnn_tensor_length(ptr, index);

  void setStride(int index, int stride) => c.mnn_tensor_set_stride(ptr, index, stride);
  void setLength(int index, int length) => c.mnn_tensor_set_length(ptr, index, length);

  ffi.Pointer<ffi.Void> host() => c.mnn_tensor_host(ptr);
  int get deviceId => c.mnn_tensor_device_id(ptr);
  ffi.Pointer<ffi.Void> get buffer => c.mnn_tensor_buffer(ptr);

  c.DimensionType get dimensionType => c.mnn_tensor_get_dimension_type(ptr);
  c.HandleDataType get handleDataType => c.mnn_tensor_get_handle_data_type(ptr);

  void setType(int type) => c.mnn_tensor_set_type(ptr, type);
  HalideType get type => HalideType.fromPointer(c.mnn_tensor_get_type(ptr).cast());

  ffi.Pointer<ffi.Void> map(c.MapType mtype, c.DimensionType dtype) => c.mnn_tensor_map(ptr, mtype, dtype);
  void unmap(c.MapType mtype, c.DimensionType dtype, ffi.Pointer<ffi.Void> mapPtr) =>
      c.mnn_tensor_unmap(ptr, mtype, dtype, mapPtr);
  void wait(c.MapType mtype, bool finish) => c.mnn_tensor_wait(ptr, mtype, finish);

  void setDevicePtr(ffi.Pointer<ffi.Void> ptr, int memoryType) =>
      c.mnn_tensor_set_device_ptr(this.ptr, ptr, memoryType);
}
