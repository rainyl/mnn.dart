import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:mnn/src/stb_image/enum.dart';
import 'package:mnn/src/stb_image/image.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;
import 'halide_runtime.dart';

class Tensor extends NativeObject {
  static final ffi.NativeFinalizer _finalizer = ffi.NativeFinalizer(c.addresses.mnn_tensor_destroy);

  Tensor.fromPointer(super.ptr, {super.attach, super.externalSize});

  factory Tensor.create({int dimSize = 4, c.DimensionType dimType = c.DimensionType.MNN_CAFFE}) {
    final p = c.mnn_tensor_create(dimSize, dimType);
    return Tensor.fromPointer(p);
  }

  factory Tensor.createDevice(
    List<int> shape,
    HalideType type, {
    c.DimensionType dimType = c.DimensionType.MNN_TENSORFLOW,
  }) {
    final pShape = calloc<ffi.Int>(shape.length);
    pShape.cast<ffi.Int32>().asTypedList(shape.length).setAll(0, shape);
    final p = c.mnn_tensor_create_device(pShape, shape.length, type.ref, dimType);
    return Tensor.fromPointer(p);
  }

  factory Tensor.fromTensor(
    Tensor tensor, {
    c.DimensionType dimType = c.DimensionType.MNN_CAFFE,
    bool allocMemory = true,
  }) {
    final p = c.mnn_tensor_create_from_tensor(tensor.ptr, dimType, allocMemory);
    return Tensor.fromPointer(p);
  }

  factory Tensor.fromData(
    List<int> shape,
    HalideType type, {
    Uint8List? data,
    c.DimensionType dimType = c.DimensionType.MNN_TENSORFLOW,
  }) {
    final pShape = calloc<ffi.Int>(shape.length);
    pShape.cast<ffi.Int32>().asTypedList(shape.length).setAll(0, shape);
    final pData = data == null ? ffi.nullptr : calloc<ffi.Uint8>(data.length);
    if (data != null) {
      pData.asTypedList(data.length).setAll(0, data);
    }
    final p = c.mnn_tensor_create_with_data(pShape, shape.length, type.ref, pData.cast(), dimType);
    return Tensor.fromPointer(p);
  }

  factory Tensor.fromImage(
    Image image, {
    c.DimensionType dimType = c.DimensionType.MNN_CAFFE,
  }) {
    final imLength = image.width * image.height * image.channels;
    final pShape = calloc<ffi.Int>(4);
    pShape.cast<ffi.Int32>().asTypedList(4).setAll(0, [1, image.channels, image.width, image.height]);
    if (image.dtype == StbiDType.f32) {
      final p = c.mnn_tensor_create_with_data(pShape, 4, HalideType.f32().ref, image.ptr, dimType);
      return Tensor.fromPointer(p, attach: false);
    } else if (image.dtype == StbiDType.u8 || image.dtype == StbiDType.u16) {
      final pData = malloc<ffi.Float>(imLength);
      for (var i = 0; i < imLength; i++) {
        pData[i] = image.values[i].toDouble();
      }
      final p = c.mnn_tensor_create_with_data(pShape, 4, HalideType.f32().ref, pData.cast(), dimType);
      return Tensor.fromPointer(p, attach: true);
    } else {
      throw Exception('Unsupported image dtype: ${image.dtype}');
    }
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  void release() {
    c.mnn_tensor_destroy(ptr);
  }

  Tensor clone({bool deepCopy = false}) {
    final p = c.mnn_tensor_clone(ptr, deepCopy);
    return Tensor.fromPointer(p);
  }

  /// @brief for DEVICE tensor, copy data from given host tensor.
  ///
  /// @param hostTensor    host tensor, the data provider.
  ///
  /// @return true for DEVICE tensor, and false for HOST tensor.
  bool copyFromHost(Tensor hostTensor) {
    final code = c.mnn_tensor_copy_from_host(ptr, hostTensor.ptr);
    return switch (code) {
      c.ErrorCode.BOOL_TRUE => true,
      c.ErrorCode.BOOL_FALSE => false,
      _ => throw Exception('copyFromHost failed: $code'),
    };
  }

  /// @brief for DEVICE tensor, copy data to given host tensor.
  ///
  /// @param hostTensor    host tensor, the data consumer.
  ///
  /// @return true for DEVICE tensor, and false for HOST tensor.
  bool copyToHost(Tensor hostTensor) {
    final code = c.mnn_tensor_copy_to_host(ptr, hostTensor.ptr);
    return switch (code) {
      c.ErrorCode.BOOL_TRUE => true,
      c.ErrorCode.BOOL_FALSE => false,
      _ => throw Exception('copyToHost failed: $code'),
    };
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

  int get count => shape.reduce((value, element) => value * element);

  /// @brief Get tensor data size in bytes
  ///
  /// @return Size in bytes
  int get size => c.mnn_tensor_size(ptr);

  /// @brief Get tensor element count
  ///
  /// @return Element count
  int get elementSize => c.mnn_tensor_element_size(ptr);

  /// @brief Get tensor shape in bytes (unsigned)
  ///
  /// @return Size in bytes
  int get usize => c.mnn_tensor_usize(ptr);

  /// @brief Get tensor width
  ///
  /// @return Width
  int get width => c.mnn_tensor_width(ptr);

  /// @brief Get tensor height
  ///
  /// @return Height
  int get height => c.mnn_tensor_height(ptr);

  /// @brief Get tensor channel
  ///
  /// @return Channel
  int get channel => c.mnn_tensor_channel(ptr);

  /// @brief Get tensor batch
  ///
  /// @return Batch
  int get batch => c.mnn_tensor_batch(ptr);

  /// @brief Get tensor stride
  ///
  /// @param [index] Dimension index
  ///
  /// @return Stride
  int getStride(int index) => c.mnn_tensor_stride(ptr, index);

  /// @brief Get tensor length
  ///
  /// @param [index] Dimension index
  ///
  /// @return Length
  int getLength(int index) => c.mnn_tensor_length(ptr, index);

  /// @brief Set tensor stride
  ///
  /// @param [index] Dimension index
  ///
  /// @param [stride] Stride value
  void setStride(int index, int stride) => c.mnn_tensor_set_stride(ptr, index, stride);

  /// @brief Set tensor length
  ///
  /// @param index Dimension index
  ///
  /// @param length Length value
  void setLength(int index, int length) => c.mnn_tensor_set_length(ptr, index, length);

  /// @brief Get host data pointer
  ///
  /// @return Data pointer or NULL
  ffi.Pointer<ffi.Void> get host => c.mnn_tensor_host(ptr);

  /// @brief Get device ID
  ///
  /// @return Device ID
  int get deviceId => c.mnn_tensor_device_id(ptr);

  /// @brief Get buffer
  ///
  /// @return Buffer pointer
  HalideBuffer get buffer => HalideBuffer.fromPointer(c.mnn_tensor_buffer(ptr), attach: false);

  c.DimensionType get dimensionType => c.mnn_tensor_get_dimension_type(ptr);
  c.HandleDataType get handleDataType => c.mnn_tensor_get_handle_data_type(ptr);

  /// @brief Set data type
  ///
  /// @param type Data type
  void setDataType(DataType type) => c.mnn_tensor_set_type(ptr, type.value);

  /// @brief Get data type
  ///
  /// @return Data type
  HalideType get type => HalideType.fromPointer(c.mnn_tensor_get_type(ptr));

  /// @brief Map tensor for access
  ///
  /// @param mtype Map type
  ///
  /// @param dtype Dimension type
  ///
  /// @return Mapped pointer or NULL
  ffi.Pointer<ffi.Void> map(c.MapType mtype, c.DimensionType dtype) => c.mnn_tensor_map(ptr, mtype, dtype);

  /// @brief Unmap tensor
  ///
  /// @param mtype Map type
  ///
  /// @param dtype Dimension type
  ///
  /// @param map_ptr Mapped pointer
  void unmap(c.MapType mtype, c.DimensionType dtype, ffi.Pointer<ffi.Void> mapPtr) =>
      c.mnn_tensor_unmap(ptr, mtype, dtype, mapPtr);

  /// @brief Wait for tensor ready
  ///
  /// @param mtype Map type
  ///
  /// @param finish Whether wait for finish
  ///
  /// @return Error code
  void wait(c.MapType mtype, bool finish) => c.mnn_tensor_wait(ptr, mtype, finish);

  /// @brief Set device pointer
  ///
  /// @param device_ptr Device pointer
  ///
  /// @param memory_type Memory type
  ///
  /// @return Error code
  void setDevicePtr(ffi.Pointer<ffi.Void> ptr, int memoryType) =>
      c.mnn_tensor_set_device_ptr(this.ptr, ptr, memoryType);

  void print() => c.mnn_tensor_print(ptr);

  void printShape() => c.mnn_tensor_print_shape(ptr);

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'Tensor(address=0x${ptr.address.toRadixString(16)}, shape=$shape, type=$type, dimType=$dimensionType)';
  }
}

enum DataType {
  DataType_DT_INVALID(0),
  DataType_DT_FLOAT(1),
  DataType_DT_DOUBLE(2),
  DataType_DT_INT32(3),
  DataType_DT_UINT8(4),
  DataType_DT_INT16(5),
  DataType_DT_INT8(6),
  DataType_DT_STRING(7),
  DataType_DT_COMPLEX64(8),
  DataType_DT_INT64(9),
  DataType_DT_BOOL(10),
  DataType_DT_QINT8(11),
  DataType_DT_QUINT8(12),
  DataType_DT_QINT32(13),
  DataType_DT_BFLOAT16(14),
  DataType_DT_QINT16(15),
  DataType_DT_QUINT16(16),
  DataType_DT_UINT16(17),
  DataType_DT_COMPLEX128(18),
  DataType_DT_HALF(19),
  DataType_DT_RESOURCE(20),
  DataType_DT_VARIANT(21),
  DataType_MIN(0),
  DataType_MAX(21);

  final int value;
  const DataType(this.value);

  static DataType fromValue(int value) => switch (value) {
        0 => DataType_DT_INVALID,
        1 => DataType_DT_FLOAT,
        2 => DataType_DT_DOUBLE,
        3 => DataType_DT_INT32,
        4 => DataType_DT_UINT8,
        5 => DataType_DT_INT16,
        6 => DataType_DT_INT8,
        7 => DataType_DT_STRING,
        8 => DataType_DT_COMPLEX64,
        9 => DataType_DT_INT64,
        10 => DataType_DT_BOOL,
        11 => DataType_DT_QINT8,
        12 => DataType_DT_QUINT8,
        13 => DataType_DT_QINT32,
        14 => DataType_DT_BFLOAT16,
        15 => DataType_DT_QINT16,
        16 => DataType_DT_QUINT16,
        17 => DataType_DT_UINT16,
        18 => DataType_DT_COMPLEX128,
        19 => DataType_DT_HALF,
        20 => DataType_DT_RESOURCE,
        21 => DataType_DT_VARIANT,
        _ => throw Exception('Invalid value for DataType: $value'),
      };
}

extension DataTypeExt on DataType {
  /// @brief Convert DataType to HalideType
  ///
  /// @param type DataType
  ///
  /// @return HalideType
  HalideType toHalideType() {
    switch (this) {
      case DataType.DataType_DT_DOUBLE:
      case DataType.DataType_DT_FLOAT:
        return HalideType.f32();
      case DataType.DataType_DT_BFLOAT16:
        return HalideType.bf16();
      case DataType.DataType_DT_QINT32:
      case DataType.DataType_DT_INT32:
      case DataType.DataType_DT_BOOL:
      case DataType.DataType_DT_INT64:
        return HalideType.i32();
      case DataType.DataType_DT_QINT8:
      case DataType.DataType_DT_INT8:
        return HalideType.i8();
      case DataType.DataType_DT_QUINT8:
      case DataType.DataType_DT_UINT8:
        return HalideType.u8();
      case DataType.DataType_DT_QUINT16:
      case DataType.DataType_DT_UINT16:
        return HalideType.u16();
      case DataType.DataType_DT_QINT16:
      case DataType.DataType_DT_INT16:
        return HalideType.i16();
      default:
        throw Exception('Unsupported data type: $this');
    }
  }
}
