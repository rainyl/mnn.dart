import 'dart:typed_data';

import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

void main() async {
  test('Tensor create', () async {
    final tensor = mnn.Tensor.create(dimSize: 4);
    expect(tensor.ptr.address, isNonZero);
    expect(tensor.dimensions, 4);
    expect(tensor.batch, 0);
    expect(tensor.channel, 0);
    expect(tensor.height, 0);
    expect(tensor.width, 0);
    expect(tensor.size, 0);
    expect(tensor.elementSize, 0);
    expect(tensor.usize, 0);
    expect(tensor.getStride(0), 0);
    expect(tensor.getStride(1), 0);
    expect(tensor.getStride(2), 0);
    expect(tensor.getStride(3), 0);
    expect(tensor.getLength(0), 0);
    expect(tensor.getLength(1), 0);
    expect(tensor.getLength(2), 0);
    expect(tensor.getLength(3), 0);
    tensor.setStride(0, 1);
    tensor.setStride(1, 2);
    tensor.setStride(2, 3);
    tensor.setStride(3, 4);
    tensor.setLength(0, 5);
    tensor.setLength(1, 6);
    tensor.setLength(2, 7);
    tensor.setLength(3, 8);
    expect(tensor.getStride(0), 1);
    expect(tensor.getStride(1), 2);
    expect(tensor.getStride(2), 3);
    expect(tensor.getStride(3), 4);
    expect(tensor.getLength(0), 5);
    expect(tensor.getLength(1), 6);
    expect(tensor.getLength(2), 7);
    expect(tensor.getLength(3), 8);
    expect(tensor.shape, [5, 6, 7, 8]);
    expect(tensor.size, 5 * 6 * 7 * 8 * 4);
    expect(tensor.elementSize, 5 * 6 * 7 * 8);
    expect(tensor.usize, 5 * 6 * 7 * 8 * 4);
  });

  test('Tensor data', () {
    final data = Float32List.fromList(List.generate(27, (index) => index.toDouble()));
    final tensor = mnn.Tensor.fromData(
      [1, 3, 3, 3],
      mnn.HalideType.f32(),
      data: data.buffer.asUint8List(),
      dimType: mnn.DimensionType.MNN_CAFFE,
    );
    expect(tensor.isEmpty, false);
    expect(tensor.dimensions, 4);
    expect(tensor.shape, [1, 3, 3, 3]);
    expect(tensor.size, 1 * 3 * 3 * 3 * 4); // 4 bytes per float32
    expect(tensor.elementSize, 1 * 3 * 3 * 3);
  });

  test('Tensor clone', () {
    final tensor = mnn.Tensor.create();
    tensor.setLength(0, 2);
    tensor.setLength(1, 3);
    tensor.setLength(2, 4);
    tensor.setLength(3, 5);

    final cloned = tensor.clone(deepCopy: true);
    expect(cloned.dimensions, tensor.dimensions);
    expect(cloned.shape, tensor.shape);
    expect(cloned.size, tensor.size);
    expect(cloned.elementSize, tensor.elementSize);
  });

  test('Tensor fromTensor', () {
    final srcTensor = mnn.Tensor.create();
    srcTensor.setLength(0, 2);
    srcTensor.setLength(1, 3);
    srcTensor.setLength(2, 4);
    srcTensor.setLength(3, 5);

    final newTensor = mnn.Tensor.fromTensor(srcTensor, dimType: mnn.DimensionType.MNN_CAFFE);
    expect(newTensor.dimensions, srcTensor.dimensions);
    expect(newTensor.shape, srcTensor.shape);
    expect(newTensor.dimensionType, mnn.DimensionType.MNN_CAFFE);
  });

  test('Tensor.createDevice', () {
    final tensor =
        mnn.Tensor.createDevice([2, 3, 4, 5], mnn.HalideType.f32(), dimType: mnn.DimensionType.MNN_CAFFE);
    expect(tensor.dimensions, 4);
    expect(tensor.elementSize, tensor.count);
    expect(tensor.shape, [2, 3, 4, 5]);
    expect(tensor.type, mnn.HalideType.f32());
    expect(tensor.dimensionType, mnn.DimensionType.MNN_CAFFE);
    expect(tensor.deviceId, isA<int>());
  });

  test('Tensor.fromImage', () {
    final image = mnn.Image.load("test/data/mnist_0.png");
    expect(image.dtype, mnn.StbiDType.u8);
    final tensor = mnn.Tensor.fromImage(image);
    expect(tensor.dimensions, 4);
    expect(tensor.shape, [1, 1, 28, 28]);
    expect(tensor.type, mnn.HalideType.f32());
    expect(tensor.dimensionType, mnn.DimensionType.MNN_CAFFE);
    // tensor.print();
  });

  test('Tensor type operations', () {
    final tensor = mnn.Tensor.create();
    tensor.setDataType(mnn.DataType.DataType_DT_FLOAT);
    expect(tensor.type, mnn.DataType.DataType_DT_FLOAT.toHalideType());
  });

  test('Tensor buffer operations', () {
    final tensor = mnn.Tensor.create();
    final buffer = tensor.buffer;
    expect(buffer, isA<mnn.HalideBuffer>());
    expect(buffer.type, mnn.HalideType.f32()); // default type is float32
    expect(buffer.dimensions, 4);
    expect(buffer.dimensions, tensor.dimensions);
  });

  test('Tensor data type conversions', () {
    final tensor = mnn.Tensor.create();
    tensor.setDataType(mnn.DataType.DataType_DT_INT32);
    expect(tensor.type, mnn.DataType.DataType_DT_INT32.toHalideType());

    tensor.setDataType(mnn.DataType.DataType_DT_UINT8);
    expect(tensor.type, mnn.DataType.DataType_DT_UINT8.toHalideType());
  });
}
