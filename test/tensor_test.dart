import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

void main() async {
  test('Tensor create', () async {
    final tensor = mnn.Tensor.create(4, mnn.DimensionType.MNN_CAFFE);
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
    expect(tensor.size, 5 * 6 * 7 * 8 * tensor.elementSize);
    expect(tensor.elementSize, 4);
    expect(tensor.usize, 5 * 6 * 7 * 8 * 4);
  });
}
