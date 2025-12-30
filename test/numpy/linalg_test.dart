import 'package:mnn/mnn.dart';
import 'package:mnn/numpy.dart' as np;
import 'package:mnn/src/numpy/linalg.dart' as la;
import 'package:test/test.dart';

void main() {
  test('norm vector 2-norm default', () {
    var a = np.array<float32>([3.0, 4.0]);
    var n = la.norm(a);
    expect(n.data!.first, equals(5.0));
  });

  test('norm vector 1-norm', () {
    var a = np.array<float32>([-3.0, 4.0]);
    var n = la.norm(a, ord: 1);
    expect(n.data!.first, equals(7.0));
  });

  test('norm vector inf-norm', () {
    var a = np.array<float32>([-3.0, 5.0, 4.0]);
    var n = la.norm(a, ord: double.infinity);
    expect(n.data!.first, equals(5.0));
  });

  test('norm vector -inf-norm', () {
    var a = np.array<float32>([-3.0, 5.0, 4.0]);
    var n = la.norm(a, ord: double.negativeInfinity);
    expect(n.data!.first, equals(3.0));
  });

  test('norm matrix frobenius default', () {
    // [[1, 2], [3, 4]]
    // 1+4+9+16 = 30. sqrt(30) approx 5.477
    var a = np.array<float32>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    var n = la.norm(a);
    expect(n.data!.first, closeTo(5.477, 0.001));
  });

  test('norm axis', () {
    // [[1, 2], [3, 4]]
    // axis 0: [sqrt(1+9), sqrt(4+16)] = [sqrt(10), sqrt(20)] = [3.162, 4.472]
    var a = np.array<float32>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    var n = la.norm(a, axis: 0);
    expect(n.shape, equals([2]));
    expect(n.data![0], closeTo(3.162, 0.001));
    expect(n.data![1], closeTo(4.472, 0.001));
  });

  test('svd', () {
    // Not sure if SVD is supported in current build (depends on MNN build options), but I'll try.
    // If MNN build doesn't include it, it might fail or return dummy.
    // Usually SVD of identity is 1s.
    var a = np.eye<float32>(2);
    var (u, w, vt) = la.svd(a);
    // w should be [1, 1]
    // Note: ordering of singular values might vary but usually sorted desc.
    expect(w.data, containsAll([1.0, 1.0]));
  });
}
