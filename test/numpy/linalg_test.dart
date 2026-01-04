import 'package:mnn/mnn.dart';
import 'package:mnn/numpy.dart' as np;
import 'package:mnn/src/numpy/linalg.dart' as la;
import 'package:test/test.dart';

import '../list_element_equals.dart';

void main() {
  test('norm vector 2-norm default', () {
    final a = np.array<float32>([3.0, 4.0]);
    final n = la.norm(a);
    expect(n.value, equals(5.0));
  });

  test('norm vector 1-norm', () {
    final a = np.array<float32>([-3.0, 4.0]);
    final n = la.norm(a, ord: 1);
    expect(n.value, equals(7.0));
  });

  test('norm vector inf-norm', () {
    final a = np.array<float32>([-3.0, 5.0, 4.0]);
    final n = la.norm(a, ord: double.infinity);
    expect(n.value, equals(5.0));
  });

  test('norm vector -inf-norm', () {
    final a = np.array<float32>([-3.0, 5.0, 4.0]);
    final n = la.norm(a, ord: double.negativeInfinity);
    expect(n.value, equals(3.0));
  });

  test('norm matrix frobenius default', () {
    // [[1, 2], [3, 4]]
    // 1+4+9+16 = 30. sqrt(30) approx 5.477
    final a = np.array<float32>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    final n = la.norm(a);
    expect(n.value, closeTo(5.477, 0.001));
  });

  test('norm axis', () {
    // [[1, 2], [3, 4]]
    // axis 0: [sqrt(1+9), sqrt(4+16)] = [sqrt(10), sqrt(20)] = [3.162, 4.472]
    final a = np.array<float32>([
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
    final n = la.norm(a, axis: 0);
    expect(n.shape, equals([2]));
    expect(n.data, listCloseTo([3.162, 4.472], 0.001));
  });

  test('svd', () {
    final a = np.eye<float32>(2);
    final (u, w, vt) = la.svd(a);
    // w should be [1, 1]
    // Note: ordering of singular values might vary but usually sorted desc.
    expect(w.data, containsAll([1.0, 1.0]));
  });
}
