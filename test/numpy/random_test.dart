import 'package:mnn/numpy.dart' as np;
import 'package:test/test.dart';

void main() {
  test('np.random', () {
    final a = np.random([3, 4], low: 0.0, high: 10.0);
    expect(a.shape, [3, 4]);
    expect(np.min(a).value, greaterThanOrEqualTo(0.0));
    expect(np.max(a).value, lessThanOrEqualTo(10.0));
  });

  test('np.randint', () {
    {
      final a = np.randint(0, high: 10, size: [2, 5, 4, 1]);
      expect(a.shape, [2, 5, 4, 1]);
      expect(np.min(a).value, greaterThanOrEqualTo(0));
      expect(np.max(a).value, lessThanOrEqualTo(10));
    }

    {
      final a = np.randint(0, high: 10, size: []);
      expect(a.shape, []);
      expect(a.value, isA<int>());
      expect(a.value, greaterThanOrEqualTo(0));
      expect(a.value, lessThanOrEqualTo(10));
    }
  });
}
