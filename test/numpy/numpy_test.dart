import 'dart:math' as math;

import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/numpy.dart' as np;
import 'package:test/test.dart';

import '../list_element_equals.dart';

void main() {
  group('np.array creation', () {
    test('np.scalar', () {
      final a = np.scalar<mnn.float32>(10.0);
      expect(a.shape, equals([]));
      expect(a.data, [10.0]);
      expect(a.value, equals(10.0));
    });
    test('np.empty', () {
      {
        final a = np.empty<mnn.float32>([3]);
        expect(a.shape, equals([3]));
        expect(a.data, isNotNull);
      }

      {
        final a = np.empty<mnn.int32>([1, 2, 3, 4]);
        expect(a.shape, equals([1, 2, 3, 4]));
        expect(a.data, isNotNull);

        final b = np.emptyLike<mnn.int32>(a);
        expect(b.shape, equals([1, 2, 3, 4]));
        expect(b.data, isNotNull);
      }
    });

    test('np.eye', () {
      final a = np.eye<mnn.float32>(3);
      expect(a.shape, equals([3, 3]));
      expect(a.data, isNotNull);
      expect(a.data![0], equals(1.0));
      expect(a.data![4], equals(1.0));
      expect(a.data![8], equals(1.0));
    });

    test('np.identity', () {
      final a = np.identity<mnn.float32>(3);
      expect(a.shape, equals([3, 3]));
      expect(a.data, isNotNull);
      expect(a.data![0], equals(1.0));
      expect(a.data![4], equals(1.0));
      expect(a.data![8], equals(1.0));
    });

    test('np.full', () {
      final a = np.full<mnn.float32>([2, 2], 10.0);
      expect(a.shape, equals([2, 2]));
      expect(a.data, [10.0, 10.0, 10.0, 10.0]);

      final b = np.fullLike<mnn.float32>(a, 20.0);
      expect(b.shape, equals([2, 2]));
      expect(b.data, [20.0, 20.0, 20.0, 20.0]);
    });

    test('np.ones', () {
      final a = np.ones<mnn.float32>([2, 2, 2]);
      expect(a.shape, equals([2, 2, 2]));
      expect(a.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);

      final b = np.onesLike<mnn.float32>(a);
      expect(b.shape, equals([2, 2, 2]));
      expect(b.data, [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    });

    test('np.zeros', () {
      final a = np.zeros<mnn.float32>([2, 2, 2]);
      expect(a.shape, equals([2, 2, 2]));
      expect(a.data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

      final b = np.zerosLike<mnn.float32>(a);
      expect(b.shape, equals([2, 2, 2]));
      expect(b.data, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    });

    test('np.copy', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      expect(a.shape, equals([3]));
      expect(a.data, [1.0, 2.0, 3.0]);

      final b = np.copy<mnn.float32>(a);
      expect(b.shape, equals([3]));
      expect(b.data, a.data);
    });

    test('np.array from list', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      expect(a.shape, equals([3]));
      expect(a.data, [1.0, 2.0, 3.0]);
    });

    test('np.array from 2D list', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      expect(a.shape, equals([2, 3]));
      expect(a.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    });

    test('np.array from 3D list', () {
      final a = np.array<mnn.float32>([
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ],
        [
          [7.0, 8.0, 9.0],
          [10.0, 11.0, 12.0],
        ],
      ]);
      expect(a.shape, equals([2, 2, 3]));
      expect(a.data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    });

    test('np.array from 4D list', () {
      final a = np.asarray<mnn.float32>([
        [
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
          ],
        ],
        [
          [
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
          ],
          [
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
          ],
        ],
      ]);
      expect(a.shape, equals([2, 2, 2, 3]));
      expect(a.data, [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
      ]);
    });

    test('np.arange', () {
      {
        final a = np.arange<mnn.float32>(start: 0.0, stop: 1.0, step: 0.1);
        expect(a.shape, equals([10]));
        final expected = List.generate(10, (i) => 0.0 + i * 0.1);
        expect(a.data, listCloseTo(expected, 0.0001));
      }

      {
        final a = np.arange<mnn.float32>(start: 0, stop: 10, step: 2);
        expect(a.shape, equals([5]));
        final expected = List.generate(5, (i) => 0.0 + i * 2.0);
        expect(a.data, listCloseTo(expected, 0.0001));
      }
    });

    test('np.linspace', () {
      {
        final a = np.linspace<mnn.float32>(0.0, 1.0, count: 100);
        expect(a.shape, equals([100]));
        expect(a.data?.length, 100);
      }

      {
        final a = np.linspace<mnn.float32>(10, 100, count: 100);
        expect(a.shape, equals([100]));
        expect(a.data?.length, 100);
      }
    });

    test('np.logspace', () {
      {
        final a = np.logspace<mnn.float32>(0.0, 1.0, count: 100);
        expect(a.shape, equals([100]));
        expect(a.data?.length, 100);
      }
    });
  });

  group('np manipulation', () {
    test('np.shape', () {
      final a = np.zeros<mnn.float32>([2, 3]);
      expect(np.shape(a), equals([2, 3]));
    });

    test('np.reshape', () {
      final a = np.arange<mnn.float32>(start: 0, stop: 6);
      final b = np.reshape(a, [2, 3]);
      expect(b.shape, equals([2, 3]));
      expect(b.data, equals([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]));
    });

    test('np.ravel', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      final b = np.ravel(a);
      expect(b.shape, equals([6]));
      expect(b.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    });

    test('np.transpose', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final b = np.transpose(a);
      expect(b.shape, equals([2, 2]));
      expect(b.data, equals([1.0, 3.0, 2.0, 4.0]));
    });

    test('np.swapaxes', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0, 3.0],
      ]);
      final b = np.swapaxes(a, 0, 1);
      expect(b.shape, equals([3, 1]));
      expect(b.data, equals([1.0, 2.0, 3.0]));
    });

    test('np.moveaxis', () {
      final a = np.zeros<mnn.float32>([3, 4, 5]);
      final b = np.moveaxis(a, [0], [-1]);
      expect(b.shape, equals([4, 5, 3]));
    });

    test('np.rollaxis', () {
      final a = np.zeros<mnn.float32>([3, 4, 5]);
      final b = np.rollaxis(a, 2, start: 0);
      expect(b.shape, equals([5, 3, 4]));
    });
  });

  group('np broadcast and join', () {
    test('np.broadcastTo', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      final b = np.broadcastTo(a, [2, 3]);
      expect(b.shape, equals([2, 3]));
      expect(b.data, equals([1.0, 2.0, 3.0, 1.0, 2.0, 3.0]));
    });

    test('np.expandDims', () {
      final a = np.array<mnn.float32>([1.0, 2.0]);
      final b = np.expandDims(a, [0]);
      expect(b.shape, equals([1, 2]));
      final c = np.expandDims(a, [1]);
      expect(c.shape, equals([2, 1]));
    });

    test('np.squeeze', () {
      final a = np.zeros<mnn.float32>([1, 2, 1, 3]);
      final b = np.squeeze(a);
      expect(b.shape, equals([2, 3]));
      final c = np.squeeze(a, axis: [0]);
      expect(c.shape, equals([2, 1, 3]));
    });

    test('np.concatenate', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final b = np.array<mnn.float32>([
        [5.0, 6.0],
      ]);
      final c = np.concatenate([a, b], axis: 0);
      expect(c.shape, equals([3, 2]));
      expect(c.data, equals([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
    });

    test('np.split', () {
      final a = np.arange<mnn.float32>(start: 0, stop: 9);
      final res = np.split(a, [3, 5, 6, 10]);
      expect(res.length, 5);
      expect(res[0].shape, equals([3]));
      expect(res[0].data, equals([0.0, 1.0, 2.0]));
      expect(res[1].shape, equals([2]));
      expect(res[1].data, equals([3.0, 4.0]));
      expect(res[2].shape, equals([1]));
      expect(res[2].data, equals([5.0]));
      expect(res[3].shape, equals([3]));
      expect(res[3].data, equals([6.0, 7.0, 8.0]));
      expect(res[4].shape, equals([1]));
      expect(res[4].data, equals([0.0]));
    });

    test('np.tile', () {
      final a = np.array<mnn.float32>([0.0, 1.0, 2.0]);
      final b = np.tile(a, [2]);
      expect(b.shape, equals([6]));
      expect(b.data, equals([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]));

      final c = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final d = np.tile(c, [2, 1]);
      expect(d.shape, equals([4, 2]));
      expect(d.data, equals([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]));
    });

    test('np.repeat', () {
      final a = np.array<mnn.float32>([3.0]);
      final b = np.repeat(a, 4);
      expect(b.shape, equals([4]));
      expect(b.data, equals([3.0, 3.0, 3.0, 3.0]));

      final c = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final d = np.repeat(c, 2);
      expect(d.shape, equals([8]));
      expect(d.data, equals([1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]));
    });
  });

  group('np logic and bitwise', () {
    test('np.bitwise', () {
      final a = np.array<mnn.int32>([1, 2]);
      final b = np.array<mnn.int32>([3, 4]);
      // 1 (01) & 3 (11) = 1
      // 2 (10) & 4 (100) = 0
      final and = np.bitwiseAnd(a, b);
      expect(and.data, equals([1, 0]));

      // 1 | 3 = 3
      // 2 | 4 = 6
      final or = np.bitwiseOr(a, b);
      expect(or.data, equals([3, 6]));

      // 1 ^ 3 = 2
      // 2 ^ 4 = 6
      final xor = np.bitwiseXor(a, b);
      expect(xor.data, equals([2, 6]));
    });

    test('np.where', () {
      final x = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      final y = np.array<mnn.float32>([4.0, 5.0, 6.0]);

      final c = np.greater(x, np.array<mnn.float32>([1.5, 1.5, 1.5]));
      // [1.0>1.5 (0), 2.0>1.5 (1), 3.0>1.5 (1)] -> [0, 1, 1]

      final res = np.where(c, x: x, y: y);
      // [0, 1, 1] -> [y[0], x[1], x[2]] -> [4.0, 2.0, 3.0]
      expect(res.data, equals([4.0, 2.0, 3.0]));
    });

    test('np.all and np.any', () {
      final a = np.array<mnn.int32>([1, 0, 1]);
      expect(np.all(a).data, equals([0])); // 0 is false
      expect(np.any(a).data, equals([1])); // 1 is true

      final b = np.array<mnn.int32>([1, 1]);
      expect(np.all(b).data, equals([1]));
    });

    test('np.comparisons', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      final b = np.array<mnn.float32>([2.0, 2.0, 2.0]);

      expect(np.less(a, b).data, equals([1, 0, 0]));
      expect(np.lessEqual(a, b).data, equals([1, 1, 0]));
      expect(np.greater(a, b).data, equals([0, 0, 1]));
      expect(np.greaterEqual(a, b).data, equals([0, 1, 1]));
      expect(np.equal(a, b).data, equals([0, 1, 0]));
      expect(np.notEqual(a, b).data, equals([1, 0, 1]));
    });

    test('np.arrayEqual', () {
      final a = np.array<mnn.float32>([1.0, 2.0]);
      final b = np.array<mnn.float32>([1.0, 2.0]);
      final c = np.array<mnn.float32>([1.0, 3.0]);

      expect(np.arrayEqual(a, b).data, equals([1]));
      expect(np.arrayEqual(a, c).data, equals([0]));
    });
  });

  group('np math', () {
    test('np.trigonometric', () {
      final a = np.array<mnn.float32>([0.0, math.pi / 2, math.pi]);
      final s = np.sin(a);
      // sin(0)=0, sin(pi/2)=1, sin(pi)=0
      expect(s.data![0], closeTo(0.0, 1e-5));
      expect(s.data![1], closeTo(1.0, 1e-5));
      expect(s.data![2], closeTo(0.0, 1e-5));

      final c = np.cos(a);
      // cos(0)=1, cos(pi/2)=0, cos(pi)=-1
      expect(c.data![0], closeTo(1.0, 1e-5));
      expect(c.data![1], closeTo(0.0, 1e-5));
      expect(c.data![2], closeTo(-1.0, 1e-5));

      final t = np.tan(np.array<mnn.float32>([0.0, math.pi / 4]));
      // tan(0)=0, tan(pi/4)=1
      expect(t.data![0], closeTo(0.0, 1e-5));
      expect(t.data![1], closeTo(1.0, 1e-5));

      final asin = np.arcsin(np.array<mnn.float32>([0.0, 1.0]));
      expect(asin.data![0], closeTo(0.0, 1e-5));
      expect(asin.data![1], closeTo(math.pi / 2, 1e-5));

      final acos = np.arccos(np.array<mnn.float32>([1.0, 0.0]));
      expect(acos.data![0], closeTo(0.0, 1e-5));
      expect(acos.data![1], closeTo(math.pi / 2, 1e-5));

      final atan = np.arctan(np.array<mnn.float32>([0.0, 1.0]));
      expect(atan.data![0], closeTo(0.0, 1e-5));
      expect(atan.data![1], closeTo(math.pi / 4, 1e-5));

      final atan2 = np.arctan2(np.array<mnn.float32>([0.0, 1.0]), np.array<mnn.float32>([1.0, 0.0]));
      expect(atan2.data![0], closeTo(0.0, 1e-5)); // atan2(0, 1) = 0
      expect(atan2.data![1], closeTo(math.pi / 2, 1e-5)); // atan2(1, 0) = pi/2
    });

    test('np.hyperbolic', () {
      final a = np.array<mnn.float32>([0.0]);
      expect(np.sinh(a).data![0], closeTo(0.0, 1e-5));
      expect(np.cosh(a).data![0], closeTo(1.0, 1e-5));
      expect(np.tanh(a).data![0], closeTo(0.0, 1e-5));

      final b = np.array<mnn.float32>([0.0]);
      expect(np.arcsinh(b).data![0], closeTo(0.0, 1e-5));

      final c = np.array<mnn.float32>([1.0]);
      expect(np.arccosh(c).data![0], closeTo(0.0, 1e-5));

      final d = np.array<mnn.float32>([0.0]);
      expect(np.arctanh(d).data![0], closeTo(0.0, 1e-5));
    });

    test('np.rounding', () {
      final a = np.array<mnn.float32>([1.1, 1.5, 1.9, 2.5, -1.1, -1.5, -1.9]);
      // around: round to nearest, .5 to even (NumPy) vs .5 away from zero (MNN/C default)
      // MNN F.round seems to round half away from zero (e.g. 2.5 -> 3.0)
      final around = np.around(a);
      // Expected if round half away from zero:
      // 1.1->1, 1.5->2, 1.9->2, 2.5->3, -1.1->-1, -1.5->-2, -1.9->-2
      expect(around.data, listCloseTo([1.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0], 1e-5));

      // round: alias for around
      final r = np.round(a);
      expect(r.data, listCloseTo([1.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0], 1e-5));

      // rint: alias for around
      final rint = np.rint(a);
      expect(rint.data, listCloseTo([1.0, 2.0, 2.0, 3.0, -1.0, -2.0, -2.0], 1e-5));

      // fix: round towards zero (trunc)
      final f = np.fix(a);
      // 1.1->1, 1.5->1, 1.9->1, 2.5->2, -1.1->-1, -1.5->-1, -1.9->-1
      expect(f.data, listCloseTo([1.0, 1.0, 1.0, 2.0, -1.0, -1.0, -1.0], 1e-5));

      final floor = np.floor(a);
      expect(floor.data, listCloseTo([1.0, 1.0, 1.0, 2.0, -2.0, -2.0, -2.0], 1e-5));

      final ceil = np.ceil(a);
      expect(ceil.data, listCloseTo([2.0, 2.0, 2.0, 3.0, -1.0, -1.0, -1.0], 1e-5));

      final trunc = np.trunc(a);
      expect(trunc.data, listCloseTo([1.0, 1.0, 1.0, 2.0, -1.0, -1.0, -1.0], 1e-5));
    });

    test('np.exponents and logs', () {
      final a = np.array<mnn.float32>([0.0, 1.0]);
      final e = np.exp(a);
      expect(e.data![0], closeTo(1.0, 1e-4));
      expect(e.data![1], closeTo(math.e, 1e-4));

      final em1 = np.expm1(np.array<mnn.float32>([0.0]));
      expect(em1.data![0], closeTo(0.0, 1e-5)); // exp(0)-1 = 0

      final l = np.log(np.array<mnn.float32>([1.0, math.e]));
      expect(l.data![0], closeTo(0.0, 1e-5));
      expect(l.data![1], closeTo(1.0, 1e-5));

      final l1p = np.log1p(np.array<mnn.float32>([0.0]));
      expect(l1p.data![0], closeTo(0.0, 1e-5)); // log(1+0) = 0
    });

    test('np.misc math', () {
      final a = np.array<mnn.float32>([-1.0, 0.0, 1.0]);
      final s = np.sign(a);
      expect(s.data, listCloseTo([-1.0, 0.0, 1.0], 1e-5));

      final r = np.reciprocal(np.array<mnn.float32>([2.0, 4.0]));
      expect(r.data, listCloseTo([0.5, 0.25], 1e-5));

      final positive = np.positive(np.array<mnn.float32>([2.0, 4.0, -1.0]));
      expect(positive.data, listCloseTo([2.0, 4.0, -1.0], 1e-5));

      final negative = np.negative(np.array<mnn.float32>([2.0, 4.0, -1.0]));
      expect(negative.data, listCloseTo([-2.0, -4.0, 1.0], 1e-5));

      final p = np.power(np.array<mnn.float32>([2.0, 3.0]), np.array<mnn.float32>([3.0, 2.0]));
      expect(p.data, listCloseTo([8.0, 9.0], 1e-5));

      final sq = np.square(np.array<mnn.float32>([2.0, 3.0]));
      expect(sq.data, listCloseTo([4.0, 9.0], 1e-5));

      final abs = np.absolute(np.array<mnn.float32>([-1.0, -2.0]));
      expect(abs.data, listCloseTo([1.0, 2.0], 1e-5));

      final fabs = np.fabs(np.array<mnn.float32>([-1.0, -2.0]));
      expect(fabs.data, listCloseTo([1.0, 2.0], 1e-5));

      final mod = np.mod(np.array<mnn.float32>([10.0]), np.array<mnn.float32>([3.0]));
      expect(mod.data, listCloseTo([1.0], 1e-5));

      final fdiv = np.floorDiv(np.array<mnn.float32>([10.0]), np.array<mnn.float32>([3.0]));
      expect(fdiv.data, listCloseTo([3.0], 1e-5));
    });
    test('np.arithmetic', () {
      final a = np.array<mnn.float32>([1.0, 2.0]);
      final b = np.array<mnn.float32>([3.0, 4.0]);

      expect(np.add(a, b).data, equals([4.0, 6.0]));
      expect(np.subtract(a, b).data, equals([-2.0, -2.0]));
      expect(np.multiply(a, b).data, equals([3.0, 8.0]));
      final div = np.divide(a, b);
      expect(div.data![0], closeTo(1.0 / 3.0, 1e-5));
      expect(div.data![1], closeTo(0.5, 1e-5));

      final tdiv = np.trueDiv(a, b);
      expect(tdiv.data![0], closeTo(1.0 / 3.0, 1e-5));
      expect(tdiv.data![1], closeTo(0.5, 1e-5));
    });

    test('np.sum and np.prod', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0, 4.0]);
      expect(np.sum(a).data, equals([10.0]));
      expect(np.prod(a).data, equals([24.0]));

      final b = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      expect(np.sum(b, axis: [0]).data, equals([4.0, 6.0])); // sum cols
      expect(np.sum(b, axis: [1]).data, equals([3.0, 7.0])); // sum rows
    });

    test('np.dot', () {
      // 1D . 1D
      final a = np.array<mnn.float32>([1.0, 2.0]);
      final b = np.array<mnn.float32>([3.0, 4.0]);
      final res1 = np.dot<mnn.float32>(a, b);
      expect(res1.data, equals([11.0])); // 1*3 + 2*4 = 3+8=11

      // 2D . 2D (MatMul)
      final m1 = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]); // 2x2
      final m2 = np.array<mnn.float32>([
        [1.0, 0.0],
        [0.0, 1.0],
      ]); // Identity
      final res2 = np.dot<mnn.float32>(m1, m2);
      expect(res2.shape, equals([2, 2]));
      expect(res2.data, equals([1.0, 2.0, 3.0, 4.0]));
    });

    test('np.matmul', () {
      final a = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final b = np.array<mnn.float32>([
        [2.0, 0.0],
        [0.0, 2.0],
      ]);
      final c = np.matmul(a, b);
      expect(c.data, equals([2.0, 4.0, 6.0, 8.0]));
    });

    test('np.min/max/abs', () {
      final a = np.array<mnn.float32>([-1.0, 2.0, -3.0]);
      expect(np.abs(a).data, equals([1.0, 2.0, 3.0]));

      final b = np.array<mnn.float32>([1.0, 5.0]);
      final c = np.array<mnn.float32>([2.0, 4.0]);

      expect(np.maximum(b, c).data, equals([2.0, 5.0]));
      expect(np.minimum(b, c).data, equals([1.0, 4.0]));
    });

    test('np.advanced_math', () {
      final a = np.array<mnn.float32>([3.0, 4.0]);
      final b = np.array<mnn.float32>([4.0, 3.0]);
      expect(np.hypot(a, b).data, listCloseTo([5.0, 5.0], 1e-5));

      final x = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      expect(np.exp2(x).data, listCloseTo([2.0, 4.0, 8.0], 1e-5));

      final l2 = np.log2(np.array<mnn.float32>([2.0, 4.0, 8.0]));
      expect(l2.data, listCloseTo([1.0, 2.0, 3.0], 1e-5));

      final l10 = np.log10(np.array<mnn.float32>([10.0, 100.0]));
      expect(l10.data, listCloseTo([1.0, 2.0], 1e-5));

      // logaddexp(x, y) = log(exp(x) + exp(y))
      final lae = np.logaddexp(np.array<mnn.float32>([0.0]), np.array<mnn.float32>([0.0]));
      // log(exp(0) + exp(0)) = log(1+1) = log(2)
      expect(lae.data![0], closeTo(math.log(2), 1e-5));

      // logaddexp2(x, y) = log2(2^x + 2^y)
      final lae2 = np.logaddexp2(np.array<mnn.float32>([1.0]), np.array<mnn.float32>([1.0]));
      // log2(2^1 + 2^1) = log2(4) = 2
      expect(lae2.data![0], closeTo(2.0, 1e-5));

      final s = np.sinc(np.array<mnn.float32>([0.0, 0.5, 1.0]));
      // sinc(0) = 1 (limit), sinc(0.5) = sin(pi*0.5)/(pi*0.5) = 1/(pi/2) = 2/pi, sinc(1) = sin(pi)/pi = 0
      expect(s.data![0], closeTo(1.0, 1e-5));
      expect(s.data![1], closeTo(2 / math.pi, 1e-5));
      expect(s.data![2], closeTo(0.0, 1e-5));

      final cbrt = np.cbrt(np.array<mnn.float32>([1.0, 8.0, 27.0]));
      expect(cbrt.data, listCloseTo([1.0, 2.0, 3.0], 1e-5));

      final clip = np.clip(np.array<mnn.float32>([-1.0, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0);
      expect(clip.data, listCloseTo([0.0, 0.0, 0.5, 1.0, 1.0], 1e-5));

      final sb = np.signbit(np.array<mnn.float32>([-1.0, 0.0, 1.0]));
      // -1 -> 1 (true), 0 -> 0 (false), 1 -> 0 (false)
      expect(sb.data, equals([1, 0, 0]));

      final cs = np.copysign(np.array<mnn.float32>([1.0, -1.0]), np.array<mnn.float32>([-1.0, 1.0]));
      // 1 with sign of -1 -> -1. -1 with sign of 1 -> 1.
      expect(cs.data, listCloseTo([-1.0, 1.0], 1e-5));

      final ld = np.ldexp(np.array<mnn.float32>([1.0, 2.0]), np.array<mnn.float32>([1.0, 2.0]));
      // 1 * 2^1 = 2. 2 * 2^2 = 8.
      expect(ld.data, listCloseTo([2.0, 8.0], 1e-5));

      final (div, mod) = np.divmod(np.array<mnn.float32>([10.0]), np.array<mnn.float32>([3.0]));
      expect(div.data, listCloseTo([3.0], 1e-5));
      expect(mod.data, listCloseTo([1.0], 1e-5));
    });
  });

  group('np cumulative', () {
    test('np.cumprod', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0, 4.0]);
      final cp = np.cumprod(a, 0);
      expect(cp.data, listCloseTo([1.0, 2.0, 6.0, 24.0], 1e-5));
    });

    test('np.cumsum', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0, 4.0]);
      final cs = np.cumsum(a, 0);
      expect(cs.data, listCloseTo([1.0, 3.0, 6.0, 10.0], 1e-5));
    });
  });

  group('np padding', () {
    test('np.pad', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0]);
      // pad width: [1, 1] -> 1 before, 1 after
      final p = np.pad(a, [1, 1]);
      expect(p.shape, equals([5]));
      expect(p.data, listCloseTo([0.0, 1.0, 2.0, 3.0, 0.0], 1e-5));

      final b = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      // pad width: [1, 1] -> broadcasts to [[1, 1], [1, 1]]
      final p2 = np.pad(b, [1, 1]);
      expect(p2.shape, equals([4, 4]));
      // check center
      // 0 0 0 0
      // 0 1 2 0
      // 0 3 4 0
      // 0 0 0 0
      // flatten: 0 0 0 0, 0 1 2 0, 0 3 4 0, 0 0 0 0
      expect(p2.data![5], closeTo(1.0, 1e-5));
      expect(p2.data![6], closeTo(2.0, 1e-5));
      expect(p2.data![9], closeTo(3.0, 1e-5));
      expect(p2.data![10], closeTo(4.0, 1e-5));
    });
  });

  group('np sort and search', () {
    test('np.sort', () {
      final a = np.array<mnn.float32>([3.0, 1.0, 2.0]);
      final s = np.sort(a);
      expect(s.data, listCloseTo([1.0, 2.0, 3.0], 1e-5));

      final b = np.array<mnn.float32>([
        [3.0, 1.0],
        [2.0, 4.0],
      ]);
      final s2 = np.sort(b, axis: 1);
      expect(s2.data, listCloseTo([1.0, 3.0, 2.0, 4.0], 1e-5));

      final ms = np.msort(b); // sort axis 0
      // [[3, 1], [2, 4]] -> col 0: 2,3. col 1: 1,4.
      // Result: [[2, 1], [3, 4]]
      expect(ms.data, listCloseTo([2.0, 1.0, 3.0, 4.0], 1e-5));
    });

    test('np.argsort', () {
      final a = np.array<mnn.float32>([3.0, 1.0, 2.0]);
      final args = np.argsort(a);
      // indices: 1 (val 1.0), 2 (val 2.0), 0 (val 3.0)
      expect(args.data, equals([1, 2, 0]));
    });

    test('np.argmax/argmin', () {
      final a = np.array<mnn.float32>([1.0, 3.0, 2.0]);
      expect(np.argmax(a).data, equals([1]));
      expect(np.argmin(a).data, equals([0]));

      final b = np.array<mnn.float32>([
        [1.0, 2.0, 3.0],
        [6.0, 5.0, 4.0],
      ]);
      expect(np.argmax(b, axis: 1).data, equals([2, 0]));
      expect(np.argmin(b, axis: 1).data, equals([0, 2]));
    });

    test('np.nonzero', () {
      final a = np.array<mnn.int32>([1, 0, 2, 0, 3]);
      final (nz, _) = np.nonzero(a);
      expect(nz.data, equals([0, 2, 4]));
      expect(np.countNonZero(a).data, equals([3]));

      final fnz = np.flatnonzero(a);
      expect(fnz.data, equals([0, 2, 4]));
    });

    test('np.argwhere', () {
      final a = np.array<mnn.int32>([1, 0, 2]);
      final aw = np.argwhere(a);
      expect(aw.data, equals([0, 2]));
    });
  });

  group('np statistics', () {
    test('np.ptp', () {
      final a = np.array<mnn.float32>([1.0, 5.0, 2.0, 8.0]);
      // range = 8 - 1 = 7
      expect(np.ptp(a).data, listCloseTo([7.0], 1e-5));

      final b = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      // ptp axis 0: [3-1, 4-2] = [2, 2]
      expect(np.ptp(b, axis: [0]).data, listCloseTo([2.0, 2.0], 1e-5));
      // ptp axis 1: [2-1, 4-3] = [1, 1]
      expect(np.ptp(b, axis: [1]).data, listCloseTo([1.0, 1.0], 1e-5));
    });

    test('np.mean', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0, 4.0]);
      // mean = (1+2+3+4)/4 = 2.5
      expect(np.mean(a).data, listCloseTo([2.5], 1e-5));

      final b = np.array<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      expect(np.mean(b, axis: [0]).data, listCloseTo([2.0, 3.0], 1e-5));
      expect(np.mean(b, axis: [1]).data, listCloseTo([1.5, 3.5], 1e-5));
    });

    test('np.variance and std', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 3.0, 4.0]);
      // mean = 2.5
      // var = ((1-2.5)^2 + ... + (4-2.5)^2) / 4 = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 5 / 4 = 1.25
      expect(np.variance(a).data, listCloseTo([1.25], 1e-5));
      expect(np.std(a).data, listCloseTo([math.sqrt(1.25)], 1e-5));
    });

    test('np.histogram', () {
      final a = np.array<mnn.float32>([1.0, 2.0, 1.0]);
      // range [1, 2], bins=2
      // edges: [1.0, 1.5, 2.0]
      // bucket 0 [1.0, 1.5): 2 elements (1.0, 1.0)
      // bucket 1 [1.5, 2.0]: 1 element (2.0)
      // Note: MNN histogram behavior might differ slightly on edges or range inclusion.
      // Usually histogram includes the rightmost edge in the last bin.
      final (hist, edges) = np.histogram(a, bins: 2, range: (1, 2));

      expect(hist.data, listCloseTo([2.0, 1.0], 1e-5));
      expect(edges.data, listCloseTo([1.0, 1.5, 2.0], 1e-5));
    });
  });
}
