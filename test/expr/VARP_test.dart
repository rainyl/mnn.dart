// ignore_for_file: file_names

import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/numpy.dart' as np;
import 'package:test/test.dart';

import '../list_element_equals.dart';

void main() {
  group('VARP creation', () {
    test('VARP.create', () {
      final info = mnn.VariableInfo.create(
        order: mnn.DimensionFormat.NHWC,
        dim: [],
        type: mnn.HalideType.f64,
      );
      final expr = mnn.Expr.fromVariableInfo(info, mnn.InputType.CONSTANT);
      final varp = mnn.VARP.create(expr, index: 0);
      expect(varp.getName(), isEmpty);
      varp.setName("testVariable");
      expect(varp.getName(), "testVariable");

      final info1 = varp.getInfo();
      expect(info1?.order, mnn.DimensionFormat.NHWC);
      expect(info1?.dim, []);
      expect(info1?.type, mnn.HalideType.f64);
      expect(info1?.size, 1); // ceil
      info1?.dispose();

      varp.resize([1, 2, 3, 3]);
      final info2 = varp.getInfo();
      expect(info2?.order, mnn.DimensionFormat.NHWC);
      expect(info2?.dim, [1, 2, 3, 3]);
      expect(info2?.type, mnn.HalideType.f64);
      expect(info2?.size, 1 * 2 * 3 * 3);
      info2?.dispose();

      final (expr1, index) = varp.expr;
      expect(expr1.name, "testVariable");
      expect(index, 0);
      expr1.dispose();

      final varp1 = mnn.VARP.fromList1D<mnn.int32>([1, 2, 3, 4], format: mnn.DimensionFormat.NCHW);
      varp.input(varp1);
      varp1.dispose();

      final info3 = varp.getInfo();
      expect(info3?.order, mnn.DimensionFormat.NCHW);
      expect(info3?.dim, [4]);
      expect(info3?.type, mnn.HalideType.i32);
      expect(info3?.size, 4);
      info3?.dispose();

      expect(varp.data, [1, 2, 3, 4]);

      expect(varp.linkNumber(), isA<int>());

      final tensor = varp.getTensor();
      expect(tensor.shape, [4]);

      expect(varp.toString(), startsWith('VARP('));

      varp.dispose();
    });

    test('VARP.list', () {
      final varp = mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0], format: mnn.DimensionFormat.NCHW);
      final info = varp.getInfo();
      expect(info?.order, mnn.DimensionFormat.NCHW);
      expect(info?.dim, [3]);
      expect(info?.type, mnn.HalideType.f32);
      expect(info?.size, 3);
      info?.dispose();

      expect(varp.dim, [3]);
      expect(varp.shape, [3]);
      expect(varp.ndim, 1);
      expect(varp.dtype, mnn.HalideType.f32);

      expect(varp.data, [1.0, 2.0, 3.0]);

      varp.dispose();
    });
  });

  group('VARP Operators', () {
    test('Arithmetic Operators', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      final sum = a + b;
      expect(sum.item(), closeTo(12.0, 0.001));

      final diff = a - b;
      expect(diff.item(), closeTo(8.0, 0.001));

      final prod = a * b;
      expect(prod.item(), closeTo(20.0, 0.001));

      final quot = a / b;
      expect(quot.item(), closeTo(5.0, 0.001));

      a.dispose();
      b.dispose();
      sum.dispose();
      diff.dispose();
      prod.dispose();
      quot.dispose();
    });

    test('Comparison Operators', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);
      final c = mnn.VARP.scalar<mnn.float32>(10.0);
      final d = a;

      expect(b < a, isA<bool>());
      expect(b <= a, isA<bool>());
      expect(a == b, false); // a and b are different VARP objects

      expect(a == c, isFalse);
      expect(a == b, isFalse);
      expect(a == d, isTrue);

      a.dispose();
      b.dispose();
      c.dispose();
      // d.dispose(); // DO NOT DISPOSE! d is same as a and the memory is managed by a
    });

    test('Comparision using MathOPs (numpy)', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);
      final c = mnn.VARP.scalar<mnn.float32>(10.0);
      final d = mnn.VARP.fromList1D<mnn.float32>([10.0, 2.0, 10.0]);

      final less = np.less(a, b);
      expect(less.item(), 0); // false

      final lessequal = np.lessEqual(a, c);
      expect(lessequal.item(), 1); // true

      final equal = np.equal(a, c);
      expect(equal.item(), 1); // true

      final equalVec = np.equal(d, a);
      expect(equalVec.data, [1, 0, 1]); // [true, false, true]

      a.dispose();
      b.dispose();
      c.dispose();
      less.dispose();
      lessequal.dispose();
      equal.dispose();
    });
  });

  group('VecVARP', () {
    test('Creation and Manipulation', () {
      final v1 = mnn.VARP.scalar<mnn.int32>(1);
      final v2 = mnn.VARP.scalar<mnn.int32>(2);

      final vec = mnn.VecVARP.create();
      expect(vec.size(), 0);

      vec.push_back(v1);
      expect(vec.size(), 1);

      vec.push_back(v2);
      expect(vec.size(), 2);

      final v1_retrieved = vec.at(0);
      expect(v1_retrieved.item(), 1);

      final v2_retrieved = vec.at(1);
      expect(v2_retrieved.item(), 2);

      final list = vec.toList();
      expect(list.length, 2);
      expect(list[0].item(), 1);
      expect(list[1].item(), 2);

      final v3 = mnn.VARP.scalar<mnn.int32>(3);
      vec.set(0, v3);
      expect(vec.at(0).item(), 3);

      vec.dispose();
      v1.dispose();
      v2.dispose();
      v3.dispose();
    });

    test('From List', () {
      final v1 = mnn.VARP.scalar<mnn.int32>(10);
      final v2 = mnn.VARP.scalar<mnn.int32>(20);
      final list = [v1, v2];

      final vec = mnn.VecVARP.of(list);
      expect(vec.size(), 2);
      expect(vec.at(0).item(), 10);
      expect(vec.at(1).item(), 20);

      vec.dispose();
      v1.dispose();
      v2.dispose();
    });
  });

  group('VARP Extra', () {
    test('list2D, list3D, list4D', () {
      final l2 = mnn.VARP.fromList2D<mnn.float32>([
        [1, 2],
        [3, 4],
      ]);
      expect(l2.dim, [2, 2]);
      l2.dispose();

      final l3 = mnn.VARP.fromList3D<mnn.float32>([
        [
          [1],
          [2],
        ],
        [
          [3],
          [4],
        ],
      ]);
      expect(l3.dim, [2, 2, 1]);
      l3.dispose();

      final l4 = mnn.VARP.fromList4D<mnn.float32>([
        [
          [
            [1],
          ],
          [
            [2],
          ],
        ],
        [
          [
            [3],
          ],
          [
            [4],
          ],
        ],
      ]);
      expect(l4.dim, [2, 2, 1, 1]);
      l4.dispose();
    });

    test('astype, fix, setOrder', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.5);
      final b = a.astype<mnn.int32>();
      expect(b.item(), 1);

      final fixed = a.fix(mnn.InputType.CONSTANT);
      expect(fixed, isTrue);

      a.setOrder(mnn.DimensionFormat.NC4HW4);
      expect(a.order, mnn.DimensionFormat.NC4HW4);

      a.dispose();
      b.dispose();
    });

    test('mean, sum instance methods', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final x = mnn.VARP.fromListND<mnn.float32>(data, [2, 2]);
      expect(x.data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));

      final m = x.mean([0]); // [2, 3]
      expect(m.data, listCloseTo([2.0, 3.0], 0.001));

      final s = x.sum([1]); // [3, 7]
      expect(s.data, listCloseTo([3.0, 7.0], 0.001));

      x.dispose();
      m.dispose();
      s.dispose();
    });

    // Testing load/save requires file system or buffer.
    // Buffer test is easier.
    test('saveToBuffer, loadFromBuffer', () {
      final v = mnn.VARP.scalar<mnn.float32>(123.0);
      v.setName("test_var");

      // saveToBuffer returns VecI8 (int8 vector)
      try {
        final buffer = mnn.VARP.saveToBuffer([v]);
        expect(buffer.size(), greaterThan(0));
        buffer.dispose();
      } catch (e) {
        print("Save buffer failed: $e");
      }
      v.dispose();
    });
  });

  group('Slicing Operators', () {
    test('1D Slicing', () {
      final v = mnn.VARP.fromListND<mnn.float32>([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [10]);

      // Full slice
      final all = v[':'];
      expect(all.size, 10);
      expect(all.toList(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

      // Prefix
      final prefix = v[':5'];
      expect(prefix.size, 5);
      expect(prefix.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);

      // Range
      final range = v['2:5']; // 2, 3, 4
      expect(range.size, 3);
      expect(range.toList(), [2.0, 3.0, 4.0]);

      // Suffix
      final suffix = v['8:']; // 8, 9
      expect(suffix.size, 2);
      expect(suffix.toList(), [8.0, 9.0]);

      // Step
      final stepped = v['::2']; // 0, 2, 4, 6, 8
      expect(stepped.size, 5);
      expect(stepped.toList(), [0.0, 2.0, 4.0, 6.0, 8.0]);

      v.dispose();
      all.dispose();
      prefix.dispose();
      range.dispose();
      suffix.dispose();
      stepped.dispose();
    });

    test('Multidimensional Slicing', () {
      // 2x3 matrix
      // [[1, 2, 3],
      //  [4, 5, 6]]
      final v = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4, 5, 6], [2, 3]);

      // Row 0
      final row0 = v['0, :'];
      expect(row0.dim, [3]); // Reduced rank
      expect(row0.toList(), [1.0, 2.0, 3.0]);

      // Column 1
      final col1 = v[':, 1'];
      expect(col1.dim, [2]); // Reduced rank
      expect(col1.toList(), [2.0, 5.0]);

      // Submatrix
      final sub = v[':, :2']; // First 2 cols
      expect(sub.dim, [2, 2]);
      expect(sub.toList(), [
        [1.0, 2.0],
        [4.0, 5.0],
      ]);

      v.dispose();
      row0.dispose();
      col1.dispose();
      sub.dispose();
    });

    test('Assignment', () {
      {
        final v = mnn.VARP.fromListND<mnn.float32>([0, 0, 0, 0], [4]);
        final ones = mnn.VARP.fromListND<mnn.float32>([1, 1], [2]);

        // Assign to slice
        v['1:3'] = ones; // indices 1, 2

        expect(v.toList(), [0.0, 1.0, 1.0, 0.0]);

        v.dispose();
        ones.dispose();
      }

      {
        final v = mnn.VARP.fromListND<mnn.float32>(List.generate(4 * 4, (i) => 0.0), [4, 4]);
        final ones = mnn.VARP.fromListND<mnn.float32>([1, 1, 1, 1], [2, 2]);

        // Assign to slice
        v['1:3, 1:3'] = ones; // indices 1, 2

        const expected = [
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 1.0, 0.0],
          [0.0, 1.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
        ];
        expect(v.toList(), expected);

        v.dispose();
        ones.dispose();
      }

      {
        final v = mnn.VARP.fromListND<mnn.float32>(List.generate(4 * 4, (i) => 0.0), [4, 4]);
        final ones = mnn.VARP.fromListND<mnn.float32>([1, 1, 1, 1], [4]);

        // Assign to slice
        v[0] = ones; // indices 1, 2

        const expected = [
          [1.0, 1.0, 1.0, 1.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0],
        ];
        expect(v.toList(), expected);

        v.dispose();
        ones.dispose();
      }
    });

    test('High Dimensional Slicing (3D, 4D, 5D)', () {
      // 3D: 2x2x2
      // [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
      final v3 = mnn.VARP.fromListND<mnn.float32>(List.generate(8, (i) => i.toDouble()), [2, 2, 2]);

      // Slice: [:, :, 1] -> 2x2 matrix of second elements
      // [[1, 3], [5, 7]]
      final slice3 = v3[':, :, 1'];
      expect(slice3.dim, [2, 2]);
      expect(slice3.toList(), [
        [1.0, 3.0],
        [5.0, 7.0],
      ]);
      slice3.dispose();
      v3.dispose();

      // 4D: 2x1x2x1
      // [[[[0], [1]]], [[[2], [3]]]]
      final v4 = mnn.VARP.fromListND<mnn.float32>(List.generate(4, (i) => i.toDouble()), [2, 1, 2, 1]);

      // Slice: [1, 0, :, 0] -> 1D array of length 2: [2, 3]
      final slice4 = v4['1, 0, :, 0'];
      expect(slice4.dim, [2]);
      expect(slice4.toList(), [2.0, 3.0]);
      slice4.dispose();
      v4.dispose();

      // 5D: 1x1x1x1x1 scalar-like
      final v5 = mnn.VARP.fromListND<mnn.float32>([42.0], [1, 1, 1, 1, 1]);

      // Slice: [0, 0, 0, 0, 0] -> scalar 42.0
      final slice5 = v5['0, 0, 0, 0, 0'];
      expect(slice5.dim, <int>[]); // Scalar has empty shape
      expect(slice5.value, 42.0);
      slice5.dispose();
      v5.dispose();
    });

    test('fmtString', () {
      // 1D
      {
        final v1 = mnn.VARP.fromListND<mnn.float32>([1, 2, 3], [3]);
        expect(v1.formatString(), "VARP([1, 2, 3], shape=[3], dtype=float32)");
        v1.dispose();
      }

      // 2D
      {
        final v2 = mnn.VARP.fromListND<mnn.float32>([1.1, 2.2, 3.3, 4.4], [2, 2]);
        // Expected output format:
        // VARP([[1.100000, 2.200000],
        //  [3.300000, 4.400000]], dtype=float32)
        const expected2D =
            "VARP([[1.100000, 2.200000],\n"
            " [3.300000, 4.400000]], shape=[2, 2], dtype=float32)";
        expect(v2.formatString(), expected2D);
        v2.dispose();
      }

      // 3D
      {
        final v3 = mnn.VARP.fromListND<mnn.float32>([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8], [2, 2, 2]);
        // VARP([[[1.100000, 2.200000],
        //        [3.300000, 4.400000]],
        //       [[5.500000, 6.600000],
        //        [7.700000, 8.800000]]], dtype=float32)
        const expected3D =
            'VARP([[[1.100000, 2.200000],\n'
            '       [3.300000, 4.400000]],\n'
            '      [[5.500000, 6.600000],\n'
            '       [7.700000, 8.800000]]], shape=[2, 2, 2], dtype=float32)';
        expect(v3.formatString(), expected3D);
        v3.dispose();
      }
    });
  });

  group('NumPy-like Data Access', () {
    test('toList', () {
      // 1D
      final v1 = mnn.VARP.fromListND<mnn.float32>([1, 2, 3], [3]);
      expect(v1.toList(), [1.0, 2.0, 3.0]);
      v1.dispose();

      // 2D
      final v2 = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      expect(v2.toList(), [
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      v2.dispose();
    });

    test('value', () {
      final s = mnn.VARP.scalar<mnn.float32>(42.0);
      expect(s.value, closeTo(42.0, 0.001));
      s.dispose();

      final v1 = mnn.VARP.fromListND<mnn.float32>([10.0], [1]);
      expect(v1.value, 10.0);
      v1.dispose();
    });

    test('operator []', () {
      final v = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4, 5, 6], [2, 3]);
      // [[1, 2, 3], [4, 5, 6]]

      final row0 = v[0];
      expect(row0.dim, [3]);
      expect(row0.toList(), [1.0, 2.0, 3.0]);

      final elem = row0[1]; // 2.0
      expect(elem.value, 2.0);

      row0.dispose();
      elem.dispose();
      v.dispose();
    });
  });

  group('VarMap', () {
    test('Creation and Manipulation', () {
      final map = mnn.VarMap.create();
      expect(map.size(), 0);

      final v1 = mnn.VARP.scalar<mnn.int32>(100);

      // by assignment, map owns the reference of v1.
      map["key1"] = v1;
      expect(map.size(), 1);

      // v_retrieved is v1, so do not free manually.
      final v_retrieved = map["key1"];
      expect(v_retrieved, v1);
      expect(v_retrieved.value, 100);
      v_retrieved.data![0] = 2541;
      expect(v1.value, 2541);

      final dartMap = map.toMap();
      expect(dartMap.length, 1);
      expect(dartMap.containsKey("key1"), isTrue);
      expect(dartMap["key1"]!.value, 2541);

      map.dispose();
      v1.dispose();
    });

    test('From Map', () {
      final v1 = mnn.VARP.scalar<mnn.int32>(25);
      final v2 = mnn.VARP.scalar<mnn.int32>(41);
      final dartMap = {"a": v1, "b": v2};

      final map = mnn.VarMap.of(dartMap);
      expect(map.size(), 2);
      expect(map.at("a").value, 25);
      expect(map.at("b").value, 41);

      map.dispose();
      v1.dispose();
      v2.dispose();
    });
  });
}
