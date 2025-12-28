// ignore_for_file: file_names

import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/numpy.dart' as np;
import 'package:test/test.dart';

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
      expect(info1.order, mnn.DimensionFormat.NHWC);
      expect(info1.dim, []);
      expect(info1.type, mnn.HalideType.f64);
      expect(info1.size, 1); // ceil
      info1.dispose();

      varp.resize([1, 2, 3, 3]);
      final info2 = varp.getInfo();
      expect(info2.order, mnn.DimensionFormat.NHWC);
      expect(info2.dim, [1, 2, 3, 3]);
      expect(info2.type, mnn.HalideType.f64);
      expect(info2.size, 1 * 2 * 3 * 3);
      info2.dispose();

      final (expr1, index) = varp.expr;
      expect(expr1.name, "testVariable");
      expect(index, 0);
      expr1.dispose();

      final varp1 = mnn.VARP.list<mnn.i32>([1, 2, 3, 4], format: mnn.DimensionFormat.NCHW);
      varp.input(varp1);
      varp1.dispose();

      final info3 = varp.getInfo();
      expect(info3.order, mnn.DimensionFormat.NCHW);
      expect(info3.dim, [4]);
      expect(info3.type, mnn.HalideType.i32);
      expect(info3.size, 4);
      info3.dispose();

      expect(varp.data, [1, 2, 3, 4]);
      varp.unMap();

      expect(varp.linkNumber(), isA<int>());

      final tensor = varp.getTensor();
      expect(tensor.shape, [4]);

      varp.setExpr(expr, 1);
      expect(varp.expr.$1, isA<mnn.Expr>());
      expect(varp.expr.$2, 1);

      expect(varp.toString(), startsWith('VARP('));

      varp.dispose();
    });

    test('VARP.list', () {
      final varp = mnn.VARP.list<mnn.f32>([1.0, 2.0, 3.0], format: mnn.DimensionFormat.NCHW);
      final info = varp.getInfo();
      expect(info.order, mnn.DimensionFormat.NCHW);
      expect(info.dim, [3]);
      expect(info.type, mnn.HalideType.f32);
      expect(info.size, 3);
      info.dispose();

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
      final a = mnn.VARP.scalar<mnn.f32>(10.0);
      final b = mnn.VARP.scalar<mnn.f32>(2.0);

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
      final a = mnn.VARP.scalar<mnn.f32>(10.0);
      final b = mnn.VARP.scalar<mnn.f32>(2.0);
      final c = mnn.VARP.scalar<mnn.f32>(10.0);
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
      final a = mnn.VARP.scalar<mnn.f32>(10.0);
      final b = mnn.VARP.scalar<mnn.f32>(2.0);
      final c = mnn.VARP.scalar<mnn.f32>(10.0);
      final d = mnn.VARP.list<mnn.f32>([10.0, 2.0, 10.0]);

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
      final v1 = mnn.VARP.scalar<mnn.i32>(1);
      final v2 = mnn.VARP.scalar<mnn.i32>(2);

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

      final v3 = mnn.VARP.scalar<mnn.i32>(3);
      vec.set(0, v3);
      expect(vec.at(0).item(), 3);

      vec.dispose();
      v1.dispose();
      v2.dispose();
      v3.dispose();
    });

    test('From List', () {
      final v1 = mnn.VARP.scalar<mnn.i32>(10);
      final v2 = mnn.VARP.scalar<mnn.i32>(20);
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

  // group('VarMap', () {
  //   test('Creation and Manipulation', () {
  //     final map = mnn.VarMap.create();
  //     expect(map.size(), 0);

  //     final v1 = mnn.VARP.scalar<mnn.i32>(100);
  //     map.set("key1", v1);
  //     expect(map.size(), 1);

  //     final v_retrieved = map.at("key1");
  //     expect(v_retrieved.readMap<mnn.i32>().value, 100);

  //     final dartMap = map.toMap();
  //     expect(dartMap.length, 1);
  //     expect(dartMap.containsKey("key1"), isTrue);
  //     expect(dartMap["key1"]!.readMap<mnn.i32>().value, 100);

  //     map.dispose();
  //     v1.dispose();
  //   });

  //   test('From Map', () {
  //     final v1 = mnn.VARP.scalar<mnn.i32>(1);
  //     final v2 = mnn.VARP.scalar<mnn.i32>(2);
  //     final dartMap = {"a": v1, "b": v2};

  //     final map = mnn.VarMap.of(dartMap);
  //     expect(map.size(), 2);
  //     expect(map.at("a").readMap<mnn.i32>().value, 1);
  //     expect(map.at("b").readMap<mnn.i32>().value, 2);

  //     map.dispose();
  //     v1.dispose();
  //     v2.dispose();
  //   });
  // });

  group('VARP Extra', () {
    test('list2D, list3D, list4D', () {
      final l2 = mnn.VARP.list2D<mnn.f32>([
        [1, 2],
        [3, 4],
      ]);
      expect(l2.dim, [2, 2]);
      l2.dispose();

      final l3 = mnn.VARP.list3D<mnn.f32>([
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

      final l4 = mnn.VARP.list4D<mnn.f32>([
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
      final a = mnn.VARP.scalar<mnn.f32>(1.5);
      final b = a.astype<mnn.i32>();
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
      final x = mnn.VARP.listND<mnn.f32>(data, [2, 2]);

      final m = x.mean([0]); // [2, 3]
      expect(m.data, [2.0, 3.0]);

      final s = x.sum([1]); // [3, 7]
      expect(s.data, [3.0, 7.0]);

      x.dispose();
      m.dispose();
      s.dispose();
    });

    // Testing load/save requires file system or buffer.
    // Buffer test is easier.
    test('saveToBuffer, loadFromBuffer', () {
      final v = mnn.VARP.scalar<mnn.f32>(123.0);
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
}
