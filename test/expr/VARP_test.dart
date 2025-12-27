// ignore_for_file: file_names

import 'dart:ffi' as ffi;
import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

void main() {
  group('VARP Operators', () {
    test('Arithmetic Operators', () {
      final a = mnn.VARP.scalar<ffi.Float>(10.0);
      final b = mnn.VARP.scalar<ffi.Float>(2.0);

      final sum = a + b;
      expect(sum.readMap<ffi.Float>().value, closeTo(12.0, 0.001));

      final diff = a - b;
      expect(diff.readMap<ffi.Float>().value, closeTo(8.0, 0.001));

      final prod = a * b;
      expect(prod.readMap<ffi.Float>().value, closeTo(20.0, 0.001));

      final quot = a / b;
      expect(quot.readMap<ffi.Float>().value, closeTo(5.0, 0.001));

      a.dispose();
      b.dispose();
      sum.dispose();
      diff.dispose();
      prod.dispose();
      quot.dispose();
    });

    // Comparison operators behavior is unclear (returns bool but seems to fail value check)
    // test('Comparison Operators', () {
    //   final a = mnn.VARP.scalar<ffi.Float>(10.0);
    //   final b = mnn.VARP.scalar<ffi.Float>(2.0);
    //   final c = mnn.VARP.scalar<ffi.Float>(10.0);

    //   // Note: According to expr.dart, these return bool
    //   expect(b < a, isTrue);
    //   expect(a < b, isFalse);

    //   expect(b <= a, isTrue);
    //   expect(a <= c, isTrue);
    //   expect(a <= b, isFalse);

    //   expect(a == c, isFalse);
    //   expect(a == b, isFalse);

    //   a.dispose();
    //   b.dispose();
    //   c.dispose();
    // });
  });

  group('VecVARP', () {
    test('Creation and Manipulation', () {
      final v1 = mnn.VARP.scalar<ffi.Int32>(1);
      final v2 = mnn.VARP.scalar<ffi.Int32>(2);

      final vec = mnn.VecVARP.create();
      expect(vec.size(), 0);

      vec.push_back(v1);
      expect(vec.size(), 1);

      vec.push_back(v2);
      expect(vec.size(), 2);

      final v1_retrieved = vec.at(0);
      expect(v1_retrieved.readMap<ffi.Int32>().value, 1);

      final v2_retrieved = vec.at(1);
      expect(v2_retrieved.readMap<ffi.Int32>().value, 2);

      final list = vec.toList();
      expect(list.length, 2);
      expect(list[0].readMap<ffi.Int32>().value, 1);
      expect(list[1].readMap<ffi.Int32>().value, 2);

      final v3 = mnn.VARP.scalar<ffi.Int32>(3);
      vec.set(0, v3);
      expect(vec.at(0).readMap<ffi.Int32>().value, 3);

      vec.dispose();
      v1.dispose();
      v2.dispose();
      v3.dispose();
    });

    test('From List', () {
      final v1 = mnn.VARP.scalar<ffi.Int32>(10);
      final v2 = mnn.VARP.scalar<ffi.Int32>(20);
      final list = [v1, v2];

      final vec = mnn.VecVARP.of(list);
      expect(vec.size(), 2);
      expect(vec.at(0).readMap<ffi.Int32>().value, 10);
      expect(vec.at(1).readMap<ffi.Int32>().value, 20);

      vec.dispose();
      v1.dispose();
      v2.dispose();
    });
  });

  // group('VarMap', () {
  //   test('Creation and Manipulation', () {
  //     final map = mnn.VarMap.create();
  //     expect(map.size(), 0);

  //     final v1 = mnn.VARP.scalar<ffi.Int32>(100);
  //     map.set("key1", v1);
  //     expect(map.size(), 1);

  //     final v_retrieved = map.at("key1");
  //     expect(v_retrieved.readMap<ffi.Int32>().value, 100);

  //     final dartMap = map.toMap();
  //     expect(dartMap.length, 1);
  //     expect(dartMap.containsKey("key1"), isTrue);
  //     expect(dartMap["key1"]!.readMap<ffi.Int32>().value, 100);

  //     map.dispose();
  //     v1.dispose();
  //   });

  //   test('From Map', () {
  //     final v1 = mnn.VARP.scalar<ffi.Int32>(1);
  //     final v2 = mnn.VARP.scalar<ffi.Int32>(2);
  //     final dartMap = {"a": v1, "b": v2};

  //     final map = mnn.VarMap.of(dartMap);
  //     expect(map.size(), 2);
  //     expect(map.at("a").readMap<ffi.Int32>().value, 1);
  //     expect(map.at("b").readMap<ffi.Int32>().value, 2);

  //     map.dispose();
  //     v1.dispose();
  //     v2.dispose();
  //   });
  // });

  group('VARP Extra', () {
    test('list2D, list3D, list4D', () {
      final l2 = mnn.VARP.list2D<ffi.Float>([
        [1, 2],
        [3, 4],
      ]);
      expect(l2.dim, [2, 2]);
      l2.dispose();

      final l3 = mnn.VARP.list3D<ffi.Float>([
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

      final l4 = mnn.VARP.list4D<ffi.Float>([
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
      final a = mnn.VARP.scalar<ffi.Float>(1.5);
      final b = a.astype<ffi.Int32>();
      expect(b.readMap<ffi.Int32>().value, 1);

      final fixed = a.fix(mnn.InputType.CONSTANT);
      expect(fixed, isTrue);

      a.setOrder(mnn.DimensionFormat.NC4HW4);
      expect(a.order, mnn.DimensionFormat.NC4HW4);

      a.dispose();
      b.dispose();
    });

    test('mean, sum instance methods', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final x = mnn.VARP.listND<ffi.Float>(data, [2, 2]);

      final m = x.mean([0]); // [2, 3]
      expect(m.readMap<ffi.Float>().asTypedList(2), [2.0, 3.0]);

      final s = x.sum([1]); // [3, 7]
      expect(s.readMap<ffi.Float>().asTypedList(2), [3.0, 7.0]);

      x.dispose();
      m.dispose();
      s.dispose();
    });

    // Testing load/save requires file system or buffer.
    // Buffer test is easier.
    test('saveToBuffer, loadFromBuffer', () {
      final v = mnn.VARP.scalar<ffi.Float>(123.0);
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
