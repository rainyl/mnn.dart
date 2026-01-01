import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/src/expr/op.dart' as op;
import 'package:test/test.dart';

import '../list_element_equals.dart';

void main() {
  group('BinaryOPs', () {
    test('add, subtract, multiply, divide, pow', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(op.add(a, b).value, closeTo(12.0, 0.001));
      expect(op.subtract(a, b).value, closeTo(8.0, 0.001));
      expect(op.multiply(a, b).value, closeTo(20.0, 0.001));
      expect(op.divide(a, b).value, closeTo(5.0, 0.001));
      expect(op.pow(a, b).value, closeTo(100.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('minimum, maximum', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(op.minimum(a, b).value, closeTo(2.0, 0.001));
      expect(op.maximum(a, b).value, closeTo(10.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('biasAdd, squaredDifference', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(op.biasAdd(a, b).value, closeTo(12.0, 0.001));
      expect(op.squaredDifference(a, b).value, closeTo(64.0, 0.001)); // (10-2)^2

      a.dispose();
      b.dispose();
    });

    test('floorDiv, floorMod', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.5);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(op.floorDiv(a, b).value, closeTo(5.0, 0.001));
      expect(op.floorMod(a, b).value, closeTo(0.5, 0.001));

      a.dispose();
      b.dispose();
    });

    // Note: Comparison and Logical ops return int (0 or 1) usually in MNN, or float 0.0/1.0 depending on type
    // Checking with cast to Int32 if needed, or check float value if that's what returns
    test('Comparison and Logical', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.0);
      final b = mnn.VARP.scalar<mnn.float32>(0.0);

      // greater
      expect(op.greater(a, b).value, 1);
      expect(op.greater(b, a).value, 0);

      // greaterEqual
      expect(op.greaterEqual(a, b).value, 1);
      expect(op.greaterEqual(a, a).value, 1);

      // less
      expect(op.less(b, a).value, 1);

      // lessEqual
      expect(op.lessEqual(b, a).value, 1);
      expect(op.lessEqual(a, a).value, 1);

      // equal
      expect(op.equal(a, a).value, 1);
      expect(op.equal(a, b).value, 0);

      // notEqual
      expect(op.notEqual(a, b).value, 1);
      expect(op.notEqual(a, a).value, 0);

      a.dispose();
      b.dispose();
    });

    test('GridSample, CosineSimilarity', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final grid = mnn.VARP.fromListND<mnn.float32>([0, 0], [1, 1, 1, 2]); // 1x1x1x2 grid
      final gs = op.GridSample(x, grid);
      expect(gs.dim, [1, 1, 1, 1]);
      gs.dispose();
      grid.dispose();

      final x1 = mnn.VARP.fromListND<mnn.float32>([1, 0], [2]);
      final x2 = mnn.VARP.fromListND<mnn.float32>([1, 0], [2]);
      final dim = mnn.VARP.scalar<mnn.int32>(0);
      final cs = op.CosineSimilarity(x1, x2, dim);
      expect(cs.value, closeTo(1.0, 0.001));
      cs.dispose();
      x1.dispose();
      x2.dispose();
      dim.dispose();

      x.dispose();
    });
  });

  group('UnaryOPs', () {
    test('abs, negative, sign', () {
      final a = mnn.VARP.scalar<mnn.float32>(-5.0);

      expect(op.abs(a).value, closeTo(5.0, 0.001));
      expect(op.negative(a).value, closeTo(5.0, 0.001));
      expect(op.sign(a).value, closeTo(-1.0, 0.001));

      a.dispose();
    });

    test('floor, ceil, round', () {
      final a = mnn.VARP.scalar<mnn.float32>(3.6);

      expect(op.floor(a).value, closeTo(3.0, 0.001));
      expect(op.ceil(a).value, closeTo(4.0, 0.001));
      expect(op.round(a).value, closeTo(4.0, 0.001));

      a.dispose();
    });

    test('square, sqrt, rsqrt', () {
      final a = mnn.VARP.scalar<mnn.float32>(4.0);

      expect(op.square(a).value, closeTo(16.0, 0.001));
      expect(op.sqrt(a).value, closeTo(2.0, 0.001));
      expect(op.rsqrt(a).value, closeTo(0.5, 0.001));

      a.dispose();
    });

    test('trig functions', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(op.sin(a).value, closeTo(0.0, 0.001));
      expect(op.cos(a).value, closeTo(1.0, 0.001));
      expect(op.tan(a).value, closeTo(0.0, 0.001));
      a.dispose();
    });

    test('exp, log, reciprocal', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.0);

      expect(op.exp(a).value, closeTo(2.718, 0.001));
      expect(op.log(op.exp(a)).value, closeTo(1.0, 0.001));

      final b = mnn.VARP.scalar<mnn.float32>(2.0);
      expect(op.reciprocal(b).value, closeTo(0.5, 0.001));

      a.dispose();
      b.dispose();
    });

    test('hyperbolic functions', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(op.sinh(a).value, closeTo(0.0, 0.001));
      expect(op.cosh(a).value, closeTo(1.0, 0.001));

      // tanh is already tested elsewhere but adding here for completeness
      expect(op.tanh(a).value, closeTo(0.0, 0.001));

      a.dispose();
    });

    test('inverse trig/hyperbolic', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);

      expect(op.asin(a).value, closeTo(0.0, 0.001));
      expect(op.atan(a).value, closeTo(0.0, 0.001));
      expect(op.asinh(a).value, closeTo(0.0, 0.001));
      expect(op.atanh(a).value, closeTo(0.0, 0.001));

      final b = mnn.VARP.scalar<mnn.float32>(1.0);
      expect(op.acos(b).value, closeTo(0.0, 0.001));
      expect(op.acosh(b).value, closeTo(0.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('erf, erfc, erfinv', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(op.erf(a).value, closeTo(0.0, 0.001));
      expect(op.erfc(a).value, closeTo(1.0, 0.001));

      // erfinv(0) = 0
      expect(op.erfinv(a).value, closeTo(0.0, 0.001));

      a.dispose();
    });
  });

  group('ReduceOPs', () {
    test('reduceSum, reduceMean, reduceMax, reduceMin, reduceVariance', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [2, 2];
      // [[1, 2],
      //  [3, 4]]
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      // Reduce along axis 0 (columns): [1+3, 2+4] = [4, 6]
      final sum0 = op.reduceSum(x, axis: [0]);
      expect(sum0.data, [4.0, 6.0]);

      // Reduce along axis 1 (rows): [1+2, 3+4] = [3, 7]
      final sum1 = op.reduceSum(x, axis: [1]);
      expect(sum1.data, [3.0, 7.0]);

      final mean = op.reduceMean(x, axis: [0, 1]); // scalar 2.5
      expect(mean.value, closeTo(2.5, 0.001));

      final max = op.reduceMax(x, axis: [0, 1]);
      expect(max.value, closeTo(4.0, 0.001));

      final min = op.reduceMin(x, axis: [0, 1]);
      expect(min.value, closeTo(1.0, 0.001));

      final variance = op.reduceVariance(x, axis: [0, 1]);
      expect(variance.value, closeTo(1.25, 0.001));

      x.dispose();
      sum0.dispose();
      sum1.dispose();
      mean.dispose();
      max.dispose();
      min.dispose();
    });

    test('reduceMutable', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [2, 2];
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      expect(op.reduceSumMutable(x).data, [10.0]);
      final sum = op.reduceSumMutable(x, axis: op.scalar<mnn.int32>(0));
      expect(sum.data, [4.0, 6.0]);

      expect(op.reduceMeanMutable(x).data, [2.5]);
      final mean = op.reduceMeanMutable(x, axis: op.scalar<mnn.int32>(1));
      expect(mean.data, [1.5, 3.5]);

      expect(op.reduceMaxMutable(x).data, [4.0]);
      final max = op.reduceMaxMutable(x, axis: op.scalar<mnn.int32>(0));
      expect(max.data, [3.0, 4.0]);

      expect(op.reduceMinMutable(x).data, [1.0]);
      final min = op.reduceMinMutable(x, axis: op.scalar<mnn.int32>(0));
      expect(min.data, [1.0, 2.0]);

      expect(op.reduceProdMutable(x).data, [24.0]);
      final prod = op.reduceProdMutable(x, axis: op.scalar<mnn.int32>(0));
      expect(prod.data, [3.0, 8.0]);

      x.dispose();
      sum.dispose();
      mean.dispose();
      max.dispose();
      min.dispose();
      prod.dispose();
    });

    test('reduceAnyMutable, reduceAllMutable', () {
      final data = [0, 1, 1, 1];
      final shape = [2, 2]; // [[0, 1], [1, 1]]
      final x = mnn.VARP.fromListND<mnn.int32>(data, shape);

      final any = op.reduceAnyMutable(x, axis: op.scalar<mnn.int32>(1)); // [1, 1]
      expect(any.data, [1, 1]);

      final all = op.reduceAllMutable(x, axis: op.scalar<mnn.int32>(1)); // [0, 1]
      expect(all.data, [0, 1]);

      x.dispose();
      any.dispose();
      all.dispose();
    });

    test('reduceProd', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [4];
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      final prod = op.reduceProd(x, axis: [0]);
      expect(prod.value, closeTo(24.0, 0.001));

      x.dispose();
      prod.dispose();
    });

    test('reduceAny, reduceAll', () {
      // 0 = false, 1 = true
      final data = [0, 1, 1, 1];
      final shape = [4];
      final x = mnn.VARP.fromListND<mnn.int32>(data, shape, format: mnn.DimensionFormat.NCHW);

      final any = op.reduceAny(x, axis: [0]);
      expect(any.value, 1);

      final all = op.reduceAll(x, axis: [0]);
      expect(all.value, 0);

      x.dispose();
      any.dispose();
      all.dispose();
    });
  });

  group('EltwiseOPs', () {
    test('prod, sum, max, sub', () {
      final a = mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0, 4.0], format: mnn.DimensionFormat.NCHW);
      final b = mnn.VARP.fromList1D<mnn.float32>([2.0, 3.0, 4.0, 5.0], format: mnn.DimensionFormat.NCHW);

      // prod: x * y element-wise
      expect(op.prod(a, b).data, [2.0, 6.0, 12.0, 20.0]);

      // sum: x + y
      expect(op.sum(a, b).data, [3.0, 5.0, 7.0, 9.0]);

      // max: max(x, y)
      expect(op.max(a, b).data, [2.0, 3.0, 4.0, 5.0]);

      // sub: x - y
      expect(op.sub(a, b).data, [-1.0, -1.0, -1.0, -1.0]);

      a.dispose();
      b.dispose();
    });
  });

  group('OtherOPs', () {
    test('cast', () {
      final a = mnn.VARP.scalar<mnn.float32>(3.14);
      final b = op.cast<mnn.int32>(a);
      expect(b.value, 3);

      a.dispose();
      b.dispose();
    });

    test('concat', () {
      final v1 = mnn.VARP.fromList1D<mnn.float32>([1, 2]);
      final v2 = mnn.VARP.fromList1D<mnn.float32>([3, 4]);

      // concat along axis 0: [1, 2, 3, 4]
      final c = op.concat([v1, v2], 0);
      expect(c.size, 4);
      expect(c.data, [1.0, 2.0, 3.0, 4.0]);

      v1.dispose();
      v2.dispose();
      c.dispose();
    });

    test('transpose', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [2, 2];
      // [[1, 2],
      //  [3, 4]]
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      // Transpose to [[1, 3], [2, 4]]
      final y = op.transpose(x, [1, 0]);

      expect(y.dim, [2, 2]);
      expect(y.data, [1.0, 3.0, 2.0, 4.0]);

      x.dispose();
      y.dispose();
    });

    test('slice', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [4];
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      final starts = mnn.VARP.fromList1D<mnn.int32>([1], format: mnn.DimensionFormat.NCHW);
      final sizes = mnn.VARP.fromList1D<mnn.int32>([2], format: mnn.DimensionFormat.NCHW);

      final y = op.slice(x, starts, sizes);
      expect(y.dim, [2]);
      expect(y.data, [2.0, 3.0]);

      x.dispose();
      starts.dispose();
      sizes.dispose();
      y.dispose();
    });

    test('fill, shape', () {
      final dims = mnn.VARP.fromList1D<mnn.int32>([2, 2], format: mnn.DimensionFormat.NCHW);
      final value = mnn.VARP.scalar<mnn.float32>(5.0);

      final filled = op.fill(dims, value);
      expect(filled.dim, [2, 2]);
      expect(filled.data, [5.0, 5.0, 5.0, 5.0]);

      final shapeVar = op.shape(filled);
      expect(shapeVar.data, [2, 2]);

      dims.dispose();
      value.dispose();
      filled.dispose();
      shapeVar.dispose();
    });

    test('reshape', () {
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final b = op.reshape(a, [4]);
      expect(b.dim, [4]);

      a.dispose();
      b.dispose();
    });

    test('OtherOps Bulk 1', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);

      // normalize
      final norm = op.normalize(x, 0, 0, 0.0, [0.5, 0.5]);
      expect(norm.data, listCloseTo([0.1581, 0.2236, 0.4743, 0.4472], 0.001));
      norm.dispose();

      // argMin
      final am = op.argMin(x, 0);
      expect(am.data, listCloseTo([0, 0, 0, 0], 0.001));
      am.dispose();

      // batchMatMul
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 2, 2]);
      final b = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [1, 2, 2]);
      final bmm = op.batchMatMul(a, b);
      expect(bmm.dim, [1, 2, 2]);
      bmm.dispose();
      a.dispose();
      b.dispose();

      // unravelIndex
      final indices = mnn.VARP.scalar<mnn.int32>(5);
      final dims = mnn.VARP.fromList1D<mnn.int32>([2, 3]);
      final ui = op.unravelIndex(indices, dims);
      // 5 in [2, 3] -> (1, 2)
      expect(ui.data, [1, 2]);
      indices.dispose();
      dims.dispose();
      ui.dispose();

      // linSpace
      final start = mnn.VARP.scalar<mnn.float32>(0);
      final stop = mnn.VARP.scalar<mnn.float32>(10);
      final num = mnn.VARP.scalar<mnn.int32>(5);
      final ls = op.linSpace(start, stop, num);
      expect(ls.dim, [5]);
      start.dispose();
      stop.dispose();
      num.dispose();
      ls.dispose();

      // randomUniform
      final shape = mnn.VARP.fromList1D<mnn.int32>([2, 2]);
      final rnd = op.randomUniform<mnn.float32>(shape);
      expect(rnd.dim, [2, 2]);
      shape.dispose();
      rnd.dispose();

      // cumSum, cumProd
      final cs = op.cumSum(mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0, 4.0]), 0);
      expect(cs.data, [1.0, 3.0, 6.0, 10.0]);

      final cp = op.cumProd(mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0, 4.0]), 0);
      expect(cp.data, [1.0, 2.0, 6.0, 24.0]);
      cs.dispose();
      cp.dispose();

      // histogram
      // TODO: add more format according to HistogramTest.cpp
      // input x is [1, 2, 3, 4]
      // bin=2, min=0, max=5. bins: [0, 2.5), [2.5, 5]
      // 1, 2 in bin 0; 3, 4 in bin 1
      final hist = op.histogram(op.constant<mnn.float32>([1.0, 2.0, 1.0, 4.0], [4]), 4, 0, 3, channel: -1);
      expect(hist.data, [0.0, 2.0, 1.0, 0.0]);
      hist.dispose();

      x.dispose();
    });

    test('svd', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [2, 2]);
      final res = op.svd(x); // returns [S, U, V] or similar
      expect(res.length, 3);
      for (final v in res) {
        v.dispose();
      }
      x.dispose();
    });

    test('scatterND, scatterElements', () {
      // scatterND
      final indices = mnn.VARP.fromListND<mnn.int32>([0, 1], [2, 1]);
      final updates = mnn.VARP.fromListND<mnn.float32>([10, 20], [2]);
      final shape = mnn.VARP.fromList1D<mnn.int32>([4]);
      final sc = op.scatterND(indices, updates, shape);
      expect(sc.data, [10.0, 20.0, 0.0, 0.0]);
      indices.dispose();
      updates.dispose();
      shape.dispose();
      sc.dispose();

      // scatterElements
      final data = mnn.VARP.fromListND<mnn.float32>([0, 0, 0], [3]);
      final idx = mnn.VARP.fromListND<mnn.int32>([1], [1]);
      final upd = mnn.VARP.fromListND<mnn.float32>([5], [1]);
      final se = op.scatterElements(data, idx, upd);
      expect(se.data, [0.0, 5.0, 0.0]);
      data.dispose();
      idx.dispose();
      upd.dispose();
      se.dispose();
    });
  });

  group('Neural Network Ops', () {
    test('NN Bulk 1', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);

      // input
      final inp = op.input<mnn.float32>([1, 3, 224, 224]);
      expect(inp.dim, [1, 3, 224, 224]);
      inp.dispose();

      // clone
      final cl = op.clone(x);
      expect(cl.dim, [1, 1, 2, 2]);
      expect(cl.data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      cl.dispose();

      // deconv, conv2dTranspose
      final w = mnn.VARP.fromListND<mnn.float32>([1, 1, 1, 1], [1, 1, 2, 2]);
      final b = mnn.VARP.scalar<mnn.float32>(0);
      final dec = op.deconv(w, b, x);
      expect(dec.ndim, 4);
      expect(dec.dim, [1, 2, 3, 1]);
      dec.dispose();

      final ct = op.conv2dTranspose(w, b, x);
      expect(ct.ndim, 4);
      ct.dispose();
      w.dispose();
      b.dispose();

      // scale
      // TODO: fails
      // final sc = op.scale(x, 1, [2.0], [0.0]);
      // expect(sc.data, listCloseTo([2.0, 4.0, 6.0, 8.0], 0.001));
      // sc.dispose();

      // ReLU6, PReLU, softMax, softPlus, softSign
      expect(
        op.ReLU(mnn.VARP.fromList1D<mnn.float32>([-1.0, 2.0, 3.0, 4.0])).data,
        listCloseTo([0.0, 2.0, 3.0, 4.0], 0.001),
      );
      expect(op.ReLU6(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(op.PReLU(x, [0.1]).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(op.softMax(x).data, listCloseTo([0.2689, 0.73105, 0.26894, 0.73105], 0.001));
      expect(op.softPlus(x).data, listCloseTo([1.3133, 2.1269, 3.0485, 4.0181], 0.001));
      expect(op.softSign(x).data, listCloseTo([0.5000, 0.6667, 0.7500, 0.8000], 0.001));

      // split
      // Input is 1x1x2x2. Axis 2 is height (2). Split into [1, 1].
      // Should result in two 1x1x1x2 tensors.
      final parts = op.split(x, [1, 1], axis: 2);
      expect(parts.length, 2);
      for (final p in parts) {
        expect(p.dim, [1, 1, 1, 2]);
        p.dispose();
      }

      x.dispose();
    });

    test('NN Bulk 2', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);

      // stridedSlice
      final begin = mnn.VARP.fromList1D<mnn.int32>([0, 0, 0, 0]);
      final end = mnn.VARP.fromList1D<mnn.int32>([1, 1, 1, 1]);
      final stride = mnn.VARP.fromList1D<mnn.int32>([1, 1, 1, 1]);
      final ss = op.stridedSlice(x, begin, end, stride, 0, 0, 0, 0, 0);
      expect(ss.dim, [1, 1, 1, 1]);
      ss.dispose();
      begin.dispose();
      end.dispose();
      stride.dispose();

      // convert, transpose1, channelShuffle, changeInputFormat
      expect(
        op.convert(x, mnn.DimensionFormat.NC4HW4).data,
        listCloseTo([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], 0.001),
      );
      // transpose1 takes VARP perm
      final perm = mnn.VARP.fromList1D<mnn.int32>([0, 1, 3, 2]);
      expect(op.transpose1(x, perm).dim, [1, 1, 2, 2]);
      perm.dispose();

      expect(op.channelShuffle(x, 1).data, listCloseTo([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], 0.001));

      // TODO: fails
      // expect(op.changeInputFormat(x, mnn.DimensionFormat.NC4HW4).data, listCloseTo([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], 0.001));

      // reverse, reverseSequence
      final axis = mnn.VARP.scalar<mnn.int32>(3);
      expect(op.reverse(x, axis).data, [2.0, 1.0, 4.0, 3.0]);
      axis.dispose();

      final seqLen = mnn.VARP.fromList1D<mnn.int32>([1]);
      expect(op.reverseSequence(x, seqLen, 0, 1).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      seqLen.dispose();

      // crop, resize, cropAndResize
      // crop
      final cropSize = mnn.VARP.fromListND<mnn.int32>([1, 1], [2]);
      // offset is List<int>
      final cr = op.crop(x, cropSize, 2, [0, 0]);
      expect(cr.dim, [1]);
      cr.dispose();

      // resize
      final res = op.resize(x, 2.0, 2.0);
      expect(res.dim, [1, 1, 4, 4]);
      res.dispose();

      // TODO
      // cropAndResize

      // expandDims
      final ed = op.expandDims(x, 0);
      expect(ed.dim, [1, 1, 1, 2, 2]);
      // ed.dispose();

      // squeeze
      final sq = op.squeeze(ed, axis: [0]);
      expect(sq.dim, [1, 1, 2, 2]);
      sq.dispose();

      // TODO
      // batchToSpaceND, spaceToBatchND

      // TODO
      // gatherND, gatherElements

      x.dispose();
    });

    test('NN Bulk 3', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);

      // selu, elu, threshold
      expect(op.selu(x, 1.0, 1.0).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(op.elu(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(op.threshold(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));

      // size, rank
      expect(op.size(x).value, 4);
      expect(op.rank(x).value, 1);

      // matrixBandPart
      // TODO: fails
      // final mbp = op.matrixBandPart(x, mnn.VARP.scalar<mnn.i32>(0), mnn.VARP.scalar<mnn.i32>(0));
      // expect(mbp.value, isNotNull);
      // mbp.dispose();

      // moments
      // returns [mean, variance]
      final m = op.moments(x, [0], mnn.VARP.scalar<mnn.float32>(0), true);
      expect(m.length, 2);
      for (final v in m) {
        v.dispose();
      }

      // setDiff1D
      final y = mnn.VARP.fromList1D<mnn.int32>([1, 2]);
      final sd = op.setDiff1D(mnn.VARP.fromList1D<mnn.int32>([1, 2, 3, 4]), y); // Should return 3, 4
      expect(sd.data, [3.0, 4.0]);
      sd.dispose();
      y.dispose();

      // zerosLike
      final zl = op.zerosLike(x);
      expect(zl.data, [0.0, 0.0, 0.0, 0.0]);
      zl.dispose();

      // range
      final r = op.range(
        mnn.VARP.scalar<mnn.float32>(0),
        mnn.VARP.scalar<mnn.float32>(5),
        mnn.VARP.scalar<mnn.float32>(1),
      );
      expect(r.data, [0.0, 1.0, 2.0, 3.0, 4.0]);
      r.dispose();

      // Permute
      final x2 = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final p = op.Permute(x2, [1, 0]);
      expect(p.dim, [2, 2]);
      x2.dispose();
      p.dispose();

      // interp
      // TODO:
      // op.interp([x], ...)

      // zeroGrad
      final zg = op.zeroGrad(x);
      expect(zg.data, isNotEmpty);
      zg.dispose();

      // floatToInt8, int8ToFloat
      final scale = mnn.VARP.scalar<mnn.float32>(1.0);
      final i8 = op.floatToInt8(x, scale);
      expect(i8.data, listCloseTo([1, 2, 3, 4], 0.001));
      final f = op.int8ToFloat(i8, scale);
      expect(f.data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      i8.dispose();
      f.dispose();
      scale.dispose();

      x.dispose();
    });

    test('maxPool', () {
      // Input: 1x1x4x4
      final input = mnn.VARP.fromListND<mnn.float32>(
        List.generate(16, (i) => i.toDouble()),
        [1, 1, 4, 4],
        format: mnn.DimensionFormat.NCHW,
      );

      // MaxPool 2x2, stride 2
      final output = op.maxPool(
        input,
        [2, 2],
        stride: [2, 2],
        pad: op.PaddingMode.VALID,
      );

      // Output should be 2x2
      expect(output.dim, [1, 1, 2, 2]);
      expect(output.data, [5.0, 7.0, 13.0, 15.0]);

      input.dispose();
      output.dispose();
    });

    test('avgPool', () {
      final input = mnn.VARP.fromListND<mnn.float32>(
        [1, 2, 3, 4],
        [1, 1, 2, 2],
        format: mnn.DimensionFormat.NCHW,
      );
      final output = op.avgPool(input, [2, 2], stride: [2, 2], pad: op.PaddingMode.VALID);
      expect(output.ndim, 4);
      expect(output.data, listCloseTo([2.5], 0.001));

      input.dispose();
      output.dispose();
    });

    test('pad', () {
      final input = mnn.VARP.fromListND<mnn.float32>([1], [1, 1]);
      // Pad 1 on all sides: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
      final paddings = mnn.VARP.fromListND<mnn.int32>([1, 1, 1, 1], [2, 2]);
      final output = op.pad(input, paddings);
      expect(output.dim, [3, 3]);

      input.dispose();
      paddings.dispose();
      output.dispose();
    });
  });

  group('More Ops', () {
    test('activations: gelu, sigmoid, silu, hardswish, log1p, expm1', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.0);

      // Just check they run and return reasonable values (not crashing)
      expect(op.gelu(a).value, isNotNull);
      expect(op.sigmoid(a).value, closeTo(0.731, 0.001));
      expect(op.silu(a).value, isNotNull);
      expect(op.hardswish(a).value, isNotNull);
      expect(op.log1p(a).value, closeTo(0.693, 0.001));
      expect(op.expm1(a).value, closeTo(1.718, 0.001));

      a.dispose();
    });

    test('logic & bits', () {
      final a = mnn.VARP.scalar<mnn.int32>(1); // 01
      final b = mnn.VARP.scalar<mnn.int32>(3); // 11

      // bitwiseAnd: 1 & 3 = 1
      expect(op.bitwiseAnd(a, b).value, 1);
      // bitwiseOr: 1 | 3 = 3
      expect(op.bitwiseOr(a, b).value, 3);
      // bitwiseXor: 1 ^ 3 = 2
      expect(op.bitwiseXor(a, b).value, 2);

      // logicalOr
      final t = mnn.VARP.scalar<mnn.int32>(1);
      final f = mnn.VARP.scalar<mnn.int32>(0);
      expect(op.logicalOr(t, f).value, 1);

      a.dispose();
      b.dispose();
      t.dispose();
      f.dispose();
    });

    test('math: atan2, mod', () {
      final y = mnn.VARP.scalar<mnn.float32>(1.0);
      final x = mnn.VARP.scalar<mnn.float32>(1.0);
      expect(op.atan2(y, x).value, closeTo(0.785, 0.001)); // pi/4

      final m1 = mnn.VARP.scalar<mnn.float32>(5.0);
      final m2 = mnn.VARP.scalar<mnn.float32>(3.0);
      expect(op.mod(m1, m2).value, closeTo(2.0, 0.001));

      y.dispose();
      x.dispose();
      m1.dispose();
      m2.dispose();
    });

    test('matrix: matMul', () {
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final b = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [2, 2]); // Identity
      final c = op.matMul(a, b);

      expect(c.data, [1.0, 2.0, 3.0, 4.0]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('manipulation: stack, unstack, tile, broadcastTo', () {
      final v1 = mnn.VARP.scalar<mnn.float32>(1.0);
      final v2 = mnn.VARP.scalar<mnn.float32>(2.0);

      final s = op.stack([v1, v2]);
      expect(s.dim, [2]);
      expect(s.data, [1.0, 2.0]);

      final u = op.unstack(s);
      expect(u.length, 2);
      expect(u[0].value, 1.0);

      // Tile expects input to have same rank as multiples usually, or broadcast rules apply?
      // Using rank-1 input to be safe and explicit
      final vRank1 = mnn.VARP.fromListND<mnn.float32>([1.0], [1]);
      final t = op.tile(vRank1, mnn.VARP.fromList1D<mnn.int32>([3]));
      expect(t.dim, [3]);

      final b = op.broadcastTo(v1, mnn.VARP.fromList1D<mnn.int32>([2, 2]));
      expect(b.dim, [2, 2]);

      v1.dispose();
      v2.dispose();
      s.dispose();
      vRank1.dispose();
      t.dispose();
      b.dispose();
    });

    test('gather, scatter (basic check)', () {
      final params = mnn.VARP.fromListND<mnn.float32>([10, 20, 30, 40], [4]);
      final indices = mnn.VARP.fromListND<mnn.int32>([1, 3], [2]);

      final g = op.gather(params, indices);
      expect(g.data, [20.0, 40.0]);

      final depth = mnn.VARP.scalar<mnn.int32>(3);
      final on = mnn.VARP.scalar<mnn.float32>(1.0);
      final off = mnn.VARP.scalar<mnn.float32>(0.0);
      final ind = mnn.VARP.fromList1D<mnn.int32>([0, 2]);
      final oh = op.oneHot(ind, depth, onValue: on, offValue: off, axis: -1);

      expect(oh.dim, [2, 3]);
      expect(oh.data?[0], 1.0);
      expect(oh.data?[5], 1.0);

      params.dispose();
      indices.dispose();
      g.dispose();
      depth.dispose();
      on.dispose();
      off.dispose();
      ind.dispose();
      oh.dispose();
    });

    test('where, select, sort, argMax', () {
      final cond = mnn.VARP.fromList1D<mnn.int32>([1, 0]);
      final x = mnn.VARP.fromList1D<mnn.float32>([10, 10]);
      final y = mnn.VARP.fromList1D<mnn.float32>([20, 20]);

      // select: cond ? x : y -> [10, 20]
      final sel = op.select(cond, x, y);
      expect(sel.data, [10.0, 20.0]);

      // where
      final wh = op.where(cond);
      // returns indices where true. [1, 1] shape?
      expect(wh.dim, isNotEmpty);
      wh.dispose();

      // argMax
      final v = mnn.VARP.fromList1D<mnn.float32>([1, 5, 2]);
      final am = op.argMax(v, 0);
      expect(am.value, 1); // Index of 5 is 1

      // sort
      final s = op.sort(v, axis: 0, descend: false);
      expect(s.data, [1.0, 2.0, 5.0]);
      s.dispose();

      cond.dispose();
      x.dispose();
      y.dispose();
      sel.dispose();
      v.dispose();
      am.dispose();
    });
  });
}
