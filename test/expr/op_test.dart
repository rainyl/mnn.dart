import 'dart:math' as math;

import 'package:mnn/expr.dart' as expr;
import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/numpy.dart' as np;
import 'package:test/test.dart';

import '../list_element_equals.dart';

List<double> _refNormalize(
  List<double> src,
  int batch,
  int channel,
  int area,
  List<double> scale,
  double eps,
) {
  final dst = List.filled(batch * channel * area, 0.0);
  for (int b = 0; b < batch; b++) {
    for (int x = 0; x < area; x++) {
      final dstX = b * area * channel + x;
      final srcX = b * area * channel + x;
      var sumSquare = 0.0;
      for (int c = 0; c < channel; c++) {
        sumSquare += src[srcX + area * c] * src[srcX + area * c];
      }
      final normalValue = 1.0 / math.sqrt(sumSquare + eps);
      for (int c = 0; c < channel; c++) {
        dst[dstX + area * c] = src[srcX + area * c] * normalValue * scale[c];
      }
    }
  }
  return dst;
}

void main() {
  group('BinaryOPs', () {
    test('add, subtract, multiply, divide, pow', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(expr.add(a, b).value, closeTo(12.0, 0.001));
      expect(expr.subtract(a, b).value, closeTo(8.0, 0.001));
      expect(expr.multiply(a, b).value, closeTo(20.0, 0.001));
      expect(expr.divide(a, b).value, closeTo(5.0, 0.001));
      expect(expr.pow(a, b).value, closeTo(100.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('minimum, maximum', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(expr.minimum(a, b).value, closeTo(2.0, 0.001));
      expect(expr.maximum(a, b).value, closeTo(10.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('biasAdd, squaredDifference', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.0);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(expr.biasAdd(a, b).value, closeTo(12.0, 0.001));
      expect(expr.squaredDifference(a, b).value, closeTo(64.0, 0.001)); // (10-2)^2

      a.dispose();
      b.dispose();
    });

    test('floorDiv, floorMod', () {
      final a = mnn.VARP.scalar<mnn.float32>(10.5);
      final b = mnn.VARP.scalar<mnn.float32>(2.0);

      expect(expr.floorDiv(a, b).value, closeTo(5.0, 0.001));
      expect(expr.floorMod(a, b).value, closeTo(0.5, 0.001));

      a.dispose();
      b.dispose();
    });

    // Note: Comparison and Logical ops return int (0 or 1) usually in MNN, or float 0.0/1.0 depending on type
    // Checking with cast to Int32 if needed, or check float value if that's what returns
    test('Comparison and Logical', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.0);
      final b = mnn.VARP.scalar<mnn.float32>(0.0);

      // greater
      expect(expr.greater(a, b).value, 1);
      expect(expr.greater(b, a).value, 0);

      // greaterEqual
      expect(expr.greaterEqual(a, b).value, 1);
      expect(expr.greaterEqual(a, a).value, 1);

      // less
      expect(expr.less(b, a).value, 1);

      // lessEqual
      expect(expr.lessEqual(b, a).value, 1);
      expect(expr.lessEqual(a, a).value, 1);

      // equal
      expect(expr.equal(a, a).value, 1);
      expect(expr.equal(a, b).value, 0);

      // notEqual
      expect(expr.notEqual(a, b).value, 1);
      expect(expr.notEqual(a, a).value, 0);

      a.dispose();
      b.dispose();
    });

    test('GridSample, CosineSimilarity', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final grid = mnn.VARP.fromListND<mnn.float32>([0, 0], [1, 1, 1, 2]); // 1x1x1x2 grid
      final gs = expr.GridSample(x, grid);
      expect(gs.dim, [1, 1, 1, 1]);
      gs.dispose();
      grid.dispose();

      final x1 = mnn.VARP.fromListND<mnn.float32>([1, 0], [2]);
      final x2 = mnn.VARP.fromListND<mnn.float32>([1, 0], [2]);
      final dim = mnn.VARP.scalar<mnn.int32>(0);
      final cs = expr.CosineSimilarity(x1, x2, dim);
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

      expect(expr.abs(a).value, closeTo(5.0, 0.001));
      expect(expr.negative(a).value, closeTo(5.0, 0.001));
      expect(expr.sign(a).value, closeTo(-1.0, 0.001));

      a.dispose();
    });

    test('floor, ceil, round', () {
      final a = mnn.VARP.scalar<mnn.float32>(3.6);

      expect(expr.floor(a).value, closeTo(3.0, 0.001));
      expect(expr.ceil(a).value, closeTo(4.0, 0.001));
      expect(expr.round(a).value, closeTo(4.0, 0.001));

      a.dispose();
    });

    test('square, sqrt, rsqrt', () {
      final a = mnn.VARP.scalar<mnn.float32>(4.0);

      expect(expr.square(a).value, closeTo(16.0, 0.001));
      expect(expr.sqrt(a).value, closeTo(2.0, 0.001));
      expect(expr.rsqrt(a).value, closeTo(0.5, 0.001));

      a.dispose();
    });

    test('trig functions', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(expr.sin(a).value, closeTo(0.0, 0.001));
      expect(expr.cos(a).value, closeTo(1.0, 0.001));
      expect(expr.tan(a).value, closeTo(0.0, 0.001));
      a.dispose();
    });

    test('exp, log, reciprocal', () {
      final a = mnn.VARP.scalar<mnn.float32>(1.0);

      expect(expr.exp(a).value, closeTo(2.718, 0.001));
      expect(expr.log(expr.exp(a)).value, closeTo(1.0, 0.001));

      final b = mnn.VARP.scalar<mnn.float32>(2.0);
      expect(expr.reciprocal(b).value, closeTo(0.5, 0.001));

      a.dispose();
      b.dispose();
    });

    test('hyperbolic functions', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(expr.sinh(a).value, closeTo(0.0, 0.001));
      expect(expr.cosh(a).value, closeTo(1.0, 0.001));

      // tanh is already tested elsewhere but adding here for completeness
      expect(expr.tanh(a).value, closeTo(0.0, 0.001));

      a.dispose();
    });

    test('inverse trig/hyperbolic', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);

      expect(expr.asin(a).value, closeTo(0.0, 0.001));
      expect(expr.atan(a).value, closeTo(0.0, 0.001));
      expect(expr.asinh(a).value, closeTo(0.0, 0.001));
      expect(expr.atanh(a).value, closeTo(0.0, 0.001));

      final b = mnn.VARP.scalar<mnn.float32>(1.0);
      expect(expr.acos(b).value, closeTo(0.0, 0.001));
      expect(expr.acosh(b).value, closeTo(0.0, 0.001));

      a.dispose();
      b.dispose();
    });

    test('erf, erfc, erfinv', () {
      final a = mnn.VARP.scalar<mnn.float32>(0.0);
      expect(expr.erf(a).value, closeTo(0.0, 0.001));
      expect(expr.erfc(a).value, closeTo(1.0, 0.001));

      // erfinv(0) = 0
      expect(expr.erfinv(a).value, closeTo(0.0, 0.001));

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
      final sum0 = expr.reduceSum(x, axis: [0]);
      expect(sum0.data, [4.0, 6.0]);

      // Reduce along axis 1 (rows): [1+2, 3+4] = [3, 7]
      final sum1 = expr.reduceSum(x, axis: [1]);
      expect(sum1.data, [3.0, 7.0]);

      final mean = expr.reduceMean(x, axis: [0, 1]); // scalar 2.5
      expect(mean.value, closeTo(2.5, 0.001));

      final max = expr.reduceMax(x, axis: [0, 1]);
      expect(max.value, closeTo(4.0, 0.001));

      final min = expr.reduceMin(x, axis: [0, 1]);
      expect(min.value, closeTo(1.0, 0.001));

      final variance = expr.reduceVariance(x, axis: [0, 1]);
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

      expect(expr.reduceSumMutable(x).data, [10.0]);
      final sum = expr.reduceSumMutable(x, axis: expr.scalar<mnn.int32>(0));
      expect(sum.data, [4.0, 6.0]);

      expect(expr.reduceMeanMutable(x).data, [2.5]);
      final mean = expr.reduceMeanMutable(x, axis: expr.scalar<mnn.int32>(1));
      expect(mean.data, [1.5, 3.5]);

      expect(expr.reduceMaxMutable(x).data, [4.0]);
      final max = expr.reduceMaxMutable(x, axis: expr.scalar<mnn.int32>(0));
      expect(max.data, [3.0, 4.0]);

      expect(expr.reduceMinMutable(x).data, [1.0]);
      final min = expr.reduceMinMutable(x, axis: expr.scalar<mnn.int32>(0));
      expect(min.data, [1.0, 2.0]);

      expect(expr.reduceProdMutable(x).data, [24.0]);
      final prod = expr.reduceProdMutable(x, axis: expr.scalar<mnn.int32>(0));
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

      final any = expr.reduceAnyMutable(x, axis: expr.scalar<mnn.int32>(1)); // [1, 1]
      expect(any.data, [1, 1]);

      final all = expr.reduceAllMutable(x, axis: expr.scalar<mnn.int32>(1)); // [0, 1]
      expect(all.data, [0, 1]);

      x.dispose();
      any.dispose();
      all.dispose();
    });

    test('reduceProd', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final shape = [4];
      final x = mnn.VARP.fromListND<mnn.float32>(data, shape, format: mnn.DimensionFormat.NCHW);

      final prod = expr.reduceProd(x, axis: [0]);
      expect(prod.value, closeTo(24.0, 0.001));

      x.dispose();
      prod.dispose();
    });

    test('reduceAny, reduceAll', () {
      // 0 = false, 1 = true
      final data = [0, 1, 1, 1];
      final shape = [4];
      final x = mnn.VARP.fromListND<mnn.int32>(data, shape, format: mnn.DimensionFormat.NCHW);

      final any = expr.reduceAny(x, axis: [0]);
      expect(any.value, 1);

      final all = expr.reduceAll(x, axis: [0]);
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
      expect(expr.prod(a, b).data, [2.0, 6.0, 12.0, 20.0]);

      // sum: x + y
      expect(expr.sum(a, b).data, [3.0, 5.0, 7.0, 9.0]);

      // max: max(x, y)
      expect(expr.max(a, b).data, [2.0, 3.0, 4.0, 5.0]);

      // sub: x - y
      expect(expr.sub(a, b).data, [-1.0, -1.0, -1.0, -1.0]);

      a.dispose();
      b.dispose();
    });
  });

  group('OtherOPs', () {
    test('cast', () {
      final a = mnn.VARP.scalar<mnn.float32>(3.14);
      final b = expr.cast<mnn.int32>(a);
      expect(b.value, 3);

      a.dispose();
      b.dispose();
    });

    test('concat', () {
      final v1 = mnn.VARP.fromList1D<mnn.float32>([1, 2]);
      final v2 = mnn.VARP.fromList1D<mnn.float32>([3, 4]);

      // concat along axis 0: [1, 2, 3, 4]
      final c = expr.concat([v1, v2], 0);
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
      final y = expr.transpose(x, [1, 0]);

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

      final y = expr.slice(x, starts, sizes);
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

      final filled = expr.fill(dims, value);
      expect(filled.dim, [2, 2]);
      expect(filled.data, [5.0, 5.0, 5.0, 5.0]);

      final shapeVar = expr.shape(filled);
      expect(shapeVar.data, [2, 2]);

      dims.dispose();
      value.dispose();
      filled.dispose();
      shapeVar.dispose();
    });

    test('reshape', () {
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final b = expr.reshape(a, [4]);
      expect(b.dim, [4]);

      a.dispose();
      b.dispose();
    });

    test('normalize', () {
      // normalize
      final x = mnn.VARP.fromListND<mnn.float32>(
        [-1.0, -2.0, 3.0, 4.0],
        [1, 2, 2, 1],
        format: mnn.DimensionFormat.NCHW,
      );
      final norm = expr.normalize(x, 0, 0, 0.0, [0.5, 0.5]);
      final expected = _refNormalize([-1.0, -2.0, 3.0, 4.0], 1, 2, 2, [0.5, 0.5], 0.0);
      expect(norm.data, listCloseTo(expected, 0.001));
      norm.dispose();
    });

    test('argmin', () {
      // argMin
      final am = expr.argMin(mnn.VARP.fromListND<mnn.float32>([-1.0, -2.0, 3.0, 4.0], [1, 4]), 0);
      expect(am.data, listCloseTo([0, 0, 0, 0], 0.001));
      am.dispose();
    });

    test('batchMatMul', () {
      // batchMatMul
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 2, 2]);
      final b = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [1, 2, 2]);
      final bmm = expr.batchMatMul(a, b);
      expect(bmm.dim, [1, 2, 2]);
      bmm.dispose();
      a.dispose();
      b.dispose();
    });

    test('unravelIndex', () {
      // unravelIndex
      final indices = mnn.VARP.scalar<mnn.int32>(5);
      final dims = mnn.VARP.fromList1D<mnn.int32>([2, 3]);
      final ui = expr.unravelIndex(indices, dims);
      // 5 in [2, 3] -> (1, 2)
      expect(ui.data, [1, 2]);
      indices.dispose();
      dims.dispose();
      ui.dispose();
    });

    test('linSpace', () {
      // linSpace
      final start = mnn.VARP.scalar<mnn.float32>(0);
      final stop = mnn.VARP.scalar<mnn.float32>(10);
      final num = mnn.VARP.scalar<mnn.int32>(5);
      final ls = expr.linSpace(start, stop, num);
      expect(ls.dim, [5]);
      start.dispose();
      stop.dispose();
      num.dispose();
      ls.dispose();
    });

    test('randomUniform', () {
      // randomUniform
      final shape = mnn.VARP.fromList1D<mnn.int32>([2, 2]);
      final rnd = expr.randomUniform<mnn.float32>(shape);
      expect(rnd.dim, [2, 2]);
      shape.dispose();
      rnd.dispose();
    });

    test('cumSum, cumProd', () {
      // cumSum, cumProd
      final cs = expr.cumSum(mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0, 4.0]), 0);
      expect(cs.data, [1.0, 3.0, 6.0, 10.0]);

      final cp = expr.cumProd(mnn.VARP.fromList1D<mnn.float32>([1.0, 2.0, 3.0, 4.0]), 0);
      expect(cp.data, [1.0, 2.0, 6.0, 24.0]);
      cs.dispose();
      cp.dispose();
    });

    test('histogram', () {
      // histogram
      // TODO: add more format according to HistogramTest.cpp
      // input x is [1, 2, 3, 4]
      // bin=2, min=0, max=5. bins: [0, 2.5), [2.5, 5]
      // 1, 2 in bin 0; 3, 4 in bin 1
      final hist = expr.histogram(
        expr.constant<mnn.float32>([1.0, 2.0, 1.0, 4.0], [4]),
        4,
        0,
        3,
        channel: -1,
      );
      expect(hist.data, [0.0, 2.0, 1.0, 0.0]);
      hist.dispose();
    });

    test('svd', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [2, 2]);
      final res = expr.svd(x); // returns [S, U, V] or similar
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
      final sc = expr.scatterND(indices, updates, shape);
      expect(sc.data, [10.0, 20.0, 0.0, 0.0]);
      indices.dispose();
      updates.dispose();
      shape.dispose();
      sc.dispose();

      // scatterElements
      final data = mnn.VARP.fromListND<mnn.float32>([0, 0, 0], [3]);
      final idx = mnn.VARP.fromListND<mnn.int32>([1], [1]);
      final upd = mnn.VARP.fromListND<mnn.float32>([5], [1]);
      final se = expr.scatterElements(data, idx, upd);
      expect(se.data, [0.0, 5.0, 0.0]);
      data.dispose();
      idx.dispose();
      upd.dispose();
      se.dispose();
    });
  });

  group('Neural Network Ops', () {
    test('input', () {
      // input
      final inp = expr.input<mnn.float32>([1, 3, 224, 224]);
      expect(inp.dim, [1, 3, 224, 224]);
      inp.dispose();
    });

    test('clone', () {
      // clone
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final cl = expr.clone(x);
      expect(cl.dim, [1, 1, 2, 2]);
      expect(cl.data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      cl.dispose();
    });

    test('conv2d', () {
      // conv2d
      final x = expr.convert(
        np.random([1, 4, 16, 16]).astype(dtype: mnn.HalideType.f32),
        mnn.DimensionFormat.NC4HW4,
      );
      final w = np.random([2, 4, 3, 3]).astype(dtype: mnn.HalideType.f32);
      final b = np.random([2]).astype(dtype: mnn.HalideType.f32);

      final conv = expr.convert(expr.conv2d(x, w, bias: b), mnn.DimensionFormat.NCHW);
      expect(conv.ndim, 4);
      expect(conv.dim, [1, 2, 14, 14]);
      conv.dispose();
      w.dispose();
      b.dispose();
      x.dispose();
    });

    test('conv2dTranspose', () {
      // conv2dTranspose
      final x = expr.convert(
        np.random([1, 4, 16, 16]).astype(dtype: mnn.HalideType.f32),
        mnn.DimensionFormat.NC4HW4,
      );
      final w = np.random([4, 2, 3, 3]).astype(dtype: mnn.HalideType.f32);
      final b = np.random([2]).astype(dtype: mnn.HalideType.f32);

      final ct = expr.convert(expr.conv2dTranspose(x, w, bias: b), mnn.DimensionFormat.NCHW);
      expect(ct.ndim, 4);
      expect(ct.dim, [1, 2, 18, 18]);
      ct.dispose();
      w.dispose();
      b.dispose();
      x.dispose();
    });

    test('scale', skip: "TODO: fails", () {
      // scale
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final sc = expr.scale(x, 1, [2.0], [0.0]);
      expect(sc.data, listCloseTo([2.0, 4.0, 6.0, 8.0], 0.001));
      sc.dispose();
    });

    test('ReLU6, PReLU, softMax, softPlus, softSign', () {
      // ReLU6, PReLU, softMax, softPlus, softSign
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      expect(
        expr.ReLU(mnn.VARP.fromList1D<mnn.float32>([-1.0, 2.0, 3.0, 4.0])).data,
        listCloseTo([0.0, 2.0, 3.0, 4.0], 0.001),
      );
      expect(expr.ReLU6(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(expr.PReLU(x, [0.1]).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(expr.softMax(x).data, listCloseTo([0.2689, 0.73105, 0.26894, 0.73105], 0.001));
      expect(expr.softPlus(x).data, listCloseTo([1.3133, 2.1269, 3.0485, 4.0181], 0.001));
      expect(expr.softSign(x).data, listCloseTo([0.5000, 0.6667, 0.7500, 0.8000], 0.001));
    });

    test('split', () {
      // split
      // Input is 1x1x2x2. Axis 2 is height (2). Split into [1, 1].
      // Should result in two 1x1x1x2 tensors.
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final parts = expr.split(x, [1, 1], axis: 2);
      expect(parts.length, 2);
      for (final p in parts) {
        expect(p.dim, [1, 1, 1, 2]);
        p.dispose();
      }

      x.dispose();
    });

    test('stridedSlice', () {
      // stridedSlice
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final begin = mnn.VARP.fromList1D<mnn.int32>([0, 0, 0, 0]);
      final end = mnn.VARP.fromList1D<mnn.int32>([1, 1, 1, 1]);
      final stride = mnn.VARP.fromList1D<mnn.int32>([1, 1, 1, 1]);
      final ss = expr.stridedSlice(x, begin, end, stride, 0, 0, 0, 0, 0);
      expect(ss.dim, [1, 1, 1, 1]);
      ss.dispose();
      begin.dispose();
      end.dispose();
      stride.dispose();
    });

    test('convert', () {
      // convert, transpose1, channelShuffle, changeInputFormat
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final converted = expr.convert(x, mnn.DimensionFormat.NC4HW4);
      expect(converted.shape, [1, 1, 2, 2]);
      expect(
        converted.data,
        listCloseTo([1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0], 0.001),
      );
    });

    test('transpose1', () {
      // transpose1 takes VARP perm
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final perm = mnn.VARP.fromList1D<mnn.int32>([0, 1, 3, 2]);
      expect(expr.transpose1(x, perm).dim, [1, 1, 2, 2]);
      perm.dispose();
    });

    test('channelShuffle', () {
      final x = mnn.VARP.fromListND<mnn.float32>(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [1, 1, 2, 4],
        format: mnn.DimensionFormat.NHWC,
      );
      final shuffled = expr.convert(expr.channelShuffle(x, 2), mnn.DimensionFormat.NHWC);
      expect(shuffled.shape, [1, 1, 2, 4]);
      expect(shuffled.data, listCloseTo([0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0], 0.001));
    });

    test('changeInputFormat', skip: "Fails", () {
      // TODO: fails
      final data = <double>[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
      final input = expr.input<mnn.float32>([1, 1, 2, 4]);
      final changed = expr.changeInputFormat(input, mnn.DimensionFormat.NCHW);
      input.data = data;
      expect(changed.data, listCloseTo([1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], 0.001));
    });

    test('reverse, reverseSequence', () {
      // reverse, reverseSequence
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final axis = mnn.VARP.scalar<mnn.int32>(3);
      expect(expr.reverse(x, axis).data, [2.0, 1.0, 4.0, 3.0]);
      axis.dispose();

      final seqLen = mnn.VARP.fromList1D<mnn.int32>([1]);
      expect(expr.reverseSequence(x, seqLen, 0, 1).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      seqLen.dispose();
    });

    test('crop, resize, cropAndResize', () {
      // crop, resize, cropAndResize
      // crop
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final cropSize = mnn.VARP.fromListND<mnn.int32>([1, 1], [2]);
      // offset is List<int>
      final cr = expr.crop(x, cropSize, 2, [0, 0]);
      expect(cr.dim, [1]);
      cr.dispose();

      // resize
      final res = expr.resize(x, 2.0, 2.0);
      expect(res.dim, [1, 1, 4, 4]);
      res.dispose();

      // TODO
      // cropAndResize
    });

    test('expandDims squeeze', () {
      // expandDims
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [1, 1, 2, 2]);
      final ed = expr.expandDims(x, 0);
      expect(ed.dim, [1, 1, 1, 2, 2]);

      final sq = expr.squeeze(ed, axis: [0]);
      expect(sq.dim, [1, 1, 2, 2]);
      sq.dispose();
    });

    test('batchToSpaceND', skip: "TODO: Fails", () {
      final x = mnn.VARP.fromListND<mnn.float32>(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [4, 1, 1, 3],
        format: mnn.DimensionFormat.NCHW,
      );
      final input = expr.convert(x, mnn.DimensionFormat.NC4HW4);
      final blockShape = expr.constant<mnn.int32>([2, 2], [2], format: mnn.DimensionFormat.NCHW);
      final crops = expr.constant<mnn.int32>([0, 0, 0, 0], [2, 2], format: mnn.DimensionFormat.NCHW);
      final expectedOutput = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
      final tmp = expr.batchToSpaceND(input, blockShape, crops);
      final output = expr.convert(tmp, mnn.DimensionFormat.NHWC);
      expect(output.shape, [1, 2, 2, 3]);
      expect(output.data, listCloseTo(expectedOutput, 0.01));

      x.dispose();
    });

    test('spaceToBatchND', () {
      final x = expr.constant<mnn.float32>(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        [3, 1, 2, 2],
        format: mnn.DimensionFormat.NCHW,
      );
      final input = expr.convert(x, mnn.DimensionFormat.NC4HW4);
      final blockShape = expr.constant<mnn.int32>([2, 2], [2], format: mnn.DimensionFormat.NCHW);
      final paddings = expr.constant<mnn.int32>([0, 0, 0, 0], [2, 2], format: mnn.DimensionFormat.NCHW);
      final tmp = expr.spaceToBatchND(input, blockShape, paddings);
      final expectedOutput = [1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 4.0, 8.0, 12.0];
      final output = expr.convert(tmp, mnn.DimensionFormat.NHWC);
      expect(output.shape, [12, 1, 1, 1]);
      expect(output.data, listCloseTo(expectedOutput, 0.01));
    });

    test('gatherND', () {
      {
        final params = expr.constant<mnn.float32>(
          [-1.0, -2.0, 3.0, 4.0],
          [2, 2],
          format: mnn.DimensionFormat.NCHW,
        );
        final indices = expr.constant<mnn.int32>([0, 0, 1, 1], [2, 2], format: mnn.DimensionFormat.NCHW);
        final output = expr.gatherND(params, indices);
        expect(output.shape, [2]);
        expect(output.data, listCloseTo([-1.0, 4.0], 0.001));
      }
      {
        final params = expr.constant<mnn.float32>(
          [1, 2, 3, 4, 5, 6, 7, 8],
          [2, 2, 2],
          format: mnn.DimensionFormat.NCHW,
        );
        final indices = expr.constant<mnn.int32>(
          [0, 0, 1, 1, 1, 0],
          [2, 3],
          format: mnn.DimensionFormat.NCHW,
        );
        final output = expr.gatherND(params, indices);
        expect(output.shape, [2]);
        expect(output.data, listCloseTo([2, 7], 0.001));
      }
    });

    test('gatherElements', () {
      {
        final data = expr.constant<mnn.float32>(
          [1, 2, 3, 4, 5, 6, 7, 8, 9],
          [3, 3],
          format: mnn.DimensionFormat.NCHW,
        );
        final indices = expr.constant<mnn.int32>(
          [1, 2, 0, 2, 0, 0],
          [2, 3],
          format: mnn.DimensionFormat.NCHW,
        );
        final output = expr.gatherElements(data, indices);
        expect(output.shape, [2, 3]);
        expect(output.data, listCloseTo([4, 8, 3, 7, 2, 3], 0.001));
      }

      {
        final data = expr.constant<mnn.float32>(
          [1, 2, 3, 4],
          [2, 2],
          format: mnn.DimensionFormat.NCHW,
        );
        final indices = expr.constant<mnn.int32>(
          [0, 0, 1, 0],
          [2, 2],
          format: mnn.DimensionFormat.NCHW,
        );
        final output = expr.gatherElements(data, indices, axis: expr.scalar<mnn.int32>(1));
        expect(output.shape, [2, 2]);
        expect(output.data, listCloseTo([1, 1, 4, 3], 0.001));
      }
    });

    test('selu, elu, threshold', () {
      // selu, elu, threshold
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      expect(expr.selu(x, 1.0, 1.0).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(expr.elu(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      expect(expr.threshold(x).data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
    });

    test('size, rank', () {
      // size, rank
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      expect(expr.size(x).value, 4);
      expect(expr.rank(x).value, 1);
    });

    test('matrixBandPart', skip: "TODO: fails", () {
      // matrixBandPart
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      final mbp = expr.matrixBandPart(x, mnn.VARP.scalar<mnn.int32>(0), mnn.VARP.scalar<mnn.int32>(0));
      expect(mbp.value, isNotNull);
      mbp.dispose();
    });

    test('moments', () {
      // moments
      // returns [mean, variance]
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      final m = expr.moments(x, [0], mnn.VARP.scalar<mnn.float32>(0), true);
      expect(m.length, 2);
      for (final v in m) {
        v.dispose();
      }
    });

    test('setDiff1D', () {
      // setDiff1D
      final y = mnn.VARP.fromList1D<mnn.int32>([1, 2]);
      final sd = expr.setDiff1D(mnn.VARP.fromList1D<mnn.int32>([1, 2, 3, 4]), y); // Should return 3, 4
      expect(sd.data, [3.0, 4.0]);
      sd.dispose();
      y.dispose();
    });

    test('zerosLike', () {
      // zerosLike
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      final zl = expr.zerosLike(x);
      expect(zl.data, [0.0, 0.0, 0.0, 0.0]);
      zl.dispose();
    });

    test('range', () {
      // range
      final r = expr.range(
        mnn.VARP.scalar<mnn.float32>(0),
        mnn.VARP.scalar<mnn.float32>(5),
        mnn.VARP.scalar<mnn.float32>(1),
      );
      expect(r.data, [0.0, 1.0, 2.0, 3.0, 4.0]);
      r.dispose();
    });

    test('Permute', () {
      // Permute
      final x2 = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final p = expr.Permute(x2, [1, 0]);
      expect(p.dim, [2, 2]);
      x2.dispose();
      p.dispose();
    });

    test('interp', () {
      // interp
      // TODO:
      // expr.interp([x], ...)
    });

    test('zeroGrad', () {
      // zeroGrad
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);
      final zg = expr.zeroGrad(x);
      expect(zg.data, isNotEmpty);
      zg.dispose();
    });

    test('floatToInt8, int8ToFloat', () {
      final x = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [4]);

      // floatToInt8, int8ToFloat
      // TODO: fails in some occasions
      // final scale = mnn.VARP.scalar<mnn.float32>(1.0);
      // final i8 = expr.floatToInt8(x, scale);
      // expect(i8.data, listCloseTo([1, 2, 3, 4], 0.001));
      // final f = expr.int8ToFloat(i8, scale);
      // expect(f.data, listCloseTo([1.0, 2.0, 3.0, 4.0], 0.001));
      // i8.dispose();
      // f.dispose();
      // scale.dispose();

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
      final output = expr.maxPool(
        input,
        [2, 2],
        stride: [2, 2],
        pad: expr.PaddingMode.VALID,
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
      final output = expr.avgPool(input, [2, 2], stride: [2, 2], pad: expr.PaddingMode.VALID);
      expect(output.ndim, 4);
      expect(output.data, listCloseTo([2.5], 0.001));

      input.dispose();
      output.dispose();
    });

    test('pad', () {
      final input = mnn.VARP.fromListND<mnn.float32>([1], [1, 1]);
      // Pad 1 on all sides: [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
      final paddings = mnn.VARP.fromListND<mnn.int32>([1, 1, 1, 1], [2, 2]);
      final output = expr.pad(input, paddings);
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
      expect(expr.gelu(a).value, isNotNull);
      expect(expr.sigmoid(a).value, closeTo(0.731, 0.001));
      expect(expr.silu(a).value, isNotNull);
      expect(expr.hardswish(a).value, isNotNull);
      expect(expr.log1p(a).value, closeTo(0.693, 0.001));
      expect(expr.expm1(a).value, closeTo(1.718, 0.001));

      a.dispose();
    });

    test('logic & bits', () {
      final a = mnn.VARP.scalar<mnn.int32>(1); // 01
      final b = mnn.VARP.scalar<mnn.int32>(3); // 11

      // bitwiseAnd: 1 & 3 = 1
      expect(expr.bitwiseAnd(a, b).value, 1);
      // bitwiseOr: 1 | 3 = 3
      expect(expr.bitwiseOr(a, b).value, 3);
      // bitwiseXor: 1 ^ 3 = 2
      expect(expr.bitwiseXor(a, b).value, 2);

      // logicalOr
      final t = mnn.VARP.scalar<mnn.int32>(1);
      final f = mnn.VARP.scalar<mnn.int32>(0);
      expect(expr.logicalOr(t, f).value, 1);

      a.dispose();
      b.dispose();
      t.dispose();
      f.dispose();
    });

    test('math: atan2, mod', () {
      final y = mnn.VARP.scalar<mnn.float32>(1.0);
      final x = mnn.VARP.scalar<mnn.float32>(1.0);
      expect(expr.atan2(y, x).value, closeTo(0.785, 0.001)); // pi/4

      final m1 = mnn.VARP.scalar<mnn.float32>(5.0);
      final m2 = mnn.VARP.scalar<mnn.float32>(3.0);
      expect(expr.mod(m1, m2).value, closeTo(2.0, 0.001));

      y.dispose();
      x.dispose();
      m1.dispose();
      m2.dispose();
    });

    test('matrix: matMul', () {
      final a = mnn.VARP.fromListND<mnn.float32>([1, 2, 3, 4], [2, 2]);
      final b = mnn.VARP.fromListND<mnn.float32>([1, 0, 0, 1], [2, 2]); // Identity
      final c = expr.matMul(a, b);

      expect(c.data, [1.0, 2.0, 3.0, 4.0]);

      a.dispose();
      b.dispose();
      c.dispose();
    });

    test('manipulation: stack, unstack, tile, broadcastTo', () {
      final v1 = mnn.VARP.scalar<mnn.float32>(1.0);
      final v2 = mnn.VARP.scalar<mnn.float32>(2.0);

      final s = expr.stack([v1, v2]);
      expect(s.dim, [2]);
      expect(s.data, [1.0, 2.0]);

      final u = expr.unstack(s);
      expect(u.length, 2);
      expect(u[0].value, 1.0);

      // Tile expects input to have same rank as multiples usually, or broadcast rules apply?
      // Using rank-1 input to be safe and explicit
      final vRank1 = mnn.VARP.fromListND<mnn.float32>([1.0], [1]);
      final t = expr.tile(vRank1, mnn.VARP.fromList1D<mnn.int32>([3]));
      expect(t.dim, [3]);

      final b = expr.broadcastTo(v1, mnn.VARP.fromList1D<mnn.int32>([2, 2]));
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

      final g = expr.gather(params, indices);
      expect(g.data, [20.0, 40.0]);

      final depth = mnn.VARP.scalar<mnn.int32>(3);
      final on = mnn.VARP.scalar<mnn.float32>(1.0);
      final off = mnn.VARP.scalar<mnn.float32>(0.0);
      final ind = mnn.VARP.fromList1D<mnn.int32>([0, 2]);
      final oh = expr.oneHot(ind, depth, onValue: on, offValue: off, axis: -1);

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
      final sel = expr.select(cond, x, y);
      expect(sel.data, [10.0, 20.0]);

      // where
      final wh = expr.where(cond);
      // returns indices where true. [1, 1] shape?
      expect(wh.dim, isNotEmpty);
      wh.dispose();

      // argMax
      final v = mnn.VARP.fromList1D<mnn.float32>([1, 5, 2]);
      final am = expr.argMax(v, 0);
      expect(am.value, 1); // Index of 5 is 1

      // sort
      final s = expr.sort(v, axis: 0, descend: false);
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
