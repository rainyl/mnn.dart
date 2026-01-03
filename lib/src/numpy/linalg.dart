import '../core/base.dart';
import '../expr/expr.dart';
import '../expr/op.dart' as F;
import 'numpy.dart' as np;

// Linear algebra
VARP norm(VARP x, {dynamic ord, dynamic axis, bool keepDims = false}) {
  var xx = F.clone(x);
  final ndim = xx.ndim!;

  if (axis == null) {
    if ((ord == null) || ((ord == 'f' || ord == 'fro') && ndim == 2) || (ord == 2 && ndim == 1)) {
      xx = np.ravel(xx);
      final sqnorm = np.dot<float32>(xx, xx);
      var ret = np.sqrt(sqnorm);
      if (keepDims) {
        ret = np.reshape(ret, List.filled(ndim, 1));
      }
      return ret;
    }
  }

  List<int> axes;
  if (axis == null) {
    axes = List.generate(ndim, (i) => i);
  } else if (axis is int) {
    axes = [axis];
  } else if (axis is List<int>) {
    axes = axis;
  } else {
    axes = (axis as List).cast<int>();
  }

  if (axes.length == 1) {
    if (ord == double.infinity) {
      return np.max(np.abs(xx), axis: axes, keepDims: keepDims);
    } else if (ord == double.negativeInfinity) {
      return np.min(np.abs(xx), axis: axes, keepDims: keepDims);
    } else if (ord == 0) {
      return np.sum(xx, axis: axes, keepDims: keepDims);
    } else if (ord == 1) {
      return np.sum(np.abs(xx), axis: axes, keepDims: keepDims);
    } else if (ord == null || ord == 2) {
      return np.sqrt(np.sum(xx * xx, axis: axes, keepDims: keepDims));
    } else if (ord is String) {
      throw ArgumentError('Invalid norm order for vectors');
    } else {
      final ordVal = (ord as num).toDouble();
      return np.power(
        np.sum(np.power(np.abs(xx), np.array<float32>(ordVal)), axis: axes, keepDims: keepDims),
        np.array<float32>(1.0 / ordVal),
      );
    }
  }

  throw UnimplementedError('norm not implemented for axis length ${axes.length}');
}

(VARP, VARP, VARP) svd(
  VARP a, {
  bool fullMatrices = true,
  bool computeUV = true,
  bool hermitian = false,
}) {
  final res = F.svd(a);
  return (res[1], res[0], res[2]);
}

VARP solve(VARP a, VARP b) {
  throw UnimplementedError('solve not supported');
}
