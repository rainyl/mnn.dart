import '../base.dart';
import '../expr/expr.dart';
import '../expr/op.dart' as F;

VARP random(
  List<int> shape, {
  double low = 0.0,
  double high = 1.0,
  int seed0 = 0,
  int seed1 = 0,
}) {
  return F.randomUniform<float32>(
    F.constant<int32>(shape, [shape.length], format: DimensionFormat.NCHW),
    low: low,
    high: high,
    seed0: seed0,
    seed1: seed1,
  );
}

VARP randint(
  int low, {
  int? high,
  List<int> size = const [],
  int seed0 = 0,
  int seed1 = 0,
}) {
  final low_ = high == null ? 0 : low;
  high ??= low;
  return F.cast<int32>(
    F.randomUniform<float32>(
      F.constant<int32>(size, [size.length], format: DimensionFormat.NCHW),
      low: low_.toDouble(),
      high: high.toDouble(),
      seed0: seed0,
      seed1: seed1,
    ),
  );
}
