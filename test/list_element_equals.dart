import 'package:test/test.dart';

ListCloseTo listCloseTo(List<num> expectedValues, [double tolerance=1.0e-8]) => ListCloseTo(expectedValues, tolerance);

class ListCloseTo extends Matcher {
  final List<num> expected;
  final double tolerance;

  const ListCloseTo(this.expected, this.tolerance);

  @override
  Description describe(Description description) {
    return description.add("$expected");
  }

  @override
  bool matches(Object? item, Map matchState) {
    final vec = item! as List<num>;
    if (vec.length != expected.length) return false;
    return vec.indexed.every((e) => e.$2 - expected[e.$1] < tolerance);
  }
}
