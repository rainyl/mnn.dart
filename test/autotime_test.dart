import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

void main() async {
  test('Timer', () async {
    final timer = mnn.Timer.create();
    timer.reset();
    final start = timer.current();
    await Future.delayed(const Duration(seconds: 1));
    timer.reset();
    final end = timer.current();
    final duration = end - start;
    expect(duration, greaterThan(0));
  });
}

