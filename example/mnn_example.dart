import 'dart:async';

import 'package:mnn/mnn.dart' as mnn;

void main() async {
  print('MNN version: ${mnn.version()}');
  final timer = mnn.Timer.create();
  final start = timer.current();
  print('start: $start');
  await Future.delayed(const Duration(seconds: 1));
  final end = timer.current();
  print('end: $end');
  print('duration: ${end - start}');
}
