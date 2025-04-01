import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'g/mnn.g.dart' as c;

String version() {
  final p = c.mnn_get_version().cast<Utf8>();
  final str = p.toDartString();
  calloc.free(p);
  return str;
}
