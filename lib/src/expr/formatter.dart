/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:math';
import 'package:mnn/src/expr/expr.dart';

class PrintOptions {
  final int precision;
  final int threshold;
  final int edgeitems;
  final int linewidth;
  final bool suppress;

  const PrintOptions({
    this.precision = 6,
    this.threshold = 1000,
    this.edgeitems = 3,
    this.linewidth = 75,
    this.suppress = false,
  });
}

String formatFloat(num x, int precision, bool suppress) {
  if (suppress && x.abs() < pow(10, -precision)) {
    return x.sign < 0 ? '-0.${'0' * precision}' : '0.${'0' * precision}';
  }
  return x.toStringAsFixed(precision);
}

String formatInt(int x) => x.toString();

List<int> calcColumnWidths(List<num> data, List<int> shape, PrintOptions options) {
  if (shape.length < 2) return [];
  final int rows = shape[0], cols = shape[1];
  final List<int> widths = List.filled(cols, 0);
  final int stride = cols;
  for (int c = 0; c < cols; ++c) {
    int maxw = 0;
    for (int r = 0; r < rows; ++r) {
      final int idx = r * stride + c;
      if (idx >= data.length) break;
      final v = data[idx];
      String s;
      if (v is int || (v is double && v == v.roundToDouble())) {
        s = formatInt(v.toInt());
      } else {
        s = formatFloat(v, options.precision, options.suppress);
      }
      maxw = max(maxw, s.length);
    }
    widths[c] = maxw;
  }
  return widths;
}

String formatAligned(num v, int width, PrintOptions options) {
  String s;
  if (v is int || (v is double && v == v.roundToDouble())) {
    s = formatInt(v.toInt());
  } else {
    s = formatFloat(v, options.precision, options.suppress);
  }
  return s.padLeft(width, ' ');
}

String array2string(
  List<num> data,
  List<int> shape, {
  PrintOptions options = const PrintOptions(),
  String separator = ', ',
  String prefix = '',
  String suffix = '',
}) {
  final int size = shape.isEmpty ? 0 : shape.reduce((a, b) => a * b);
  final bool summarised = size > options.threshold;
  final edge = options.edgeitems;

  final List<int> colWidths = shape.length == 2 ? calcColumnWidths(data, shape, options) : [];

  String formatArray(int index, int dim) {
    if (dim == shape.length) {
      final v = data[index];
      if (colWidths.isNotEmpty && shape.length == 2) {
        final int stride = shape[1];
        final int col = index % stride;
        return formatAligned(v, colWidths[col], options);
      }
      if (v is int) return formatInt(v);
      if (v is double) {
        if (v is int || (v == v.roundToDouble())) {
          return v.toInt().toString();
        }
        return formatFloat(v, options.precision, options.suppress);
      }
      return v.toString();
    }

    final int curDim = shape[dim];
    int stride = 1;
    for (int i = dim + 1; i < shape.length; ++i) {
      stride *= shape[i];
    }

    final List<String> items = [];
    if (summarised && curDim > 2 * edge) {
      for (int i = 0; i < edge; ++i) {
        items.add(formatArray(index + i * stride, dim + 1));
      }
      items.add('...');
      for (int i = curDim - edge; i < curDim; ++i) {
        items.add(formatArray(index + i * stride, dim + 1));
      }
    } else {
      for (int i = 0; i < curDim; ++i) {
        items.add(formatArray(index + i * stride, dim + 1));
      }
    }
    String sep;
    if (dim == shape.length - 1) {
      sep = separator;
    } else if (shape.length == 2 && dim == 0) {
      sep = ',\n ';
    } else {
      sep = ',\n${' ' * (prefix.length + dim + 1)}';
    }
    return '[${items.join(sep)}]';
  }

  if (size == 0) return '[]';

  String arrayStr = formatArray(0, 0);

  if (summarised && shape.isNotEmpty) {
    arrayStr += ', shape=$shape';
  }
  return prefix + arrayStr + suffix;
}

/// Extension methods for VARP formatting
extension VARPFormatting on VARP {
  /// Convert VARP to formatted string
  String formatString([PrintOptions? options]) {
    return array2string(
      data ?? [],
      shape ?? [],
      options: options ?? const PrintOptions(),
      prefix: 'VARP(',
      suffix: ', shape=${shape ?? []}, dtype=${dtype ?? ""})',
    );
  }

  /// Pretty print VARP to console
  void print_([PrintOptions? options]) {
    final str = formatString(options);
    // ignore: avoid_print
    print(str);
  }
}
