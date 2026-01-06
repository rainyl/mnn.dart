import 'dart:ffi' as ffi;
import 'dart:math' as math;

import 'package:mnn/cv.dart' as cv;
import 'package:mnn/expr.dart' as expr;
import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/nn.dart' as nn;
import 'package:test/test.dart';

void main() {
  test('yolo', () {
    final model = nn.Module.loadFromFile('test/data/yolo11n.mnn');
    final imgOriginal = cv.imread("test/data/18128.png", flags: cv.IMREAD_COLOR_RGB);
    var img = cv.resize(
      expr.clone(imgOriginal),
      (640, 640),
      mean: [0, 0, 0],
      norm: [1 / 255, 1 / 255, 1 / 255],
    );
    final scale = math.max(imgOriginal.shape?[0] ?? 0, imgOriginal.shape?[1] ?? 0) / 640;
    expect(img.shape, [640, 640, 3]);
    expect(img.dtype, mnn.HalideType.f32);
    expect(img.dataFormat, mnn.DimensionFormat.NHWC);
    img = expr.convert(expr.unsqueeze(img, axis: [0]), mnn.DimensionFormat.NCHW);
    expect(img.shape, [1, 3, 640, 640]);

    var output = model.forward(img);
    expect(output.shape, [1, 660, 8400]);

    output = expr.squeeze(output, axis: [0]);
    final cx = output[0];
    final cy = output[1];
    final w = output[2];
    final h = output[3];
    final probs = output["4:, :"];
    // [cx, cy, w, h] -> [x0, y0, x1, y1]
    final x0 = cx - w * expr.scalar<mnn.float32>(0.5);
    final y0 = cy - h * expr.scalar<mnn.float32>(0.5);
    final x1 = cx + w * expr.scalar<mnn.float32>(0.5);
    final y1 = cy + h * expr.scalar<mnn.float32>(0.5);
    final boxes = expr.stack([x0, y0, x1, y1], axis: 1);
    final scores = expr.reduceMax(probs, axis: [0]);
    final ids = expr.argMax(probs, 0);
    final resultIds = expr.nms(
      boxes,
      scores,
      100,
      iouThreshold: 0.45,
      scoreThreshold: 0.25,
    );
    final boxPtr = boxes.readMap<mnn.float32>();
    final scorePtr = scores.readMap<mnn.float32>();
    final resultPtr = resultIds.readMap<mnn.int32>();
    final idsPtr = ids.readMap<mnn.int32>();
    final rval = <(double, double, double, double, double, int)>[];
    for (var i = 0; i < (resultIds.size ?? 0); i++) {
      final idx = resultPtr[i];
      if (idx < 0) {
        break;
      }
      final (x0, y0, x1, y1) = (
        boxPtr[idx * 4 + 0] * scale,
        boxPtr[idx * 4 + 1] * scale,
        boxPtr[idx * 4 + 2] * scale,
        boxPtr[idx * 4 + 3] * scale,
      );
      rval.add((
        x0,
        y0,
        x1,
        y1,
        scorePtr[idx],
        idsPtr[idx],
      ));
      cv.rectangle(imgOriginal, (x0, y0), (x1, y1), cv.Scalar([0, 0, 255, 0]));
    }
    cv.imwrite("aaa.png", imgOriginal);
    output.dispose();
    // x0 y0 x1 y1 are hold by boxes, do not free separately
    // x0.dispose();
    // y0.dispose();
    // x1.dispose();
    // y1.dispose();
    boxes.dispose();
    scores.dispose();
    ids.dispose();
    resultIds.dispose();
  });
}
