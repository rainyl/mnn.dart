import 'dart:ffi';
import 'dart:typed_data';

import 'package:mnn/mnn.dart' as mnn;
import 'package:dartcv4/dartcv.dart' as cv;

mnn.Interpreter loadModel(String modelPath) {
  final net = mnn.Interpreter.fromFile(modelPath);
  net.setSessionMode(mnn.SessionMode.Session_Backend_Fix);
  // net.setSessionHint(mnn.HintMode.MAX_TUNING_NUMBER, 5);
  return net;
}

List<List<(int, double)>> inference(
  mnn.Session session,
  List<String> imagePaths, {
  int topK = 3,
}) {
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  final input = session.getInput();
  if (input == null || input.isEmpty) {
    print("Can't get input tensor");
    return [];
  }
  session.interpreter.resizeTensor(input, [imagePaths.length, 3, 224, 224]);
  session.resize();
  final output = session.getOutput();
  final (B, C, H, W) = (imagePaths.length, 3, 224, 224);

  if (output == null || output.isEmpty || output.elementSize == 0) {
    throw Exception("Resize error, the model can't run batch: $B");
  }

  final memoryUsage = session.memoryInfo;
  final flops = session.flopsInfo;
  final backendType = session.backendsInfo;
  print(
    'Session Info:'
    '  memory use ${memoryUsage.toStringAsFixed(2)} MB,'
    '  flops is ${flops.toStringAsFixed(2)} M,'
    '  backendType is $backendType,'
    '  batch size = $B',
  );

  final nchwTensor = mnn.Tensor.fromTensor(
    input,
    dimType: mnn.DimensionType.MNN_CAFFE,
  );
  for (var i = 0; i < imagePaths.length; i++) {
    if (!cv.haveImageReader(imagePaths[i])) {
      throw Exception('OpenCV has no reader for: ${imagePaths[i]}');
    }
    final img = cv.imread(imagePaths[i], flags: cv.IMREAD_COLOR);
    if (img.isEmpty) {
      throw Exception('Can\'t read image: ${imagePaths[i]}');
    }
    cv.cvtColor(img, cv.COLOR_BGR2RGB, dst: img);
    cv.resize(img, (W, H), dst: img, interpolation: cv.INTER_LINEAR);
    img.convertTo(cv.MatType.CV_32FC3, alpha: 1 / 255.0, inplace: true);
    // img.subtract(cv.Scalar(mean[0], mean[1], mean[2]), inplace: true);
    // img.divideScalar(cv.Scalar(std[0], std[1], std[2]), inplace: true);
    final temp = img.data.buffer.asFloat32List();
    final pixData = Float32List.fromList(List.filled(temp.length, 0));
    final frameSize = W * H;
    for (var i = 0; i < frameSize; i++) {
      pixData[i] = temp[i * 3];
      pixData[i + frameSize] = temp[i * 3 + 1];
      pixData[i + frameSize * 2] = temp[i * 3 + 2];
    }
    nchwTensor.setImageBytes(i, pixData.buffer.asUint8List());
  }

  input.copyFromHost(nchwTensor);
  session.run();

  final outputUser = mnn.Tensor.fromTensor(
    output,
    dimType: mnn.DimensionType.MNN_CAFFE,
  );
  output.copyToHost(outputUser);

  final result = <List<(int, double)>>[];
  final type = outputUser.type;
  for (var batch = 0; batch < B; batch++) {
    final size = outputUser.getStride(0);
    final tempValues = <(int, double)>[];
    if (type.code == mnn.HalideTypeCode.halide_type_float) {
      final values = outputUser.cast<mnn.float32>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i]));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_uint &&
        type.bytes == 1) {
      final values = outputUser.cast<mnn.uint8>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i].toDouble()));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_int &&
        type.bytes == 1) {
      final values = outputUser.cast<mnn.int8>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i].toDouble()));
      }
    } else {
      throw Exception("Unsupported type: $type");
    }
    final sorted = tempValues..sort((a, b) => b.$2.compareTo(a.$2));
    result.add(sorted.take(topK).toList());
  }
  return result;
}
