// ignore_for_file: avoid_print

import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'package:mnn/mnn.dart' as mnn;

void main(List<String> args) async {
  if (args.length < 2) {
    print('Usage: dart mnn_image_recognition.dart model.mnn input0.jpg input1.jpg ...\n'
        'Please download and convert model first, e.g., mobilenetv2 from \n'
        'https://github.com/onnx/models/tree/main/validated/vision/classification/mobilenet/model\n'
        'convert command:\n'
        '\tMNNConvert -f ONNX --fp16 --modelFile mobilenetv2-7.onnx --MNNModel mobilenetv2-7.mnn --bizCode mobilenetv2-7');
    return;
  }

  const topK = 3;

  final net = mnn.Interpreter.fromFile(args[0]);
  net.setCacheFile('.cachefile');
  net.setSessionMode(mnn.SessionMode.Session_Backend_Auto);
  net.setSessionHint(mnn.HintMode.MAX_TUNING_NUMBER, 5);

  final config = mnn.ScheduleConfig.create(type: mnn.ForwardType.MNN_FORWARD_AUTO);
  final session = net.createSession(config: config);

  final input = net.getSessionInput(session);
  if (input == null || input.isEmpty) {
    print("Can't get input tensor");
    return;
  }
  final shape = input.shape;
  assert(shape.length == 4);
  if (shape[1] == -1) shape[1] = 3;
  if (shape[2] == -1) shape[2] = 224;
  if (shape[3] == -1) shape[3] = 224;
  shape[0] = args.length - 1;
  net.resizeTensor(input, shape);
  net.resizeSession(session);
  final output = net.getSessionOutput(session);
  final (B, C, H, W) = (shape[0], shape[1], shape[2], shape[3]);

  if (output == null || output.isEmpty || output.elementSize == 0) {
    throw Exception("Resize error, the model can't run batch: $B");
  }

  final memoryUsage = session.memoryInfo;
  final flops = session.flopsInfo;
  final backendType = session.backendsInfo;
  print('Session Info:'
      '  memory use ${memoryUsage.toStringAsFixed(2)} MB,'
      '  flops is ${flops.toStringAsFixed(2)} M,'
      '  backendType is $backendType,'
      '  batch size = ${args.length - 1}');

  final nchwTensor = mnn.Tensor.fromTensor(input, dimType: mnn.DimensionType.MNN_CAFFE);
  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  for (var i = 1; i < args.length; i++) {
    final imagePath = args[i];
    final imageFile = File(imagePath);
    if (!imageFile.existsSync()) {
      print("Can't open $imagePath");
      continue;
    }
    final im = mnn.Image.fromMemory(await imageFile.readAsBytes());
    if (im.isEmpty) {
      throw Exception("Can't decode $imagePath");
    }
    print("image: ${im.values.sublist(0, 7)}");
    final im1 = im.normalize(scale: 255.0, mean: mean, std: std).toPlanar();
    print("image: ${im1.values.sublist(0, 7)}");

    nchwTensor.setImage(i - 1, im1);
    print(nchwTensor.cast<mnn.float32>().asTypedList(7));
  }
  input.copyFromHost(nchwTensor);
  nchwTensor.dispose();

  net.runSession(session);
  final dimType = output.type.code == mnn.HalideTypeCode.halide_type_float
      ? output.dimensionType
      : mnn.DimensionType.MNN_CAFFE;
  final outputUser = mnn.Tensor.fromTensor(output, dimType: dimType);
  output.copyToHost(outputUser);
  outputUser.printShape();
  final type = outputUser.type;
  for (var batch = 0; batch < B; batch++) {
    print('For Image: ${args[batch + 1]}');
    final size = outputUser.getStride(0);
    final tempValues = <(int, double)>[];
    if (type.code == mnn.HalideTypeCode.halide_type_float) {
      final values = outputUser.host.cast<mnn.float32>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i]));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_uint && type.bytes == 1) {
      final values = outputUser.host.cast<mnn.uint8>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i].toDouble()));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_int && type.bytes == 1) {
      final values = outputUser.host.cast<mnn.int8>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i].toDouble()));
      }
    } else {
      throw Exception("Unsupported type: $type");
    }
    final sorted = tempValues..sort((a, b) => b.$2.compareTo(a.$2));

    final topN = min(size, topK);
    for (var j = 0; j < topN; j++) {
      print('${sorted[j].$1}, ${sorted[j].$2}');
    }
  }

  net.updateCacheFile(session);
}
