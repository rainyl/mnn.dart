// ignore_for_file: avoid_print

import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'package:image/image.dart' as img;
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
  print(input);
  final shape = input.shape;
  assert(shape.length == 4);
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

  const mean = [0.485, 0.456, 0.406];
  const std = [0.229, 0.224, 0.225];
  final pixData = <double>[]; // RRRGGGBBB...
  for (var i = 1; i < args.length; i++) {
    final imagePath = args[i];
    final imageFile = File(imagePath);
    if (!imageFile.existsSync()) {
      print("Can't open $imagePath");
      continue;
    }
    final im = await (img.Command()
          ..decodeImage(imageFile.readAsBytesSync())
          ..copyResize(height: H, width: W, interpolation: img.Interpolation.linear))
        .executeThread();
    assert(im.outputImage != null);
    assert(im.outputImage!.length == input.count);

    final pixR = <double>[];
    final pixG = <double>[];
    final pixB = <double>[];
    for (final pix in im.outputImage!) {
      // print("${pix.y}, ${pix.x}, ${pix.r}, ${pix.g}, ${pix.b}");
      // final _pix = [
      //   (pix.r / 255.0 - mean[0]) / std[0],
      //   (pix.g / 255.0 - mean[1]) / std[1],
      //   (pix.b / 255.0 - mean[2]) / std[2],
      // ];
      // pixData.addAll(_pix);
      pixR.add((pix.r / 255.0 - mean[0]) / std[0]);
      pixG.add((pix.g / 255.0 - mean[1]) / std[1]);
      pixB.add((pix.b / 255.0 - mean[2]) / std[2]);
    }
    pixData.addAll(pixR);
    pixData.addAll(pixG);
    pixData.addAll(pixB);
  }
  print(pixData.sublist(0, 6));
  final host = input.map(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType);
  assert(host.address != 0);
  host.cast<mnn.f32>().asTypedList(pixData.length).setAll(0, pixData);
  input.unmap(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType, host);

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
      final values = outputUser.host.cast<mnn.f32>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i]));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_uint && type.bytes == 1) {
      final values = outputUser.host.cast<mnn.u8>() + batch * size;
      for (var i = 0; i < size; i++) {
        tempValues.add((i, values[i].toDouble()));
      }
    } else if (type.code == mnn.HalideTypeCode.halide_type_int && type.bytes == 1) {
      final values = outputUser.host.cast<mnn.i8>() + batch * size;
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
