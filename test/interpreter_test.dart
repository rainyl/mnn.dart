import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

const modelPath = "test/data/mnist-8.mnn";
const imagePaths = [
  ("test/data/mnist_0.png", 0),
  ("test/data/mnist_1.png", 1),
  ("test/data/mnist_2.png", 2),
  ("test/data/mnist_4.png", 4),
  ("test/data/mnist_5.png", 5),
  ("test/data/mnist_8.png", 8),
  ("test/data/mnist_9.png", 9),
  ("test/data/mnist_9_1.png", 9),
];

Future<void> testSessionMnist(
  mnn.Session session, {
  String imgPath = "test/data/mnist_0.png",
  int target = 0,
}) async {
  expect(session.isEmpty, false);
  {
    final input = session.getInput();
    expect(input, isNotNull);
    expect(input?.shape, [1, 1, 28, 28]);
    expect(input?.type, mnn.HalideType.f32());
    final output = session.getOutput();
    expect(output, isNotNull);
    expect(output?.shape, [1, 10]);
    expect(output?.type, mnn.HalideType.f32());
  }
  {
    final input = session.getInput(name: "Input3");
    expect(input, isNotNull);
    expect(input?.shape, [1, 1, 28, 28]);
    expect(input?.type, mnn.HalideType.f32());
    final output = session.getOutput(name: "Plus214_Output_0");
    expect(output, isNotNull);
    expect(output?.shape, [1, 10]);
    expect(output?.type, mnn.HalideType.f32());
  }
  final inputs = session.getInputAll();
  expect(inputs.containsKey("Input3"), true);
  expect(inputs.length, 1);
  final outputs = session.getOutputAll();
  expect(outputs.containsKey("Plus214_Output_0"), true);
  expect(outputs.length, 1);

  // run session
  final input = session.getInput();
  expect(input, isNotNull);
  final im = await img.decodePngFile(imgPath);
  expect(im, isNotNull);
  final gray = img.grayscale(im!);
  final resized = img.copyResize(gray, width: 28, height: 28);
  expect(resized.length, 28 * 28);
  final pixData = Float32List.fromList(resized.map((e) => e.first / 255.0).toList());
  expect(pixData.length, input!.count);

  // copy
  {
    final inputTensor = mnn.Tensor.fromData(
      input.shape,
      mnn.HalideType.f32(),
      data: pixData.buffer.asUint8List(),
      dimType: mnn.DimensionType.MNN_CAFFE,
    );
    input.copyFromHost(inputTensor);
    inputTensor.dispose();

    final code = session.run();
    expect(code, mnn.ErrorCode.NO_ERROR);

    final output = session.getOutput();
    expect(output, isNotNull);
    final outputTensor = mnn.Tensor.fromTensor(output!, dimType: mnn.DimensionType.MNN_CAFFE);
    output.copyToHost(outputTensor);
    expect(outputTensor.shape, [1, 10]);
    expect(outputTensor.type, mnn.HalideType.f32());
    final logits = outputTensor.host.cast<mnn.f32>().asTypedList(10);
    expect(logits.indexOf(logits.reduce(max)), target);
  }

  // mapping
  {
    final host = input.map(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType);
    expect(host.address, isNonZero);
    host.cast<mnn.f32>().asTypedList(input.count).setAll(0, pixData);
    input.unmap(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType, host);

    final code = session.run();
    expect(code, mnn.ErrorCode.NO_ERROR);

    final output = session.getOutput();
    final hostOut = output!.map(mnn.MapType.MNN_MAP_TENSOR_READ, output.dimensionType);
    expect(hostOut.address, isNonZero);
    final logits = hostOut.cast<mnn.f32>().asTypedList(output.count);
    expect(logits.indexOf(logits.reduce(max)), target);
    output.unmap(mnn.MapType.MNN_MAP_TENSOR_READ, output.dimensionType, hostOut);
  }
}

void main() async {
  group('Interpreter', () {
    test('create from file', () async {
      final model = mnn.Interpreter.fromFile("test/data/mnist-8.mnn");
      expect(model.bizCode, "mnist");
      expect(model.uuid, "8282614c-2d46-42ac-9761-3edc2051a907");
      expect(model.modelVersion, "3.1.2");
    });

    test('create from buffer', () async {
      final buffer = await File(modelPath).readAsBytes();
      final model = mnn.Interpreter.fromBuffer(buffer);
      expect(model.bizCode, "mnist");
      expect(model.uuid, "8282614c-2d46-42ac-9761-3edc2051a907");
    });

    test('session', () async {
      final model = mnn.Interpreter.fromFile(modelPath);
      expect(model.isEmpty, false);
      final session = model.createSession();
      await testSessionMnist(session);
    });

    test('schedule', () async {
      final buffer = await File(modelPath).readAsBytes();
      final model = mnn.Interpreter.fromBuffer(buffer);
      expect(model.isEmpty, false);

      final config = mnn.ScheduleConfig.create();
      config.mode = mnn.MNN_GPU_TUNING_NORMAL | mnn.MNN_GPU_MEMORY_IMAGE;
      config.backendConfig = mnn.BackendConfig.create();

      final session = model.createSession(config: config);
      for (final (imgPath, target) in imagePaths) {
        print(imgPath); // ignore: avoid_print
        await testSessionMnist(session, imgPath: imgPath, target: target);
      }
    });

    test('runtime', () async {
      final config = mnn.ScheduleConfig.create();
      config.mode = mnn.MNN_GPU_TUNING_NORMAL | mnn.MNN_GPU_MEMORY_IMAGE;
      final runtime = mnn.Interpreter.createRuntime([config]);
      expect(runtime.isEmpty, false);

      final net1 = mnn.Interpreter.fromFile(modelPath);
      final session1 = net1.createSessionWithRuntime(config, runtime);

      final net2 = mnn.Interpreter.fromFile(modelPath);
      final session2 = net2.createSessionWithRuntime(config, runtime);

      await testSessionMnist(session1);
      await testSessionMnist(session2);
    });
  });
}
