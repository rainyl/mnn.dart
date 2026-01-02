import 'dart:io';
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/nn.dart' as nn;
import 'package:mnn/numpy.dart' as np;
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

Future<void> testInferenceMnist(
  nn.Module model, {
  String imgPath = "test/data/mnist_0.png",
  int target = 0,
  bool runAsync = false,
}) async {
  final im = await img.decodePngFile(imgPath);
  expect(im, isNotNull);
  final gray = img.grayscale(im!);
  final resized = img.copyResize(gray, width: 28, height: 28);
  expect(resized.length, 28 * 28);
  final pixData = Float32List.fromList(resized.map((e) => e.first / 255.0).toList());

  final varp = mnn.VARP.fromListND<mnn.float32>(pixData, [1, 1, 28, 28], format: mnn.DimensionFormat.NCHW);
  expect(varp.shape, [1, 1, 28, 28]);
  expect(varp.isEmpty, false);

  final output = runAsync ? await model.forwardAsync(varp) : model.forward(varp);
  expect(output.shape, [1, 10]);
  expect(output.ndim, 2);

  final probs = np.argmax(output);
  expect(probs.ndim, 0);
  expect(probs.value, isA<int>());
  expect(probs.value, target);
}

void main() async {
  group('NN API Tests', () {
    test('RuntimeManager creation and destruction', () {
      final mgr = nn.RuntimeManager.create();
      expect(mgr.isEmpty, false);

      mgr.setCache("test_cache");
      mgr.updateCache();

      mgr.dispose();
    });

    test('Module load from file', () {
      final model = nn.Module.loadFromFile("test/data/mnist-8.mnn");
      expect(model.isEmpty, false);
      model.dispose();
    });

    test('Module load from buffer', () {
      final modelFile = File("test/data/mnist-8.mnn");
      final modelBuffer = modelFile.readAsBytesSync();
      // will use global Executor
      final model = nn.Module.loadFromBuffer(modelBuffer);
      expect(model.isEmpty, false);
      model.dispose();
    });

    test('Module load with RuntimeManager', () async {
      final schedule = mnn.ScheduleConfig.create();
      final mgr = nn.RuntimeManager.create(schedule);
      final model = nn.Module.loadFromFile("test/data/mnist-8.mnn", runtimeManager: mgr);
      expect(model.isEmpty, false);

      for (final (imgPath, target) in imagePaths) {
        await testInferenceMnist(model, imgPath: imgPath, target: target);
        await testInferenceMnist(model, imgPath: imgPath, target: target, runAsync: true);
      }
      model.dispose();
    });

    test(
      'Module with ExecutorScope',
      skip: "ExecutorScope is not supported because of the resource management",
      () async {
        final executor = nn.Executor.create(mnn.ForwardType.MNN_FORWARD_CPU, mnn.BackendConfig.create(), 4);
        final scope = nn.ExecutorScope.create(executor);
        final model = nn.Module.loadFromFile("test/data/mnist-8.mnn");
        expect(model.isEmpty, false);

        for (final (imgPath, target) in imagePaths) {
          await testInferenceMnist(model, imgPath: imgPath, target: target);
          // await testInferenceMnist(model, imgPath: imgPath, target: target, runAsync: true);
        }
        model.dispose();
        scope.dispose();
      },
    );
  });
}
