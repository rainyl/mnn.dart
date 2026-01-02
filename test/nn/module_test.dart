import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/nn.dart' as nn;
import 'package:test/test.dart';

void main() {
  group('ModuleConfig Tests', () {
    test('ModuleConfig creation and properties', () {
      final config = nn.ModuleConfig.create(
        dynamic: true,
        shapeMutable: true,
        rearrange: true,
        backendType: mnn.ForwardType.MNN_FORWARD_OPENCL,
      );

      expect(config.dynamic, true);
      expect(config.shapeMutable, true);
      expect(config.rearrange, true);
      expect(config.backendType, mnn.ForwardType.MNN_FORWARD_OPENCL);

      // Test setters
      config.dynamic = false;
      expect(config.dynamic, false);

      config.shapeMutable = false;
      expect(config.shapeMutable, false);

      config.rearrange = false;
      expect(config.rearrange, false);

      config.backendType = mnn.ForwardType.MNN_FORWARD_CPU;
      expect(config.backendType, mnn.ForwardType.MNN_FORWARD_CPU);

      config.dispose();
    });

    test('ModuleConfig with BackendConfig', () {
      final backendConfig = mnn.BackendConfig.create(
        memory: mnn.MNN_MEMORY_HIGH,
        power: mnn.MNN_POWER_HIGH,
        precision: mnn.MNN_PRECISION_HIGH,
      );

      final config = nn.ModuleConfig.create(backendConfig: backendConfig);

      expect(config.backendConfig, isNotNull);
      expect(config.backendConfig!.memory, mnn.MNN_MEMORY_HIGH);

      // Test setter
      final newBackendConfig = mnn.BackendConfig.create(memory: mnn.MNN_MEMORY_LOW);
      config.backendConfig = newBackendConfig;
      expect(config.backendConfig!.memory, mnn.MNN_MEMORY_LOW);

      // Test null setter
      config.backendConfig = null;
      expect(config.backendConfig, isNull);

      newBackendConfig.dispose();
      backendConfig.dispose();
      config.dispose();
    });
  });

  group('Module Tests', () {
    late nn.Module module;

    setUp(() {
      module = nn.Module.loadFromFile("test/data/mnist-8.mnn");
    });

    tearDown(() {
      module.dispose();
    });

    test('Module properties getters and setters', () {
      expect(module.isTraining, isA<bool>());
      module.isTraining = true;
      expect(module.isTraining, true);

      module.isTraining = false;
      expect(module.isTraining, false);

      // Name and Type might be empty by default or specific strings depending on model
      // We can test setting them
      module.name = "TestModule";
      expect(module.name, "TestModule");

      module.type = "TestType";
      expect(module.type, "TestType");
    });

    test('Module clearCache', () {
      module.clearCache();
    });

    test('Module clone', () {
      final cloned = module.clone(shareParams: true);
      expect(cloned, isNotNull);
      // Verify cloned module works or has properties
      cloned.isTraining = true;
      expect(cloned.isTraining, true);
      // Original should remain unchanged if they don't share training state (though they might share params)
      // Actually isTraining is usually a per-module flag.
      module.isTraining = false;
      expect(module.isTraining, false);
      expect(cloned.isTraining, true);

      cloned.dispose();
    });

    test('Module info', () {
      final info = module.info;
      expect(info, isNotNull);
      // Check some basic info properties to ensure it's valid
      // Mnist model usually has 1 input
      expect(info.inputsLength, greaterThanOrEqualTo(0));
    });

    test('Module addParameter and setParameter', () {
      // Create a dummy parameter
      final param = mnn.VARP.scalar<ffi.Float>(1.0);
      param.setName("TestParam");
      final index = module.addParameter(param);
      expect(index, greaterThanOrEqualTo(0));

      final param2 = mnn.VARP.scalar<ffi.Float>(2.0);
      param2.setName("TestParam2");
      module.setParameter(param2, index);

      param.dispose();
      param2.dispose();
    });

    test('Module onForward', () {
      nn.usingExecutor((executor) {
        // Mnist input 1x1x28x28
        final inputData = Float32List(1 * 1 * 28 * 28);
        final input = mnn.VARP.fromListND<ffi.Float>(inputData, [
          1,
          1,
          28,
          28,
        ], format: mnn.DimensionFormat.NCHW);
        final inputs = mnn.VecVARP.of([input]);

        final outputs = module.onForward(inputs);
        expect(outputs.length, greaterThan(0));

        // Cleanup
        input.dispose();
        inputs.dispose();
        outputs.dispose();
      });
    });

    test('Module onForwardAsync', () async {
      await nn.usingExecutor((e) async {
        // Mnist input 1x1x28x28
        final inputData = Float32List(1 * 1 * 28 * 28);
        final input = mnn.VARP.fromListND<ffi.Float>(inputData, [
          1,
          1,
          28,
          28,
        ], format: mnn.DimensionFormat.NCHW);
        final inputs = mnn.VecVARP.create();
        inputs.push_back(input);

        final outputs = await module.onForwardAsync(inputs);
        expect(outputs.length, greaterThan(0));

        // Cleanup
        input.dispose();
        inputs.dispose();
        outputs.dispose();
      });
    });
  });
}
