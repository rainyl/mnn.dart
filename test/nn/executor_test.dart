import 'package:mnn/mnn.dart' as mnn;
import 'package:mnn/nn.dart' as nn;
import 'package:test/test.dart';

void main() {
  group('RuntimeManager Tests', () {
    late nn.RuntimeManager manager;

    setUp(() {
      manager = nn.RuntimeManager.create();
    });

    tearDown(() {
      manager.dispose();
    });

    test('RuntimeManager creation', () {
      expect(manager, isNotNull);
      expect(manager.isEmpty, false);
    });

    test('RuntimeManager creation with ScheduleConfig', () {
      final config = mnn.ScheduleConfig.create(type: mnn.ForwardType.MNN_FORWARD_CPU, numThread: 2);
      final mgr = nn.RuntimeManager.create(config);
      expect(mgr, isNotNull);
      expect(mgr.isEmpty, false);
      mgr.dispose();
      config.dispose();
    });

    test('RuntimeManager setCache and updateCache', () {
      manager.setCache("test_cache_path");
      manager.updateCache();
    });

    test('RuntimeManager setExternalPath and setExternalFile', () {
      manager.setExternalPath("external_path", nn.ExternalPathType.EXTERNAL_WEIGHT_DIR);
      manager.setExternalFile("external_file");
    });

    test('RuntimeManager isBackendSupport', () {
      // CPU should be supported
      final supported = manager.isBackendSupport(mnn.ForwardType.MNN_FORWARD_CPU);
      expect(supported, isTrue);
      expect(manager.isBackendSupport(mnn.ForwardType.MNN_FORWARD_OPENCL), isA<bool>());
    });

    test('RuntimeManager setMode and setHint', () {
      manager.setMode(mnn.SessionMode.Session_Backend_Auto);
      manager.setHint(mnn.HintMode.MAX_TUNING_NUMBER, 0);
    });

    test('RuntimeManager getDeviceInfo', () {
      final (success, info) = nn.RuntimeManager.getDeviceInfo("dsp_arch", mnn.ForwardType.MNN_FORWARD_CPU);
      expect(success, isA<bool>());
      expect(info, isA<String>());
      final (success2, info2) = nn.RuntimeManager.getDeviceInfo("soc_id", mnn.ForwardType.MNN_FORWARD_CPU);
      expect(success2, isA<bool>());
      expect(info2, isA<String>());
    });
  });

  group('ModuleInfo Tests', () {
    late nn.Module module;
    late nn.ModuleInfo info;

    setUp(() {
      module = nn.Module.loadFromFile("test/data/mnist-8.mnn");
      info = module.info;
    });

    tearDown(() {
      info.dispose();
      module.dispose();
    });

    test('ModuleInfo inputsLength', () {
      // MNIST model usually has 1 input
      expect(info.inputsLength, greaterThanOrEqualTo(1));
    });

    test('ModuleInfo getInputNames and getOutputNames', () {
      final inputNames = info.getInputNames();
      expect(inputNames, ["Input3"]);

      final outputNames = info.getOutputNames();
      expect(outputNames, ["Plus214_Output_0"]);
    });

    test('ModuleInfo defaultFormat', () {
      final format = info.defaultFormat();
      expect(format, isA<mnn.DimensionFormat>());
    });

    test('ModuleInfo version and bizCode', () {
      expect(info.version, isA<String>());
      expect(info.version, isNotEmpty);
      expect(info.bizCode, isA<String>());
      expect(info.bizCode, isNotEmpty);
    });

    test('ModuleInfo metadata', () {
      final metadata = info.metadata;
      expect(metadata, isA<Map<String, String>>());
    });

    test('ModuleInfo getInput', () {
      if (info.inputsLength > 0) {
        final variableInfo = info.getInput(0);
        expect(variableInfo, isNotNull);
        variableInfo.dispose();
      }
    });
  });

  test('ExecutorScope.current', () {
    final executor = nn.ExecutorScope.current();
    expect(executor.isEmpty, false);
  });

  test('Executor', () {
    final executor = nn.Executor.create(mnn.ForwardType.MNN_FORWARD_CPU, mnn.BackendConfig.create(), 4);
    expect(executor.isEmpty, false);

    executor.lazyMode = nn.LazyMode.LAZY_CONTENT;
    expect(executor.lazyMode, nn.LazyMode.LAZY_CONTENT);
    executor.lazyEval = true;
    expect(executor.lazyEval, true);

    executor.gc(nn.GCFlag.PART);

    final info = nn.Executor.getRuntimeInfo();
    expect(info.isEmpty, false);

    final status = executor.getCurrentRuntimeStatus(nn.RuntimeStatus.STATUS_COUNT);
    expect(status, isA<int>());

    executor.dispose();
  });

  test('Executor.global', () {
    final executor = nn.Executor.global;
    expect(executor.isEmpty, false);
    executor.dispose();
  });
}
