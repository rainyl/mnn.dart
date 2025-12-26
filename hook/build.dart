// Copyright (c) 2025, rainyl. All rights reserved. Use of this source code is governed by a
// Apache-2.0 license that can be found in the LICENSE file.

// ignore_for_file: avoid_print, dead_code

// ignore: unused_import
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

void main(List<String> args) async {
  await build(args, _builder);
}

Future<void> _builder(BuildInput input, BuildOutputBuilder output) async {
  final packageName = input.packageName;
  final packagePath = Uri.directory(await getPackagePath(packageName));
  final sourceDir = packagePath.resolve('src/');
  // final outDir = Uri.directory(packagePath).resolve('build/');
  final userDefines = input.userDefines;
  final defsOptions = userDefines["options"] as Map<String, dynamic>?;

  final debugMode = defsOptions?['debug'] as bool? ?? false;
  final buildLocal = defsOptions?['build_local'] as bool? ?? false;

  final logger = Logger("")
    ..level = Level.ALL
    ..onRecord.listen((record) => debugMode ? stderr.writeln(record.message) : print(record.message));

  final defsDefines = userDefines["defines"] as Map<String, dynamic>?;
  final defsCommon = (defsDefines?['common'] as Map<String, dynamic>? ?? {}).cast<String, String>();
  final defsPlatform = switch (input.config.code.targetOS) {
    OS.android => (defsDefines?["android"] as Map<String, String>? ?? {}).cast<String, String>(),
    OS.iOS => (defsDefines?["ios"] as Map<String, String>? ?? {}).cast<String, String>(),
    OS.linux => (defsDefines?["linux"] as Map<String, String>? ?? {}).cast<String, String>(),
    OS.macOS => (defsDefines?["macos"] as Map<String, String>? ?? {}).cast<String, String>(),
    OS.windows => (defsDefines?["windows"] as Map<String, String>? ?? {}).cast<String, String>(),
    _ => <String, String>{},
  };
  final defines = {...defsCommon, ...defsPlatform};

  logger.info("defines: $defines");

  final builder = CMakeBuilder.create(
    name: packageName,
    sourceDir: sourceDir,
    // outDir: outDir,
    generator: Generator.ninja,
    buildLocal: buildLocal,
    targets: ['install'],
    defines: {
      'CMAKE_INSTALL_PREFIX': input.outputDirectory.resolve('install').toFilePath(),
      'MNN_BUILD_BENCHMARK': 'OFF',
      'MNN_BUILD_TEST': 'OFF',
      'MNN_BUILD_TOOLS': 'OFF',
      'MNN_SEP_BUILD': 'OFF',
      'MNN_BUILD_SHARED_LIBS': 'OFF',
      'MNN_BUILD_TRAIN': 'OFF',
      'MNN_BUILD_DEMO': 'OFF',
      'MNN_BUILD_QUANTOOLS': 'OFF',
      'MNN_EVALUATION': 'OFF',
      'MNN_BUILD_CONVERTER': 'OFF',
      'MNN_SUPPORT_DEPRECATED_OP': 'OFF',
      'MNN_AAPL_FMWK': 'OFF',
      'MNN_BUILD_CODEGEN': 'OFF',
      'MNN_ENABLE_COVERAGE': 'OFF',
      'MNN_JNI': 'OFF',
      "MNN_KLEIDIAI": 'OFF',
      ...defines,
    },
  );

  await builder.run(input: input, output: output, logger: logger);

  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install'),
    names: {"mnn_c_api": "mnn.dart"},
    logger: logger,
  );
}
