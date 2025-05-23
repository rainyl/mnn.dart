// Copyright (c) 2025, rainyl. All rights reserved. Use of this source code is governed by a
// Apache-2.0 license that can be found in the LICENSE file.

// ignore_for_file: avoid_print, dead_code

// ignore: unused_import
import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:logging/logging.dart';
import 'package:mnn/src/hook_helpers/parse_user_define.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

/// Implements the protocol from `package:native_assets_cli` by building
/// the C code in `src/` and reporting what native assets it built.
void main(List<String> args) async {
  await build(args, _builder);
}

Future<void> _builder(BuildInput input, BuildOutputBuilder output) async {
  final packageName = input.packageName;
  final packagePath = Uri.directory(await getPackagePath(packageName));
  final sourceDir = packagePath.resolve('src/');
  // final outDir = Uri.directory(packagePath).resolve('build/');
  final logger = Logger("")
    ..level = Level.ALL
    ..onRecord.listen((record) => stderr.writeln(record.message));
  // ..onRecord.listen((record) => print(record.message));

  final defsDefault = parseUserDefinedOptions(packagePath.resolve("pubspec.yaml").toFilePath());
  final defsUser = parseUserDefinedOptions(
    Platform.script.resolve('../../../../pubspec.yaml').toFilePath(),
  );
  final defsFinal = {...defsDefault, ...defsUser};
  final defines = (defsFinal["defines"]! as Map<String, Object>)["common"]! as Map<String, String>;
  final definesPlatform = switch (input.config.code.targetOS) {
    OS.android => ((defsFinal["defines"]! as Map<String, Object>)["android"]! as Map<String, String>),
    OS.iOS => ((defsFinal["defines"]! as Map<String, Object>)["ios"]! as Map<String, String>),
    OS.linux => ((defsFinal["defines"]! as Map<String, Object>)["linux"]! as Map<String, String>),
    OS.macOS => ((defsFinal["defines"]! as Map<String, Object>)["macos"]! as Map<String, String>),
    OS.windows => ((defsFinal["defines"]! as Map<String, Object>)["windows"]! as Map<String, String>),
    _ => <String, String>{},
  };
  defines.addAll(definesPlatform);
  final options = (defsFinal["options"] as Map<String, Object>?) ?? {};

  logger.info("defines: $defines");
  logger.info("options: $options");

  final builder = CMakeBuilder.create(
    name: packageName,
    sourceDir: sourceDir,
    // outDir: outDir,
    generator: Generator.ninja,
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
      ...defines,
    },
    buildLocal: options["build_local"] as bool? ?? false,
  );

  await builder.run(input: input, output: output, logger: logger);

  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install'),
    names: {"mnn_c_api": "mnn.dart"},
  );
}
