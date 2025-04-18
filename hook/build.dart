// Copyright (c) 2025, rainyl. All rights reserved. Use of this source code is governed by a
// Apache-2.0 license that can be found in the LICENSE file.

// ignore_for_file: avoid_print

// ignore: unused_import
import 'dart:io';

import 'package:logging/logging.dart';
import 'package:mnn/src/hook_helpers/parse_user_define.dart';
import 'package:native_assets_cli/native_assets_cli.dart';
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

  logger.info("default defines: $defsDefault");
  logger.info("user defines: $defsUser");
  logger.info("final defines: $defsFinal");

  final builder = CMakeBuilder.create(
    name: packageName,
    sourceDir: sourceDir,
    // outDir: outDir,
    generator: Generator.ninja,
    targets: ['install'],
    defines: {
      'CMAKE_INSTALL_PREFIX': input.outputDirectory.resolve('install').toFilePath(),
      ...defsFinal,
    },
    buildLocal: true,
  );

  await builder.run(input: input, output: output, logger: logger);

  await output.findAndAddCodeAssets(
    input,
    outDir: input.outputDirectory.resolve('install'),
    names: {"mnn_c_api": "mnn.dart"},
  );
}
