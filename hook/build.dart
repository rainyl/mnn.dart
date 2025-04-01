// Copyright (c) 2025, rainyl. All rights reserved. Use of this source code is governed by a
// Apache-2.0 license that can be found in the LICENSE file.

// ignore_for_file: avoid_print

import 'package:logging/logging.dart';
import 'package:native_assets_cli/native_assets_cli.dart';
import 'package:native_toolchain_cmake/native_toolchain_cmake.dart';

/// Implements the protocol from `package:native_assets_cli` by building
/// the C code in `src/` and reporting what native assets it built.
void main(List<String> args) async {
  await build(args, _builder);
}

Future<void> _builder(BuildInput input, BuildOutputBuilder output) async {
  final packageName = input.packageName;
  final packagePath = await getPackagePath(packageName);
  final cbuilder = CMakeBuilder.create(
    name: packageName,
    sourceDir: Uri.directory(packagePath).resolve('src/'),
    targets: [
      // '',
    ],
    defines: {
      'CMAKE_INSTALL_PREFIX':
          input.outputDirectory.resolve('install').toFilePath(),
    },
  );

  await cbuilder.run(
    input: input,
    output: output,
    logger: Logger("")
      ..level = Level.ALL
      ..onRecord.listen((record) => print(record.message)),
  );

  await output.findAndAddCodeAssets(input, names: {"mnn_c_api": "mnn.dart"});
}
