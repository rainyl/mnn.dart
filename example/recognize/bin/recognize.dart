import 'package:args/args.dart';
import 'package:mnn/mnn.dart' as mnn;

import 'inference.dart';

const String version = '0.0.1';

ArgParser buildParser() {
  return ArgParser()
    ..addFlag(
      'help',
      abbr: 'h',
      negatable: false,
      help: 'Print this usage information.',
    )
    ..addFlag(
      'verbose',
      abbr: 'v',
      negatable: false,
      help: 'Show additional command output.',
    )
    ..addFlag('version', negatable: false, help: 'Print the tool version.')
    ..addOption("model", abbr: 'm', help: "model path", mandatory: true)
    ..addMultiOption("image", abbr: "i", help: "image paths")
    ..addOption(
      "topk",
      abbr: "k",
      help: "topk",
      defaultsTo: "3",
      callback: (s) => int.parse(s!),
    );
}

void printUsage(ArgParser argParser) {
  print('Usage: dart recognize.dart <flags> [arguments]');
  print(argParser.usage);
}

void main(List<String> arguments) {
  final ArgParser argParser = buildParser();
  try {
    final ArgResults results = argParser.parse(arguments);
    bool verbose = false;

    // Process the parsed arguments.
    if (results.flag('help')) {
      printUsage(argParser);
      return;
    }
    if (results.flag('version')) {
      print('recognize version: $version');
      return;
    }
    if (results.flag('verbose')) {
      verbose = true;
    }

    // Act on the arguments provided.
    print('Positional arguments: ${results.rest}');
    if (verbose) {
      print('[VERBOSE] All arguments: ${results.arguments}');
    }

    final modelPath = results.option('model');
    final imagePaths = results.multiOption("image");
    final topk = int.parse(results.option('topk')!);

    final model = loadModel(modelPath!);
    final config = mnn.ScheduleConfig.create(
      type: mnn.ForwardType.MNN_FORWARD_METAL,
    );
    final session = model.createSession(config: config);
    final result = inference(session, imagePaths, topK: topk);
    print(result);
  } on FormatException catch (e) {
    // Print usage information if an invalid argument was provided.
    print(e.message);
    print('');
    printUsage(argParser);
  }
}
