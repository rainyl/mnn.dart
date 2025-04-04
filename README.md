# MNN.dart

A Dart wapper for [MNN](https://github.com/alibaba/MNN).

## Features

- Inference MNN models in Dart.
- Native-Assets support.

## Getting started

```
dart --enable-experiment=native-assets run example/image_recognition.dart
```

## Usage

```dart
import 'package:mnn/mnn.dart' as mnn;

void main() {
  final net = mnn.Interpreter.fromFile("example.mnn");
  final session = net.createSession();
  final input = session.getInput();
  final output = session.getOutput();
  // fill input
  session.run();
  // process output
  final outputUser = mnn.Tensor.fromTensor(output, dimType: dimType);
  output.copyToHost(outputUser);
  outputUser.printShape();
}
```

## Authors

- [rainyl](https://github.com/rainyl)

## License

[Apache-2.0](LICENSE)
