import 'dart:ffi';
import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;
import 'package:mnn/mnn.dart' as mnn;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'mnn.dart image recognition demo'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _selectedImage;
  List<(String, double)> _recognitionResult = [];
  mnn.Interpreter? _net;
  mnn.Session? _session;
  List<String> _classLabels = [];
  final mean = [0.485, 0.456, 0.406];
  final std = [0.229, 0.224, 0.225];

  @override
  void dispose() {
    _net?.release();
    super.dispose();
  }

  List<double> _softmax(List<(int, double)> values) {
    final expValues = values.map((e) => exp(e.$2)).toList();
    final sumExp = expValues.reduce((a, b) => a + b);
    return expValues.map((e) => e / sumExp).toList();
  }

  @override
  void initState() {
    super.initState();
    _loadModel();
    _loadClassLabels();
  }

  Future<void> _loadClassLabels() async {
    try {
      final data = await rootBundle.loadString('assets/class_desc.txt');
      setState(() {
        _classLabels = data.split('\n');
      });
    } catch (e) {
      debugPrint('Failed to load class labels: $e');
    }
  }

  Future<void> _loadModel() async {
    try {
      final modelPath = 'assets/mobilenetv2-7.mnn';
      final byteData = await rootBundle.load(modelPath);
      final bytes = byteData.buffer.asUint8List();
      _net = mnn.Interpreter.fromBuffer(bytes);
      _net!.setCacheFile('.cachefile');
      _net!.setSessionMode(mnn.SessionMode.Session_Backend_Auto);
      _net!.setSessionHint(mnn.HintMode.MAX_TUNING_NUMBER, 5);

      final config = mnn.ScheduleConfig.create(type: mnn.ForwardType.MNN_FORWARD_AUTO);
      _session = _net!.createSession(config: config);
    } catch (e) {
      debugPrint('Failed to load model: $e');
    }
  }

  Future<void> _pickImage() async {
    final pickedFile = await ImagePicker().pickImage(source: ImageSource.gallery);
    debugPrint('pickedFile: ${pickedFile?.path}');
    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
        _recognitionResult = [];
      });
      _runRecognition();
    }
  }

  Future<void> _runRecognition() async {
    if (_selectedImage == null || _net == null) return;

    try {
      final input = _net!.getSessionInput(_session!);
      if (input == null || input.shape.length < 4) {
        await showDialog(
          context: context,
          builder:
              (context) => AlertDialog(
                title: Text('输入错误'),
                content: Text('模型输入不合法，请检查模型文件'),
                actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: Text('确定'))],
              ),
        );
        return;
      }
      final shape = input.shape;
      final (_, C, H, W) = (shape[0], shape[1], shape[2], shape[3]);

      final im =
          await (img.Command()
                ..decodeImage(_selectedImage!.readAsBytesSync())
                ..copyResize(height: H, width: W, interpolation: img.Interpolation.linear))
              .executeThread();

      final pixData = <double>[];
      final pixR = <double>[];
      final pixG = <double>[];
      final pixB = <double>[];
      for (final pix in im.outputImage!) {
        pixR.add((pix.r / 255.0 - mean[0]) / std[0]);
        pixG.add((pix.g / 255.0 - mean[1]) / std[1]);
        pixB.add((pix.b / 255.0 - mean[2]) / std[2]);
      }
      pixData.addAll(pixR);
      pixData.addAll(pixG);
      pixData.addAll(pixB);

      final host = input.map(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType);
      host.cast<mnn.f32>().asTypedList(pixData.length).setAll(0, pixData);
      input.unmap(mnn.MapType.MNN_MAP_TENSOR_WRITE, input.dimensionType, host);

      _net!.runSession(_session!);
      final output = _net!.getSessionOutput(_session!);
      final outputUser = mnn.Tensor.fromTensor(output, dimType: output.dimensionType);
      output.copyToHost(outputUser);

      final size = outputUser.getStride(0);
      final tempValues = <(int, double)>[];
      final type = outputUser.type;

      if (type.code == mnn.HalideTypeCode.halide_type_float) {
        final values = outputUser.host.cast<mnn.f32>();
        for (var i = 0; i < size; i++) {
          tempValues.add((i, values[i]));
        }
      }

      final probabilities = _softmax(tempValues);
      final sorted =
          tempValues.asMap().map((i, e) => MapEntry(i, (e.$1, probabilities[i]))).values.toList()
            ..sort((a, b) => b.$2.compareTo(a.$2));
      final topN = min(size, 10);
      final results = <(String, double)>[];
      for (var j = 0; j < topN; j++) {
        final classId = sorted[j].$1;
        final className = _classLabels.isNotEmpty && classId < _classLabels.length ? _classLabels[classId] : 'Unknown';
        results.add((className, sorted[j].$2));
      }

      setState(() {
        _recognitionResult = results;
      });
    } catch (e) {
      await showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: Text('识别错误'),
          content: Text('图像识别过程中发生错误: $e'),
          actions: [TextButton(onPressed: () => Navigator.of(context).pop(), child: Text('确定'))],
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    // This method is rerun every time setState is called, for instance as done
    // by the _incrementCounter method above.
    //
    // The Flutter framework has been optimized to make rerunning build methods
    // fast, so that you can just rebuild anything that needs updating rather
    // than having to individually change instances of widgets.
    return Scaffold(
      appBar: AppBar(
        // TRY THIS: Try changing the color here to a specific color (to
        // Colors.amber, perhaps?) and trigger a hot reload to see the AppBar
        // change color while the other colors stay the same.
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        // Here we take the value from the MyHomePage object that was created by
        // the App.build method, and use it to set our appbar title.
        title: Text(widget.title),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.start,
            children: <Widget>[
              if (_selectedImage != null)
                Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Image.file(_selectedImage!, height: 300),
                ),
              if (_recognitionResult.isNotEmpty)
                SizedBox(
                  height: 500,
                  child: ListView.builder(
                    itemCount: _recognitionResult.length,
                    itemBuilder: (context, index) {
                      return Container(
                        decoration: BoxDecoration(
                          color: Colors.grey[200],
                          borderRadius: BorderRadius.circular(8.0),
                        ),
                        margin: const EdgeInsets.symmetric(vertical: 4.0, horizontal: 16.0),
                        child: Column(
                          children: [
                            Padding(
                              padding: const EdgeInsets.all(12.0),
                              child: Text(
                                '${_recognitionResult[index].$1} (${(_recognitionResult[index].$2 * 100).toStringAsFixed(2)}%)',
                                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                                  color: Colors.black87,
                                ),
                              ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
            ],
          ),
        ),
      ),
      floatingActionButton: Column(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          FloatingActionButton(onPressed: _pickImage, tooltip: 'Select Image', child: const Icon(Icons.image)),
        ],
      ),
    );
  }
}
