import 'dart:io';

import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

void main() {
  test('VariableInfo', () {
    final shape = [2, 1, 640, 640];
    final info = mnn.VariableInfo.create(
      order: mnn.DimensionFormat.NCHW,
      dim: shape,
      type: mnn.HalideType.f64,
    );
    info.syncSize();
    expect(info.order, mnn.DimensionFormat.NCHW);
    expect(info.dim, shape);
    expect(info.ndim, shape.length);
    expect(info.type, mnn.HalideType.f64);
    expect(info.size, shape.reduce((a, b) => a * b));
    expect(info.toString(), startsWith('VariableInfo('));

    info.dispose();
  });

  test('Expr', () {
    final shape = [2, 1, 640, 640];
    final info = mnn.VariableInfo.create(
      order: mnn.DimensionFormat.NCHW,
      dim: shape,
      type: mnn.HalideType.f64,
    );
    final expr = mnn.Expr.fromVariableInfo(info, mnn.InputType.CONSTANT);

    // info will be copied internally, so we can dispose it after use
    info.dispose();

    expr.name = "testExpr";
    expect(expr.name, "testExpr");
    expect(expr.outputSize, 1);
    expect(expr.requireInfo, true);
    expect(expr.inputType, mnn.InputType.CONSTANT);
    expect(expr.outputName(0), "");
    expect(expr.outputInfo(0), isA<mnn.VariableInfo>());

    final tensor = mnn.Tensor.create();
    final expr1 = mnn.Expr.fromTensor(tensor);
    expr1.name = "exprFromTensor";
    expect(expr1.name, "exprFromTensor");

    mnn.Expr.replace(expr, expr1);
    expect(expr.name, expr1.name);

    expect(expr1.toString(), startsWith('Expr('));
    expect(expr1.toString(), contains('name=exprFromTensor'));

    expr.dispose();
    expr1.dispose();
  });

  test('Variable load save', () {
    final outFile = File('test/tmp/variable_save.mnn');
    if (!outFile.parent.existsSync()) {
      outFile.parent.createSync(recursive: true);
    }
    final varp = mnn.VARP.list<mnn.i32>([1, 2, 3, 4], format: mnn.DimensionFormat.NCHW);
    print(varp.getInfo());

    mnn.VARP.saveToFile([varp], outFile.path);

    final buf = outFile.readAsBytesSync();
    final bufOut = mnn.VARP.saveToBuffer([varp]);
    expect(bufOut.data, buf);

    final varpLoaded = mnn.VARP.loadFromBuffer(bufOut.data);
    final varpLoadedFile = mnn.VARP.loadFromFile(outFile.path);
    expect(varpLoaded[0].data, varp.data);
    expect(varpLoadedFile[0].data, varp.data);

    // final varpMap = mnn.VARP.loadMapFromFile("test/data/mnist-8.mnn");
    // print(varpMap["Plus214_Output_0"]?.data);
  });
}
