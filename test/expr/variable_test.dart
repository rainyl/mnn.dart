import 'dart:ffi';
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

  test('VARP', () {
    final info = mnn.VariableInfo.create(
      order: mnn.DimensionFormat.NHWC,
      dim: [],
      type: mnn.HalideType.f64,
    );
    final expr = mnn.Expr.fromVariableInfo(info, mnn.InputType.CONSTANT);
    final varp = mnn.VARP.create(expr, index: 0);
    expect(varp.getName(), isEmpty);
    varp.setName("testVariable");
    expect(varp.getName(), "testVariable");

    final info1 = varp.getInfo();
    expect(info1.order, mnn.DimensionFormat.NHWC);
    expect(info1.dim, []);
    expect(info1.type, mnn.HalideType.f64);
    expect(info1.size, 1); // ceil
    info1.dispose();

    varp.resize([1, 2, 3, 3]);
    final info2 = varp.getInfo();
    expect(info2.order, mnn.DimensionFormat.NHWC);
    expect(info2.dim, [1, 2, 3, 3]);
    expect(info2.type, mnn.HalideType.f64);
    expect(info2.size, 1 * 2 * 3 * 3);
    info2.dispose();

    final (expr1, index) = varp.expr;
    expect(expr1.name, "testVariable");
    expect(index, 0);
    expr1.dispose();

    final varp1 = mnn.VARP.list<mnn.i32>([1, 2, 3, 4], format: mnn.DimensionFormat.NCHW);
    varp.input(varp1);
    varp1.dispose();

    final info3 = varp.getInfo();
    expect(info3.order, mnn.DimensionFormat.NCHW);
    expect(info3.dim, [4]);
    expect(info3.type, mnn.HalideType.i32);
    expect(info3.size, 4);
    info3.dispose();

    final pRead = varp.readMap<mnn.i32>();
    expect(pRead.asTypedList(4), [1, 2, 3, 4]);
    varp.unMap();

    expect(varp.linkNumber(), isA<int>());

    final tensor = varp.getTensor();
    expect(tensor.shape, [4]);

    varp.setExpr(expr, 1);
    expect(varp.expr.$1, isA<mnn.Expr>());
    expect(varp.expr.$2, 1);

    expect(varp.toString(), startsWith('VARP('));

    varp.dispose();
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
