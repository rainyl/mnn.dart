import 'dart:io';

import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

const path = "test/data/mnist_0.png";

void main() {
  group('stb image', () {
    test('load', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      expect(image.width, 28);
      expect(image.height, 28);
      expect(image.channels, 1);
      expect(image.shape, (1, 28, 28));
      expect(image.desiredChannels, mnn.StbiChannel.rgb);
    });

    test('load from memory', () {
      final bytes = File(path).readAsBytesSync();
      final image = mnn.Image.fromMemory(bytes, desiredChannel: mnn.StbiChannel.rgb);
      expect(image.width, 28);
      expect(image.height, 28);
      expect(image.channels, 1);
      expect(image.shape, (1, 28, 28));
      expect(image.desiredChannels, mnn.StbiChannel.rgb);
    });

    test('info', () {
      expect(mnn.Image.info(path), (1, 28, 28));
      final bytes = File(path).readAsBytesSync();
      expect(mnn.Image.infoFromMemory(bytes), (1, 28, 28));
    });

    test('clone', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      final cloned = image.clone();
      expect(cloned.ptr.address != image.ptr.address, true);
      expect(cloned != image, true);
      expect(cloned.width, image.width);
      expect(cloned.height, image.height);
      expect(cloned.channels, image.channels);
      expect(cloned.shape, image.shape);
      expect(cloned.desiredChannels, image.desiredChannels);
      expect(cloned.bytes, image.bytes);
      expect(cloned.values, image.values);
    });

    test('operator +', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      var resized = image.resize(28, 28, dtype: mnn.StbirDataType.STBIR_TYPE_FLOAT);
      expect(resized.row(0), List.filled(28, 0.0));
      resized += [1.0];
      expect(resized.row(0), List.filled(28, 1.0));
    });

    test('operator -', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      var resized = image.resize(28, 28, dtype: mnn.StbirDataType.STBIR_TYPE_FLOAT);
      expect(resized.row(0), List.filled(28, 0.0));
      resized -= [1.0];
      expect(resized.row(0), List.filled(28, -1.0));
    });

    test('operator *', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      var resized = image.resize(28, 28, dtype: mnn.StbirDataType.STBIR_TYPE_FLOAT);
      expect(resized.row(0), List.filled(28, 0.0));
      resized += [1.0];
      resized += [2.0];
      expect(resized.row(0), List.filled(28, 2.0));
    });

    test('operator /', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      var resized = image.resize(28, 28, dtype: mnn.StbirDataType.STBIR_TYPE_FLOAT);
      expect(resized.row(0), List.filled(28, 0.0));
      resized += [1.0];
      resized /= [2.0];
      expect(resized.row(0), List.filled(28, 0.5));
    });

    test('normalize', () {
      var image = mnn.Image.load("test/data/lenna.png", desiredChannel: mnn.StbiChannel.rgb);
      image = image.resize(224, 224, dtype: mnn.StbirDataType.STBIR_TYPE_FLOAT);
      image = image / [255.0, 255.0, 255.0];
      final normalized = image.normalize(mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]);
      expect(normalized.pixel(25, 41), [1, 2, 3]);
    });

    test('misc', () {
      final bytes = File(path).readAsBytesSync();
      expect(mnn.Image.isHdr(path), false);
      expect(mnn.Image.isHdrFromMemory(bytes), false);

      expect(mnn.Image.is16Bit(path), false);
      expect(mnn.Image.is16BitFromMemory(bytes), false);
    });

    test('resize', () {
      final image = mnn.Image.load(path, desiredChannel: mnn.StbiChannel.rgb);
      expect(image.shape, (1, 28, 28));
      final resized = image.resize(241, 241);
      expect(resized.shape, (1, 241, 241));
    });
  });
}
