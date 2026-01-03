import 'dart:io';

import 'package:mnn/cv.dart' as cv;
import 'package:test/test.dart';

void main() {
  const imagePath = 'test/data/lenna.png';

  group('Image Loading and Basic Info', () {
    test('Image.load', () {
      final image = cv.Image.load(imagePath);
      expect(image.width, greaterThan(0));
      expect(image.height, greaterThan(0));
      expect(image.channels, equals(3));
      expect(image.dtype, equals(cv.StbiDType.u8));
      image.dispose();
    });

    test('Image.fromMemory', () {
      final bytes = File(imagePath).readAsBytesSync();
      final image = cv.Image.fromMemory(bytes);
      expect(image.width, greaterThan(0));
      expect(image.height, greaterThan(0));
      expect(image.channels, equals(3));
      image.dispose();
    });

    test('Image.info', () {
      final info = cv.Image.info(imagePath);
      expect(info.$1, isNot(0)); // result
      expect(info.$2, equals(3)); // channels
      expect(info.$3, greaterThan(0)); // width
      expect(info.$4, greaterThan(0)); // height
    });

    test('Image.infoFromMemory', () {
      final bytes = File(imagePath).readAsBytesSync();
      final info = cv.Image.infoFromMemory(bytes);
      expect(info.$1, isNot(0));
      expect(info.$2, equals(3));
      expect(info.$3, greaterThan(0));
      expect(info.$4, greaterThan(0));
    });

    test('Image.isHdr', () {
      expect(cv.Image.isHdr(imagePath), isFalse);
    });

    test('Image.is16Bit', () {
      expect(cv.Image.is16Bit(imagePath), isFalse);
    });
  });

  group('Image Operations', () {
    late cv.Image image;

    setUp(() {
      image = cv.Image.load(imagePath);
    });

    tearDown(() {
      image.dispose();
    });

    test('Image.clone', () {
      final cloned = image.clone();
      expect(cloned.width, equals(image.width));
      expect(cloned.height, equals(image.height));
      expect(cloned.channels, equals(image.channels));
      expect(cloned.dtype, equals(image.dtype));
      cloned.dispose();
    });

    test('Image.resize', () {
      final resized = image.resize(100, 100);
      expect(resized.width, equals(100));
      expect(resized.height, equals(100));
      expect(resized.channels, equals(image.channels));
      resized.dispose();
    });

    test('Image.normalize', () {
      final normalized = image.normalize();
      expect(normalized.dtype, equals(cv.StbiDType.f32));
      expect(normalized.width, equals(image.width));
      expect(normalized.height, equals(image.height));
      normalized.dispose();
    });

    test('Image.toPlanar', () {
      final planar = image.toPlanar();
      expect(planar.width, equals(image.width));
      expect(planar.height, equals(image.height));
      expect(planar.channels, equals(image.channels));
      planar.dispose();
    });
  });

  group('Image Data Access', () {
    late cv.Image image;

    setUp(() {
      image = cv.Image.load(imagePath);
    });

    tearDown(() {
      image.dispose();
    });

    test('Image.bytes and values', () {
      final bytes = image.bytes;
      final values = image.values;
      expect(bytes.length, equals(image.elemCount));
      expect(values.length, equals(image.elemCount));
    });

    test('Image.pixel, row, col', () {
      final pixel = image.pixel(0, 0);
      expect(pixel.length, equals(image.channels));

      final row = image.row(0);
      expect(row.length, equals(image.width * image.channels));

      final col = image.col(0);
      expect(col.length, equals(image.height * image.channels));
    });

    test('Image.forEachPixel', () {
      int count = 0;
      image.forEachPixel((x, y, pixel) {
        count++;
      });
      expect(count, equals(image.width * image.height));
    });
  });

  group('Image Saving', () {
    late cv.Image image;

    setUp(() {
      image = cv.Image.load(imagePath);
    });

    tearDown(() {
      image.dispose();
    });

    test('Image.save PNG', () {
      const savePath = 'test/data/save_test.png';
      final result = image.save(savePath);
      expect(result, isNot(0));
      expect(File(savePath).existsSync(), isTrue);
      File(savePath).deleteSync();
    });

    test('Image.save JPG', () {
      const savePath = 'test/data/save_test.jpg';
      final result = image.save(savePath);
      expect(result, isNot(0));
      expect(File(savePath).existsSync(), isTrue);
      File(savePath).deleteSync();
    });

    test('Image.save BMP', () {
      const savePath = 'test/data/save_test.bmp';
      final result = image.save(savePath);
      expect(result, isNot(0));
      expect(File(savePath).existsSync(), isTrue);
      File(savePath).deleteSync();
    });

    test('Image.save', () {

    });
  });
}
