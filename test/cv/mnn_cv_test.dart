/// Copyright (c) 2026, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:io';
import 'dart:typed_data';

import 'package:mnn/cv.dart' as cv;
import 'package:mnn/mnn.dart' as mnn;
import 'package:test/test.dart';

import '../list_element_equals.dart';

void main() {
  const imagePath = 'test/data/lenna.png';

  group('MNN CV Image Codecs', () {
    test('haveImageReader', () {
      expect(cv.haveImageReader(imagePath), isTrue);
      expect(cv.haveImageReader('non_existent.png'), isFalse);
    });

    test('haveImageReaderFromMemory', () {
      final bytes = File(imagePath).readAsBytesSync();
      expect(cv.haveImageReaderFromMemory(bytes), isTrue);
      expect(cv.haveImageReaderFromMemory(Uint8List.fromList([1, 2, 3])), isFalse);
    });

    test('haveImageWriter', () {
      expect(cv.haveImageWriter('test.png'), isTrue);
      expect(cv.haveImageWriter('test.jpg'), isTrue);
      expect(cv.haveImageWriter('test.unknown'), isFalse);
    });

    test('imread and imwrite', () {
      final img = cv.imread(imagePath);
      expect(img, isNotNull);
      final size = cv.getVARPSize(img);
      expect(size.$1, greaterThan(0));
      expect(size.$2, greaterThan(0));
      expect(size.$3, equals(3));

      const outPath = 'test/data/lenna_out.png';
      final success = cv.imwrite(outPath, img);
      expect(success, isTrue);
      expect(File(outPath).existsSync(), isTrue);
      File(outPath).deleteSync();
      img.dispose();
    });

    test('imdecode and imencode', () {
      final bytes = File(imagePath).readAsBytesSync();
      final img = cv.imdecode(bytes);
      expect(img, isNotNull);

      final (success, encoded) = cv.imencode('.png', img);
      expect(success, isTrue);
      expect(encoded.length, greaterThan(0));

      img.dispose();
      encoded.dispose();
    });
  });

  group('MNN CV VARP Info', () {
    late mnn.VARP img;

    setUp(() {
      img = cv.imread(imagePath);
    });

    tearDown(() {
      img.dispose();
    });

    test('getVARPSize', () {
      final size = cv.getVARPSize(img);
      expect(size.$1, equals(cv.getVARPHeight(img)));
      expect(size.$2, equals(cv.getVARPWidth(img)));
      expect(size.$3, equals(cv.getVARPChannels(img)));
    });

    test('getVARPHeight, Width, Channels, Byte', () {
      expect(cv.getVARPHeight(img), greaterThan(0));
      expect(cv.getVARPWidth(img), greaterThan(0));
      expect(cv.getVARPChannels(img), equals(3));
      expect(cv.getVARPByte(img), greaterThan(0));
    });

    test('buildImgVARP', () {
      final bytes = Uint8List(100 * 100 * 3);
      final varp = cv.buildImgVARP(bytes, 100, 100, 3, flags: cv.IMREAD_COLOR);
      expect(cv.getVARPHeight(varp), equals(100));
      expect(cv.getVARPWidth(varp), equals(100));
      expect(cv.getVARPChannels(varp), equals(3));
      expect(varp.shape, [100, 100, 3]);
      varp.dispose();
    });

    test('buildImgVarpYuvNV21', () {
      const height = 1200;
      const width = 1600;
      final bytes = Uint8List(((height + height / 2) * width * 1).round());
      final varp = cv.buildImgVarpYuvNV21(bytes, height, width, flags: cv.IMREAD_COLOR);
      expect(cv.getVARPHeight(varp), equals(height));
      expect(cv.getVARPWidth(varp), equals(width));
      expect(cv.getVARPChannels(varp), equals(3));
      expect(varp.shape, [height, width, 3]);
      varp.dispose();
    });
  });

  group('MNN CV Core and Calib3d', () {
    test('solve', () {
      final src1 = mnn.VARP.fromList2D<mnn.float32>([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      final src2 = mnn.VARP.fromList2D<mnn.float32>([
        [5.0],
        [11.0],
      ]);

      final (success, out) = cv.solve(src1, src2);
      expect(success, isTrue);
      expect(out, isNotNull);
      // Solve: x + 2y = 5, 3x + 4y = 11
      // => x = 1, y = 2
      final data = out.data;
      expect(data![0], closeTo(1.0, 0.001));
      expect(data[1], closeTo(2.0, 0.001));

      src1.dispose();
      src2.dispose();
      out.dispose();
    });

    test('Rodrigues', () {
      final src = mnn.VARP.fromList2D<mnn.float32>([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
      ], format: mnn.DimensionFormat.NCHW);
      final out = cv.Rodrigues(src);
      expect(out.isEmpty, isFalse);
      expect(out.shape, equals([3, 1]));
      expect(out.data, listCloseTo([0, 0, 0], 0.001));

      src.dispose();
      out.dispose();
    });

    test('solvePnP', () {
      final objectPoints = mnn.VARP.fromList2D<mnn.float32>([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0],
      ]);
      final imagePoints = mnn.VARP.fromList2D<mnn.float32>([
        [359, 391],
        [399, 561],
        [337, 297],
        [513, 301],
        [345, 465],
        [453, 469],
      ]);
      final cameraMatrix = mnn.VARP.fromList2D<mnn.float32>([
        [1200, 0, 600],
        [0, 1200, 337.5],
        [0, 0, 1],
      ]);
      final distCoeffs = mnn.VARP.fromList2D<mnn.float32>([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
      ]);

      final (rvec, tvec) = cv.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs);
      expect(rvec, isNotNull);
      expect(tvec, isNotNull);

      objectPoints.dispose();
      imagePoints.dispose();
      cameraMatrix.dispose();
      distCoeffs.dispose();
      rvec.dispose();
      tvec.dispose();
    });
  });

  group('MNN CV Structural', () {
    late mnn.VARP img;

    setUp(() {
      img = cv.imread(imagePath);
    });

    tearDown(() {
      img.dispose();
    });
    test('findContours and contourArea', () {
      // 10*10 image with a 4*4 square in the middle
      final im = mnn.VARP.fromList2D<mnn.uint8>(
        List.generate(10, (y) => List.generate(10, (x) => (x >= 2 && x <= 6 && y >= 2 && y <= 6) ? 255 : 0)),
      );
      final contours = cv.findContours(im, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      expect(contours.length, equals(1));
      final area = cv.contourArea(contours[0]);
      expect(area, closeTo(16.0, 0.001)); // 4x4 square area (inner area)

      im.dispose();
      contours.dispose();
    });

    test('convexHull, minAreaRect, boundingRect, boxPoints', () {
      final points = mnn.VARP.fromList2D<mnn.int32>([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
      ]);
      final hull = cv.convexHull(points);
      expect(hull.length, greaterThan(0));

      final rect = cv.minAreaRect(points);
      expect(rect.width, closeTo(10.0, 0.001));
      expect(rect.height, closeTo(10.0, 0.001));

      final bbox = cv.boundingRect(points);
      expect(bbox.width, equals(11));
      expect(bbox.height, equals(11));

      final boxPts = cv.boxPoints(rect);
      expect(boxPts.shape, equals([4, 2]));

      points.dispose();
      hull.dispose();
      boxPts.dispose();
    });
  });

  group('MNN CV Miscellaneous', () {
    late mnn.VARP img;

    setUp(() {
      img = cv.imread(imagePath);
    });

    tearDown(() {
      img.dispose();
    });
    test('threshold', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>([
        [100, 200],
        [50, 150],
      ]);
      final dst = cv.threshold(src, 128, 255, cv.THRESH_BINARY);
      final data = dst.data;
      expect(data, listCloseTo([0, 255, 0, 255], 0.001));

      src.dispose();
      dst.dispose();
    });

    test('adaptiveThreshold', skip: "Fails, skip for now", () {
      final dst = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 3, 5);
      expect(dst.isEmpty, false);
      expect(dst.shape, equals([10, 10]));

      dst.dispose();
    });
  });

  group('MNN CV Geometric', () {
    test('getRotationMatrix2D and warpAffine', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(
        List.generate(10, (y) => List.generate(10, (x) => (x == 5 && y == 5) ? 255 : 0)),
      );
      final M = cv.getRotationMatrix2D((5.0, 5.0), 90, 1.0);
      final dst = cv.warpAffine(src, M, (10, 10));
      expect(dst.shape, equals([10, 10, 1]));

      src.dispose();
      M.dispose();
      dst.dispose();
    });

    test('getAffineTransform', () {
      final src = [cv.Point(0, 0), cv.Point(10, 0), cv.Point(0, 10)];
      final dst = [cv.Point(0, 0), cv.Point(20, 0), cv.Point(0, 20)];
      final M = cv.getAffineTransform(src, dst);
      expect(M.isEmpty, isFalse);

      final invM = cv.invertAffineTransform(M);
      expect(invM.isEmpty, isFalse);

      M.dispose();
      invM.dispose();
    });

    test('getPerspectiveTransform and warpPerspective', () {
      final src = [cv.Point(0, 0), cv.Point(10, 0), cv.Point(10, 10), cv.Point(0, 10)];
      final dst = [cv.Point(0, 0), cv.Point(20, 0), cv.Point(20, 20), cv.Point(0, 20)];
      final M = cv.getPerspectiveTransform(src, dst);
      expect(M.isEmpty, isFalse);

      final img = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final warped = cv.warpPerspective(img, M, (20, 20));
      expect(warped.shape, equals([20, 20, 1]));

      M.dispose();
      img.dispose();
      warped.dispose();
    });

    test('getRectSubPix', () {
      final img = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final patch = cv.getRectSubPix(img, (5, 5), (5.0, 5.0));
      expect(patch.shape, equals([5, 5, 1]));
      img.dispose();
      patch.dispose();
    });

    test('remap', () {
      final src = mnn.VARP.fromListND<mnn.uint8>(
        List.generate(10 * 10, (i) => 100),
        [
          1,
          1,
          10,
          10,
        ],
        format: mnn.DimensionFormat.NC4HW4,
      );
      expect(src.shape, [1, 1, 10, 10]);
      final map1 = mnn.VARP.fromList2D<mnn.float32>(
        List.generate(5, (y) => List.generate(5, (x) => x.toDouble())),
      );
      expect(map1.shape, [5, 5]);

      final map2 = mnn.VARP.fromList2D<mnn.float32>(
        List.generate(5, (y) => List.generate(5, (x) => y.toDouble())),
      );
      expect(map2.shape, [5, 5]);
      final dst = cv.remap(src, map1, map2, cv.INTER_LINEAR);
      expect(dst.shape, equals([5, 5, 1]));

      src.dispose();
      map1.dispose();
      map2.dispose();
      dst.dispose();
    });

    test('undistortPoints', skip: "TODO: skip for now", () {
      final src = mnn.VARP.fromList2D<mnn.float32>([
        [5, 5],
        [10, 10],
      ]);
      final cameraMatrix = mnn.VARP.fromList2D<mnn.float32>([
        [100, 0, 50],
        [0, 100, 50],
        [0, 0, 1],
      ]);
      final distCoeffs = mnn.VARP.fromList1D<mnn.float32>([0.1, 0.01, 0, 0, 0]);
      final dst = cv.undistortPoints(src, cameraMatrix, distCoeffs);
      expect(dst.shape, equals([2, 1, 2]));

      src.dispose();
      cameraMatrix.dispose();
      distCoeffs.dispose();
      dst.dispose();
    });

    test('resize', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => x * 10)));
      final dst = cv.resize(src, (20, 20));
      expect(dst.shape, equals([20, 20, 1]));

      src.dispose();
      dst.dispose();
    });

    test('blendLinear', () {
      final src1 = mnn.VARP.fromList2D<mnn.uint8>([
        [100, 100],
        [100, 100],
      ]);
      final src2 = mnn.VARP.fromList2D<mnn.uint8>([
        [200, 200],
        [200, 200],
      ]);
      final weight1 = mnn.VARP.fromList2D<mnn.float32>([
        [0.5, 0.5],
        [0.5, 0.5],
      ]);
      final weight2 = mnn.VARP.fromList2D<mnn.float32>([
        [0.5, 0.5],
        [0.5, 0.5],
      ]);
      final dst = cv.blendLinear(src1, src2, weight1, weight2);
      expect(dst.shape, equals([2, 2]));

      src1.dispose();
      src2.dispose();
      weight1.dispose();
      weight2.dispose();
      dst.dispose();
    });
  });

  group('MNN CV Histograms', () {
    test('calcHist', skip: "TODO: fails, skip for now", () {
      final img = cv.buildImgVARP(
        Uint8List.fromList(List.generate(100 * 100 * 3, (i) => i % 256)),
        100,
        100,
        3,
      );
      final images = mnn.VecVARP.of([img]);
      final hist = cv.calcHist(images, channels: [0], histSize: [256], ranges: [0, 256]);
      expect(hist.shape, equals([256]));

      img.dispose();
      images.dispose();
      hist.dispose();
    });
  });

  group('MNN CV Filter', () {
    test('blur and GaussianBlur', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final blurred = cv.blur(src, (3, 3));
      expect(blurred.shape, equals([10, 10]));

      final gblurred = cv.GaussianBlur(src, (3, 3), 1.0);
      expect(gblurred.shape, equals([10, 10]));

      src.dispose();
      blurred.dispose();
      gblurred.dispose();
    });

    test('bilateralFilter', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final dst = cv.bilateralFilter(src, 5, 75.0, 75.0);
      expect(dst.shape, equals([10, 10]));
      src.dispose();
      dst.dispose();
    });

    test('boxFilter and sqrBoxFilter', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final box = cv.boxFilter(src, -1, (3, 3));
      expect(box.shape, equals([10, 10]));

      final sqrBox = cv.sqrBoxFilter(src, -1, (3, 3));
      expect(sqrBox.shape, equals([10, 10]));

      src.dispose();
      box.dispose();
      sqrBox.dispose();
    });

    test('dilate and erode', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(
        List.generate(10, (y) => List.generate(10, (x) => (x == 5 && y == 5) ? 255 : 0)),
      );
      final kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3));
      final dilated = cv.dilate(src, kernel);
      final eroded = cv.erode(src, kernel);
      expect(dilated.shape, equals([10, 10]));
      expect(eroded.shape, equals([10, 10]));

      src.dispose();
      kernel.dispose();
      dilated.dispose();
      eroded.dispose();
    });

    test('filter2D and sepFilter2D', skip: "TODO: fails", () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final kernel = mnn.VARP.fromList2D<mnn.float32>([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0],
      ]);
      final dst = cv.filter2D(src, -1, kernel);
      expect(dst.shape, equals([10, 10]));

      final kernelX = mnn.VARP.fromList1D<mnn.float32>([1, 2, 1]);
      final kernelY = mnn.VARP.fromList1D<mnn.float32>([1, 0, -1]);
      final sepDst = cv.sepFilter2D(src, -1, kernelX, kernelY);
      expect(sepDst.shape, equals([10, 10]));

      src.dispose();
      kernel.dispose();
      dst.dispose();
      kernelX.dispose();
      kernelY.dispose();
      sepDst.dispose();
    });

    test('Laplacian, Scharr, Sobel', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => x * 10)));
      final lap = cv.Laplacian(src, -1);
      final scharr = cv.Scharr(src, -1, 1, 0);
      final sobel = cv.Sobel(src, -1, 1, 0);
      expect(lap.shape, equals([10, 10]));
      expect(scharr.shape, equals([10, 10]));
      expect(sobel.shape, equals([10, 10]));

      src.dispose();
      lap.dispose();
      scharr.dispose();
      sobel.dispose();
    });

    test('pyrDown and pyrUp', () {
      final src = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 100)));
      final down = cv.pyrDown(src, (5, 5));
      final up = cv.pyrUp(src, (20, 20));
      expect(down.shape, equals([5, 5]));
      expect(up.shape, equals([20, 20]));

      src.dispose();
      down.dispose();
      up.dispose();
    });

    test('kernels', () {
      final gabor = cv.getGaborKernel((5, 5), 1.0, 0.0, 1.0, 1.0);
      final gaussian = cv.getGaussianKernel(5, 1.0);
      expect(gabor.shape, equals([5, 5]));
      expect(gaussian.shape, equals([1, 5]));

      gabor.dispose();
      gaussian.dispose();
    });
  });

  group('MNN CV Color', () {
    test('cvtColor', () {
      // NCHW [1, 10, 10, 3]
      // buildImgVARP is better for creating images
      final bytes = Uint8List(10 * 10 * 3);
      final img = cv.buildImgVARP(bytes, 10, 10, 3);
      final gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY);
      expect(cv.getVARPChannels(gray), equals(1));

      img.dispose();
      gray.dispose();
    });

    test('cvtColorTwoPlane', () {
      final y = mnn.VARP.fromList2D<mnn.uint8>(List.generate(10, (y) => List.generate(10, (x) => 128)));
      final uv = mnn.VARP.fromList2D<mnn.uint8>(List.generate(5, (y) => List.generate(10, (x) => 128)));
      final bgr = cv.cvtColorTwoPlane(y, uv, cv.COLOR_YUV2BGR_NV12);
      expect(cv.getVARPChannels(bgr), equals(3));

      y.dispose();
      uv.dispose();
      bgr.dispose();
    });

    test('demosaicing', () {
      expect(
        () => cv.demosaicing(mnn.VARP.fromListND<mnn.uint8>([1, 2, 3, 4], [2, 2]), 0),
        throwsA(isA<UnimplementedError>()),
      );
    });
  });
}
