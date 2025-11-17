/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'exception.dart';
import 'g/mnn.g.dart' as c;
import 'halide_runtime.dart';
import 'image/enums.dart';
import 'image/image.dart';
import 'matrix.dart';
import 'tensor.dart';

class ImageProcessConfig extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(calloc.nativeFree);
  ImageProcessConfig.fromPointer(
    ffi.Pointer<c.mnn_image_process_config_t> ptr, {
    super.attach,
    super.externalSize,
  }) : super(ptr.cast());

  factory ImageProcessConfig.create({
    ImageFormat sourceFormat = ImageFormat.RGBA,
    ImageFormat destFormat = ImageFormat.RGBA,
    Filter filterType = Filter.NEAREST,
    Wrap wrap = Wrap.CLAMP_TO_EDGE,
    List<double>? mean,
    List<double>? normal,
  }) {
    final ptr = calloc<c.mnn_image_process_config_t>()
      ..ref.sourceFormat = sourceFormat.value
      ..ref.destFormat = destFormat.value
      ..ref.filterType = filterType.value
      ..ref.wrap = wrap.value;
    if (mean != null) {
      if (mean.length > 4) {
        throw ArgumentError('mean.length=${mean.length} > 4');
      }
      for (var i = 0; i < mean.length; i++) {
        ptr.ref.mean[i] = mean[i];
      }
    }
    if (normal != null) {
      if (normal.length > 4) {
        throw ArgumentError('normal.length=${normal.length} > 4');
      }
      for (var i = 0; i < normal.length; i++) {
        ptr.ref.normal[i] = normal[i];
      }
    }
    return ImageProcessConfig.fromPointer(ptr.cast());
  }

  c.mnn_image_process_config_t get ref => ptr.cast<c.mnn_image_process_config_t>().ref;

  Filter get filterType => Filter.fromValue(ref.filterType);
  set filterType(Filter value) => ref.filterType = value.value;

  ImageFormat get sourceFormat => ImageFormat.fromValue(ref.sourceFormat);
  set sourceFormat(ImageFormat value) => ref.sourceFormat = value.value;

  ImageFormat get destFormat => ImageFormat.fromValue(ref.destFormat);
  set destFormat(ImageFormat value) => ref.destFormat = value.value;

  List<double> get mean => List<double>.generate(4, (i) => ref.mean[i]);
  void setMean(List<double> value) {
    if (value.length != 4) {
      throw ArgumentError('value.length=${value.length}!= 4');
    }
    for (var i = 0; i < 4; i++) {
      ref.mean[i] = value[i];
    }
  }

  List<double> get normal => List<double>.generate(4, (i) => ref.normal[i]);
  void setNormal(List<double> value) {
    if (value.length != 4) {
      throw ArgumentError('value.length=${value.length}!= 4');
    }
    for (var i = 0; i < 4; i++) {
      ref.normal[i] = value[i];
    }
  }

  Wrap get wrap => Wrap.fromValue(ref.wrap);
  set wrap(Wrap value) => ref.wrap = value.value;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [sourceFormat, destFormat, filterType, wrap, mean, normal];

  @override
  String toString() {
    return 'ImageProcessConfig(address=0x${ptr.address.toRadixString(16)}, $sourceFormat, $destFormat, $filterType, $wrap)';
  }

  @override
  void release() {
    calloc.free(ptr.cast<c.mnn_image_process_config_t>());
  }
}

class ImageProcess extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(c.addresses.mnn_cv_image_process_destroy);
  ImageProcess.fromPointer(ffi.Pointer<c.mnn_cv_image_process_t> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  factory ImageProcess.fromConfig(ImageProcessConfig config, {Tensor? dstTendor}) {
    final dstPtr = dstTendor == null ? ffi.nullptr : dstTendor.ptr;
    final ptr = c.mnn_cv_image_process_create_with_config(config.ref, dstPtr);
    return ImageProcess.fromPointer(ptr.cast());
  }

  factory ImageProcess.create({
    ImageFormat sourceFormat = ImageFormat.RGBA,
    ImageFormat destFormat = ImageFormat.RGBA,
    List<double>? means,
    List<double>? normals,
    Filter filterType = Filter.BILINEAR,
    Wrap wrap = Wrap.CLAMP_TO_EDGE,
    Tensor? dstTendor,
  }) {
    final pMeans = means == null ? ffi.nullptr : calloc<ffi.Float>(means.length);
    final pNormals = normals == null ? ffi.nullptr : calloc<ffi.Float>(normals.length);
    if (pMeans != ffi.nullptr && means != null) {
      pMeans.asTypedList(means.length).setAll(0, means);
    }
    if (pNormals != ffi.nullptr && normals != null) {
      pNormals.asTypedList(normals.length).setAll(0, normals);
    }
    final dstPtr = dstTendor == null ? ffi.nullptr : dstTendor.ptr;
    final ptr = c.mnn_cv_image_process_create(
      sourceFormat.value,
      destFormat.value,
      pMeans,
      means?.length ?? 0,
      pNormals,
      normals?.length ?? 0,
      filterType.value,
      wrap.value,
      dstPtr,
    );
    calloc.free(pMeans);
    calloc.free(pNormals);
    return ImageProcess.fromPointer(ptr.cast());
  }

  /// @brief create tensor with given data.
  ///
  /// @param width     image width.
  /// @param height     image height.
  /// @param bytesPerChannel   bytes per pixel.
  /// @param image    image.
  ///
  /// @return created tensor.
  static Tensor createImageTensor(
    HalideType type,
    int width,
    int height,
    int bytesPerChannel, {
    Image? image,
  }) {
    final pImage = image == null ? ffi.nullptr : image.ptr;
    final ptr =
        c.mnn_cv_image_process_create_image_tensor(type.native.ref, width, height, bytesPerChannel, pImage);
    return Tensor.fromPointer(ptr);
  }

  void setPadding(int value) => c.mnn_cv_image_process_set_padding(ptr, value);

  void setDraw() => c.mnn_cv_image_process_set_draw(ptr);

  /// @brief draw color to regions of img.
  ///
  /// @param img  the image to draw.
  /// @param w  the image's width.
  /// @param h  the image's height.
  /// @param c  the image's channel.
  /// @param regions  the regions to draw, size is [num * 3] contain num x { y, xl, xr }
  /// @param num  regions num
  /// @param color  the color to draw.
  void draw(Image img, int width, int height, int channel, List<int> regions, List<int> colors) {
    final pRegions = calloc<ffi.Int>(regions.length);
    pRegions.cast<ffi.Int32>().asTypedList(regions.length).setAll(0, regions);
    final pColors = calloc<ffi.Uint8>(colors.length);
    pColors.asTypedList(colors.length).setAll(0, colors);
    c.mnn_cv_image_process_draw(
      ptr,
      img.ptr.cast(),
      width,
      height,
      channel,
      pRegions,
      regions.length,
      pColors,
    );
    calloc.free(pRegions);
    calloc.free(pColors);
  }

  Matrix get matrix => Matrix.fromPointer(c.mnn_cv_image_process_get_matrix(ptr).cast());

  set matrix(Matrix value) {
    final code = c.mnn_cv_image_process_set_matrix(ptr, value.ptr.cast());
    if (code != c.ErrorCode.NO_ERROR) {
      throw MNNException('set matrix failed');
    }
  }

  Tensor convert(Uint8List src, int iw, int ih, int stride, Tensor dst) {
    final pSrc = calloc<ffi.Uint8>(src.length);
    pSrc.asTypedList(src.length).setAll(0, src);
    final code = c.mnn_cv_image_process_convert(ptr, pSrc, iw, ih, stride, dst.ptr);
    calloc.free(pSrc);
    if (code != c.ErrorCode.NO_ERROR) {
      throw MNNException('convert failed');
    }
    return dst;
  }

  Tensor convertImage(Image src, Tensor dst) {
    final code = c.mnn_cv_image_process_convert(ptr, src.ptr.cast(), src.width, src.height, 0, dst.ptr);
    if (code != c.ErrorCode.NO_ERROR) {
      throw MNNException('convert failed');
    }
    return dst;
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'ImageProcess(address=0x${ptr.address.toRadixString(16)})';
  }

  @override
  void release() {
    c.mnn_cv_image_process_destroy(ptr);
  }
}

enum Filter {
  NEAREST(0),
  BILINEAR(1),
  BICUBIC(2);

  final int value;
  const Filter(this.value);

  static Filter fromValue(int value) => switch (value) {
        0 => NEAREST,
        1 => BILINEAR,
        2 => BICUBIC,
        _ => throw MNNException('Unknown Filter: $value'),
      };
}

enum Wrap {
  CLAMP_TO_EDGE(0),
  ZERO(1),
  REPEAT(2);

  final int value;
  const Wrap(this.value);

  static Wrap fromValue(int value) => switch (value) {
        0 => CLAMP_TO_EDGE,
        1 => ZERO,
        2 => REPEAT,
        _ => throw MNNException('Unknown Wrap: $value'),
      };
}
