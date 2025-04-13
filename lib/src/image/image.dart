import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import '../base.dart';
import '../exception.dart';
import '../g/mnn.g.dart' as c;
import 'enums.dart';

class Image extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(c.addresses.stbi_image_free);

  int _width;
  int _height;
  int _channels;
  StbiChannel _desiredChannels;

  int get width => _width;
  int get height => _height;
  int get channels => _channels;
  StbiChannel get desiredChannels => _desiredChannels;
  (int, int, int) get shape => (_channels, _width, _height);
  int get elemCount => _width * _height * _channels;

  final StbiDType dtype;

  Image.fromPointer(
    ffi.Pointer<ffi.SizedNativeType> ptr, {
    required int width,
    required int height,
    required int channels,
    required StbiChannel desiredChannels,
    required this.dtype,
    super.attach,
  })  : _width = width,
        _height = height,
        _channels = channels,
        _desiredChannels = desiredChannels,
        super(ptr.cast(), externalSize: width * height * channels);

  factory Image.load(
    String path, {
    StbiChannel desiredChannel = StbiChannel.default_,
    StbiDType dtype = StbiDType.u8,
  }) {
    final pPath = path.toNativeUtf8().cast<ffi.Char>();
    final pW = calloc<ffi.Int>();
    final pH = calloc<ffi.Int>();
    final pChannels = calloc<ffi.Int>();
    final ptr = switch (dtype) {
      StbiDType.u8 => c.stbi_load(pPath, pW, pH, pChannels, desiredChannel.value),
      StbiDType.u16 => c.stbi_load_16(pPath, pW, pH, pChannels, desiredChannel.value),
      StbiDType.f32 => c.stbi_loadf(pPath, pW, pH, pChannels, desiredChannel.value),
    };
    if (ptr == ffi.nullptr) {
      final pMsg = c.stbi_failure_reason();
      throw MNNException("Failed to load image: $path, ${pMsg.cast<Utf8>().toDartString()}");
    }
    final width = pW.value;
    final height = pH.value;
    final channels = pChannels.value;
    calloc.free(pPath);
    calloc.free(pW);
    calloc.free(pH);
    calloc.free(pChannels);

    return Image.fromPointer(
      ptr,
      width: width,
      height: height,
      channels: channels,
      desiredChannels: desiredChannel,
      dtype: dtype,
    );
  }

  factory Image.fromMemory(
    Uint8List bytes, {
    StbiChannel desiredChannel = StbiChannel.default_,
    StbiDType dtype = StbiDType.u8,
  }) {
    final pBytes = malloc<ffi.Uint8>(bytes.length);
    pBytes.asTypedList(bytes.length).setAll(0, bytes);
    final pW = calloc<ffi.Int>();
    final pH = calloc<ffi.Int>();
    final pChannels = calloc<ffi.Int>();
    final ptr = switch (dtype) {
      StbiDType.u8 =>
        c.stbi_load_from_memory(pBytes.cast(), bytes.length, pW, pH, pChannels, desiredChannel.value),
      StbiDType.u16 =>
        c.stbi_load_16_from_memory(pBytes.cast(), bytes.length, pW, pH, pChannels, desiredChannel.value),
      StbiDType.f32 =>
        c.stbi_loadf_from_memory(pBytes.cast(), bytes.length, pW, pH, pChannels, desiredChannel.value),
    };
    if (ptr == ffi.nullptr) {
      final pMsg = c.stbi_failure_reason();
      throw MNNException("Failed to load image from memory: ${pMsg.cast<Utf8>().toDartString()}");
    }
    final width = pW.value;
    final height = pH.value;
    final channels = pChannels.value;
    calloc.free(pBytes);
    calloc.free(pW);
    calloc.free(pH);
    calloc.free(pChannels);
    return Image.fromPointer(
      ptr,
      width: width,
      height: height,
      channels: channels,
      desiredChannels: desiredChannel,
      dtype: dtype,
    );
  }

  Image clone() {
    final pBytes = malloc<ffi.Uint8>(bytes.length);
    pBytes.asTypedList(bytes.length).setAll(0, bytes);
    return Image.fromPointer(
      pBytes.cast(),
      width: _width,
      height: _height,
      channels: _channels,
      desiredChannels: _desiredChannels,
      dtype: dtype,
    );
  }

  /// will clone data
  // Image astype(StbiDType dtype) {
  //   if (dtype == this.dtype) {
  //     return clone();
  //   }
  // }

  Image normalize({
    List<double> mean = const [0.0, 0.0, 0.0, 0.0],
    List<double> std = const [1.0, 1.0, 1.0, 1.0],
    double scale = 1.0,
  }) {
    final size = _width * _height * _channels;
    final p = calloc<ffi.Float>(size);
    p.asTypedList(size).setAll(
          0,
          values.indexed
              .map((e) => (e.$2.toDouble() / scale - mean[e.$1 % _channels]) / std[e.$1 % _channels])
              .toList(),
        );
    return Image.fromPointer(
      p.cast(),
      width: _width,
      height: _height,
      channels: _channels,
      desiredChannels: _desiredChannels,
      dtype: StbiDType.f32,
    );
  }

  // Image normalize({List<double>? mean, List<double>? std, bool inplace = false}) {
  //   mean ??= List.filled(channels, 0.0);
  //   std ??= List.filled(channels, 1.0);
  //   if (mean.length != channels || std.length != channels) {
  //     throw Exception("mean and std must have the same length as channels");
  //   }

  //   final dst = inplace ? this : clone();

  //   dst.forEachPixel((i, j, pix) {
  //     for (var c = 0; c < channels; c++) {
  //       pix[c] = (pix[c] - mean![c]) / std![c];
  //     }
  //   });
  //   return dst;
  // }

  static bool isHdr(String path) {
    final pPath = path.toNativeUtf8().cast<ffi.Char>();
    final result = c.stbi_is_hdr(pPath);
    calloc.free(pPath);
    return result != 0;
  }

  static bool isHdrFromMemory(Uint8List bytes) {
    final pBytes = malloc<ffi.Uint8>(bytes.length);
    pBytes.asTypedList(bytes.length).setAll(0, bytes);
    final result = c.stbi_is_hdr_from_memory(pBytes.cast(), bytes.length);
    calloc.free(pBytes);
    return result != 0;
  }

  static (int result, int channels, int width, int height) info(String path) {
    final pPath = path.toNativeUtf8().cast<ffi.Char>();
    final pW = calloc<ffi.Int>();
    final pH = calloc<ffi.Int>();
    final pChannels = calloc<ffi.Int>();
    final result = c.stbi_info(pPath, pW, pH, pChannels);
    final rval = (result, pChannels.value, pW.value, pH.value);
    calloc.free(pPath);
    calloc.free(pW);
    calloc.free(pH);
    calloc.free(pChannels);
    return rval;
  }

  static (int result, int channels, int width, int height) infoFromMemory(Uint8List bytes) {
    final pBytes = malloc<ffi.Uint8>(bytes.length);
    pBytes.asTypedList(bytes.length).setAll(0, bytes);
    final pW = calloc<ffi.Int>();
    final pH = calloc<ffi.Int>();
    final pChannels = calloc<ffi.Int>();
    final result = c.stbi_info_from_memory(pBytes.cast(), bytes.length, pW, pH, pChannels);
    final rval = (result, pChannels.value, pW.value, pH.value);
    calloc.free(pBytes);
    calloc.free(pW);
    calloc.free(pH);
    calloc.free(pChannels);
    return rval;
  }

  static bool is16Bit(String path) {
    final pPath = path.toNativeUtf8().cast<ffi.Char>();
    final result = c.stbi_is_16_bit(pPath);
    calloc.free(pPath);
    return result != 0;
  }

  static bool is16BitFromMemory(Uint8List bytes) {
    final pBytes = malloc<ffi.Uint8>(bytes.length);
    pBytes.asTypedList(bytes.length).setAll(0, bytes);
    final result = c.stbi_is_16_bit_from_memory(pBytes.cast(), bytes.length);
    calloc.free(pBytes);
    return result != 0;
  }

  Uint8List get bytes {
    final size = _width * _height * _channels;
    final p = switch (dtype) {
      StbiDType.u8 => ptr.cast<ffi.Uint8>().asTypedList(size).buffer.asUint8List(),
      StbiDType.u16 => ptr.cast<ffi.Uint16>().asTypedList(size).buffer.asUint8List(),
      StbiDType.f32 => ptr.cast<ffi.Float>().asTypedList(size).buffer.asUint8List(),
    };
    return p;
  }

  List<num> get values {
    final size = _width * _height * _channels;
    return switch (dtype) {
      StbiDType.u8 => ptr.cast<ffi.Uint8>().asTypedList(size),
      StbiDType.u16 => ptr.cast<ffi.Uint16>().asTypedList(size),
      StbiDType.f32 => ptr.cast<ffi.Float>().asTypedList(size),
    };
  }

  List<num> row(int y) {
    final size = _width * _channels;
    return switch (dtype) {
      StbiDType.u8 => (ptr.cast<ffi.Uint8>() + y * size).asTypedList(size),
      StbiDType.u16 => (ptr.cast<ffi.Uint16>() + y * size).asTypedList(size),
      StbiDType.f32 => (ptr.cast<ffi.Float>() + y * size).asTypedList(size),
    };
  }

  List<num> col(int x) {
    final size = _height * _channels;
    final result = List<num>.filled(size, 0);
    final stride = _width * _channels;

    for (var y = 0; y < _height; y++) {
      for (var c = 0; c < _channels; c++) {
        final srcIdx = y * stride + x * _channels + c;
        final dstIdx = y * _channels + c;
        result[dstIdx] = switch (dtype) {
          StbiDType.u8 => ptr.cast<ffi.Uint8>()[srcIdx],
          StbiDType.u16 => ptr.cast<ffi.Uint16>()[srcIdx],
          StbiDType.f32 => ptr.cast<ffi.Float>()[srcIdx],
        };
      }
    }
    return result;
  }

  List<num> pixel(int x, int y) {
    final stride = _width * _channels;
    final offset = y * stride + x * _channels;
    return switch (dtype) {
      StbiDType.u8 => (ptr.cast<ffi.Uint8>() + offset).asTypedList(_channels),
      StbiDType.u16 => (ptr.cast<ffi.Uint16>() + offset).asTypedList(_channels),
      StbiDType.f32 => (ptr.cast<ffi.Float>() + offset).asTypedList(_channels),
    };
  }

  void forEachPixel(void Function(int x, int y, List<num> pixel) fn) {
    for (var y = 0; y < _height; y++) {
      for (var x = 0; x < _width; x++) {
        fn(x, y, pixel(x, y));
      }
    }
  }

  Image resize(
    int newWidth,
    int newHeight, {
    c.StbirPixelLayout pixelLayout = c.StbirPixelLayout.STBIR_RGB,
    c.StbirDataType dtype = c.StbirDataType.STBIR_TYPE_UINT8,
    c.StbirEdge edge = c.StbirEdge.STBIR_EDGE_CLAMP,
    c.StbirFilter filter = c.StbirFilter.STBIR_FILTER_DEFAULT,
    int? inputStride,
    int? outputStride,
  }) {
    inputStride ??= 0;
    outputStride ??= 0;
    final pOutPixels = c.stbir_resize(
      ptr,
      _width,
      _height,
      inputStride,
      ffi.nullptr,
      newWidth,
      newHeight,
      outputStride,
      pixelLayout,
      dtype,
      edge,
      filter,
    );
    if (pOutPixels == ffi.nullptr) {
      final pReason = c.stbi_failure_reason();
      throw MNNException("Failed to resize image: ${pReason.cast<Utf8>().toDartString()}");
    }
    return Image.fromPointer(
      pOutPixels.cast(),
      width: newWidth,
      height: newHeight,
      channels: _channels,
      desiredChannels: _desiredChannels,
      dtype: StbiDType.fromStbirDataType(dtype),
    );
  }

  /// Convert RGBRGBRGB to RRRGGGBBB, will copy data
  Image toPlanar() {
    final frameSize = _width * _height;
    switch (dtype) {
      case StbiDType.u8:
        final temp = bytes;
        final p = malloc<ffi.Uint8>(elemCount);
        for (var i = 0; i < frameSize; i++) {
          p[i] = temp[i * 3];
          p[i + frameSize] = temp[i * 3 + 1];
          p[i + frameSize * 2] = temp[i * 3 + 2];
        }
        return Image.fromPointer(
          p.cast(),
          width: _width,
          height: _height,
          channels: _channels,
          desiredChannels: _desiredChannels,
          dtype: StbiDType.u8,
        );
      case StbiDType.u16:
        final temp = bytes.buffer.asUint16List();
        final p = malloc<ffi.Uint16>(elemCount);
        for (var i = 0; i < frameSize; i++) {
          p[i] = temp[i * 3];
          p[i + frameSize] = temp[i * 3 + 1];
          p[i + frameSize * 2] = temp[i * 3 + 2];
        }
        return Image.fromPointer(
          p.cast(),
          width: _width,
          height: _height,
          channels: _channels,
          desiredChannels: _desiredChannels,
          dtype: StbiDType.u16,
        );
      case StbiDType.f32:
        final temp = bytes.buffer.asFloat32List();
        final p = malloc<ffi.Float>(elemCount);
        for (var i = 0; i < frameSize; i++) {
          p[i] = temp[i * 3];
          p[i + frameSize] = temp[i * 3 + 1];
          p[i + frameSize * 2] = temp[i * 3 + 2];
        }
        return Image.fromPointer(
          p.cast(),
          width: _width,
          height: _height,
          channels: _channels,
          desiredChannels: _desiredChannels,
          dtype: StbiDType.f32,
        );
    }
  }

  static bool haveReader(String path) => info(path).$1 != 0;
  static bool haveReaderFromMemory(Uint8List bytes) => infoFromMemory(bytes).$1 != 0;

  int save(String path, {ImWriteFormats? format, Map<ImWriteFlags, int> params = const {}}) {
    final cPath = path.toNativeUtf8().cast<ffi.Char>();
    final pathExt = path.split('.').last;
    format ??= ImWriteFormats.fromValue(pathExt);

    int rval = -1;
    switch (format) {
      case ImWriteFormats.PNG:
        rval = c.stbi_write_png(cPath, _width, _height, _channels, ptr, 0);
      case ImWriteFormats.JPEG || ImWriteFormats.JPG:
        rval = c.stbi_write_jpg(
          cPath,
          _width,
          _height,
          _channels,
          ptr,
          params[ImWriteFlags.IMWRITE_JPEG_QUALITY] ?? 95,
        );
      case ImWriteFormats.BMP:
        rval = c.stbi_write_bmp(cPath, _width, _height, _channels, ptr);
      case ImWriteFormats.TGA:
        rval = c.stbi_write_tga(cPath, _width, _height, _channels, ptr);
      case ImWriteFormats.HDR:
        rval = c.stbi_write_hdr(cPath, _width, _height, _channels, ptr.cast());
    }
    calloc.free(cPath);
    return rval;
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    c.stbi_image_free(ptr);
  }
}

void setUnpremultiplyOnLoad({bool flagTrueIfShouldUnpremultiply = true}) =>
    c.stbi_set_unpremultiply_on_load(flagTrueIfShouldUnpremultiply ? 1 : 0);

void convertIphonePngToRgb({bool flagTrueIfShouldConvert = true}) =>
    c.stbi_convert_iphone_png_to_rgb(flagTrueIfShouldConvert ? 1 : 0);

void setFlipVerticallyOnLoad({bool flagTrueIfShouldFlip = true}) =>
    c.stbi_set_flip_vertically_on_load(flagTrueIfShouldFlip ? 1 : 0);

void hdrToLdrGamma(double gamma) => c.stbi_hdr_to_ldr_gamma(gamma);

void hdrToLdrScale(double scale) => c.stbi_hdr_to_ldr_scale(scale);

void ldrToHdrGamma(double gamma) => c.stbi_ldr_to_hdr_gamma(gamma);
void ldrToHdrScale(double scale) => c.stbi_ldr_to_hdr_scale(scale);
