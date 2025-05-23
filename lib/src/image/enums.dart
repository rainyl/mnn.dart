import '../g/mnn.g.dart' as c;

enum ImreadModes {
  /// uint8_t gray
  IMREAD_GRAYSCALE(0),

  /// uint8_t bgr
  IMREAD_COLOR(1),

  /// float bgr
  IMREAD_ANYDEPTH(4);

  final int value;
  const ImreadModes(this.value);
  static ImreadModes fromValue(int value) => switch (value) {
        0 => IMREAD_GRAYSCALE,
        1 => IMREAD_COLOR,
        4 => IMREAD_ANYDEPTH,
        _ => throw ArgumentError("Invalid value: $value"),
      };
}

enum ImageFormat {
  RGBA(0),
  RGB(1),
  BGR(2),
  GRAY(3),
  BGRA(4),
  YCrCb(5),
  YUV(6),
  HSV(7),
  XYZ(8),
  BGR555(9),
  BGR565(10),
  YUV_NV21(11),
  YUV_NV12(12),
  YUV_I420(13),
  HSV_FULL(14);

  final int value;
  const ImageFormat(this.value);
  static ImageFormat fromValue(int value) => switch (value) {
        0 => RGBA,
        1 => RGB,
        2 => BGR,
        3 => GRAY,
        4 => BGRA,
        5 => YCrCb,
        6 => YUV,
        7 => HSV,
        8 => XYZ,
        9 => BGR555,
        10 => BGR565,
        11 => YUV_NV21,
        12 => YUV_NV12,
        13 => YUV_I420,
        14 => HSV_FULL,
        _ => throw ArgumentError("Invalid value: $value"),
      };
}

enum ImWriteFormats {
  PNG("png"),
  BMP("bmp"),
  TGA("tga"),
  HDR("hdr"),
  JPG("jpg"),
  JPEG("jpeg");

  final String value;
  const ImWriteFormats(this.value);

  static ImWriteFormats fromValue(String value) => switch (value) {
        "png" => PNG,
        "bmp" => BMP,
        "tga" => TGA,
        "hdr" => HDR,
        "jpg" => JPG,
        "jpeg" => JPEG,
        _ => throw ArgumentError("Image format $value is not supported"),
      };
}

enum ImWriteFlags {
  /// jpg, default is 95
  IMWRITE_JPEG_QUALITY(1);

  final int value;
  const ImWriteFlags(this.value);
  static ImWriteFlags fromValue(int value) => switch (value) {
        1 => IMWRITE_JPEG_QUALITY,
        _ => throw ArgumentError("Invalid value: $value"),
      };
}

enum StbiChannel {
  default_(c.STBI_default),
  grey(c.STBI_grey),
  greyAlpha(c.STBI_grey_alpha),
  rgb(c.STBI_rgb),
  rgba(c.STBI_rgb_alpha);

  final int value;
  const StbiChannel(this.value);

  static StbiChannel fromValue(int value) => switch (value) {
        c.STBI_default => StbiChannel.default_,
        c.STBI_grey => StbiChannel.grey,
        c.STBI_grey_alpha => StbiChannel.greyAlpha,
        c.STBI_rgb => StbiChannel.rgb,
        c.STBI_rgb_alpha => StbiChannel.rgba,
        _ => throw ArgumentError("Invalid value: $value"),
      };
}

enum StbiDType {
  u8,
  u16,
  // f16,
  f32;

  static StbiDType fromStbirDataType(c.StbirDataType value) => switch (value) {
        c.StbirDataType.STBIR_TYPE_UINT8 ||
        c.StbirDataType.STBIR_TYPE_UINT8_SRGB ||
        c.StbirDataType.STBIR_TYPE_UINT8_SRGB_ALPHA =>
          StbiDType.u8,
        c.StbirDataType.STBIR_TYPE_UINT16 => StbiDType.u16,
        c.StbirDataType.STBIR_TYPE_FLOAT => StbiDType.f32,
        c.StbirDataType.STBIR_TYPE_HALF_FLOAT => throw UnimplementedError(),
      };
}

const int COLOR_BGR2BGRA = 0;
const int COLOR_RGB2RGBA = COLOR_BGR2BGRA;
const int COLOR_BGRA2BGR = 1;
const int COLOR_RGBA2RGB = COLOR_BGRA2BGR;
const int COLOR_BGR2RGBA = 2;
const int COLOR_RGB2BGRA = COLOR_BGR2RGBA;
const int COLOR_RGBA2BGR = 3;
const int COLOR_BGRA2RGB = COLOR_RGBA2BGR;
const int COLOR_BGR2RGB = 4;
const int COLOR_RGB2BGR = COLOR_BGR2RGB;
const int COLOR_BGRA2RGBA = 5;
const int COLOR_RGBA2BGRA = COLOR_BGRA2RGBA;
const int COLOR_BGR2GRAY = 6;
const int COLOR_RGB2GRAY = 7;
const int COLOR_GRAY2BGR = 8;
const int COLOR_GRAY2RGB = COLOR_GRAY2BGR;
const int COLOR_GRAY2BGRA = 9;
const int COLOR_GRAY2RGBA = COLOR_GRAY2BGRA;
const int COLOR_BGRA2GRAY = 10;
const int COLOR_RGBA2GRAY = 11;
const int COLOR_BGR2BGR565 = 12;
const int COLOR_RGB2BGR565 = 13;
const int COLOR_BGR5652BGR = 14;
const int COLOR_BGR5652RGB = 15;
const int COLOR_BGRA2BGR565 = 16;
const int COLOR_RGBA2BGR565 = 17;
const int COLOR_BGR5652BGRA = 18;
const int COLOR_BGR5652RGBA = 19;
const int COLOR_GRAY2BGR565 = 20;
const int COLOR_BGR5652GRAY = 21;
const int COLOR_BGR2BGR555 = 22;
const int COLOR_RGB2BGR555 = 23;
const int COLOR_BGR5552BGR = 24;
const int COLOR_BGR5552RGB = 25;
const int COLOR_BGRA2BGR555 = 26;
const int COLOR_RGBA2BGR555 = 27;
const int COLOR_BGR5552BGRA = 28;
const int COLOR_BGR5552RGBA = 29;
const int COLOR_GRAY2BGR555 = 30;
const int COLOR_BGR5552GRAY = 31;
const int COLOR_BGR2XYZ = 32;
const int COLOR_RGB2XYZ = 33;
const int COLOR_XYZ2BGR = 34;
const int COLOR_XYZ2RGB = 35;
const int COLOR_BGR2YCrCb = 36;
const int COLOR_RGB2YCrCb = 37;
const int COLOR_YCrCb2BGR = 38;
const int COLOR_YCrCb2RGB = 39;
const int COLOR_BGR2HSV = 40;
const int COLOR_RGB2HSV = 41;
const int COLOR_BGR2Lab = 44;
const int COLOR_RGB2Lab = 45;
const int COLOR_BGR2Luv = 50;
const int COLOR_RGB2Luv = 51;
const int COLOR_BGR2HLS = 52;
const int COLOR_RGB2HLS = 53;
const int COLOR_HSV2BGR = 54;
const int COLOR_HSV2RGB = 55;
const int COLOR_Lab2BGR = 56;
const int COLOR_Lab2RGB = 57;
const int COLOR_Luv2BGR = 58;
const int COLOR_Luv2RGB = 59;
const int COLOR_HLS2BGR = 60;
const int COLOR_HLS2RGB = 61;
const int COLOR_BGR2HSV_FULL = 66;
const int COLOR_RGB2HSV_FULL = 67;
const int COLOR_BGR2HLS_FULL = 68;
const int COLOR_RGB2HLS_FULL = 69;
const int COLOR_HSV2BGR_FULL = 70;
const int COLOR_HSV2RGB_FULL = 71;
const int COLOR_HLS2BGR_FULL = 72;
const int COLOR_HLS2RGB_FULL = 73;
const int COLOR_LBGR2Lab = 74;
const int COLOR_LRGB2Lab = 75;
const int COLOR_LBGR2Luv = 76;
const int COLOR_LRGB2Luv = 77;
const int COLOR_Lab2LBGR = 78;
const int COLOR_Lab2LRGB = 79;
const int COLOR_Luv2LBGR = 80;
const int COLOR_Luv2LRGB = 81;
const int COLOR_BGR2YUV = 82;
const int COLOR_RGB2YUV = 83;
const int COLOR_YUV2BGR = 84;
const int COLOR_YUV2RGB = 85;
const int COLOR_YUV2RGB_NV12 = 90;
const int COLOR_YUV2BGR_NV12 = 91;
const int COLOR_YUV2RGB_NV21 = 92;
const int COLOR_YUV2BGR_NV21 = 93;
const int COLOR_YUV420sp2RGB = COLOR_YUV2RGB_NV21;
const int COLOR_YUV420sp2BGR = COLOR_YUV2BGR_NV21;
const int COLOR_YUV2RGBA_NV12 = 94;
const int COLOR_YUV2BGRA_NV12 = 95;
const int COLOR_YUV2RGBA_NV21 = 96;
const int COLOR_YUV2BGRA_NV21 = 97;
const int COLOR_YUV420sp2RGBA = COLOR_YUV2RGBA_NV21;
const int COLOR_YUV420sp2BGRA = COLOR_YUV2BGRA_NV21;
const int COLOR_YUV2RGB_YV12 = 98;
const int COLOR_YUV2BGR_YV12 = 99;
const int COLOR_YUV2RGB_IYUV = 100;
const int COLOR_YUV2BGR_IYUV = 101;
const int COLOR_YUV2RGB_I420 = COLOR_YUV2RGB_IYUV;
const int COLOR_YUV2BGR_I420 = COLOR_YUV2BGR_IYUV;
const int COLOR_YUV420p2RGB = COLOR_YUV2RGB_YV12;
const int COLOR_YUV420p2BGR = COLOR_YUV2BGR_YV12;
const int COLOR_YUV2RGBA_YV12 = 102;
const int COLOR_YUV2BGRA_YV12 = 103;
const int COLOR_YUV2RGBA_IYUV = 104;
const int COLOR_YUV2BGRA_IYUV = 105;
const int COLOR_YUV2RGBA_I420 = COLOR_YUV2RGBA_IYUV;
const int COLOR_YUV2BGRA_I420 = COLOR_YUV2BGRA_IYUV;
const int COLOR_YUV420p2RGBA = COLOR_YUV2RGBA_YV12;
const int COLOR_YUV420p2BGRA = COLOR_YUV2BGRA_YV12;
const int COLOR_YUV2GRAY_420 = 106;
const int COLOR_YUV2GRAY_NV21 = COLOR_YUV2GRAY_420;
const int COLOR_YUV2GRAY_NV12 = COLOR_YUV2GRAY_420;
const int COLOR_YUV2GRAY_YV12 = COLOR_YUV2GRAY_420;
const int COLOR_YUV2GRAY_IYUV = COLOR_YUV2GRAY_420;
const int COLOR_YUV2GRAY_I420 = COLOR_YUV2GRAY_420;
const int COLOR_YUV420sp2GRAY = COLOR_YUV2GRAY_420;
const int COLOR_YUV420p2GRAY = COLOR_YUV2GRAY_420;
const int COLOR_YUV2RGB_UYVY = 107;
const int COLOR_YUV2BGR_UYVY = 108;
const int COLOR_YUV2RGB_Y422 = COLOR_YUV2RGB_UYVY;
const int COLOR_YUV2BGR_Y422 = COLOR_YUV2BGR_UYVY;
const int COLOR_YUV2RGB_UYNV = COLOR_YUV2RGB_UYVY;
const int COLOR_YUV2BGR_UYNV = COLOR_YUV2BGR_UYVY;
const int COLOR_YUV2RGBA_UYVY = 111;
const int COLOR_YUV2BGRA_UYVY = 112;
const int COLOR_YUV2RGBA_Y422 = COLOR_YUV2RGBA_UYVY;
const int COLOR_YUV2BGRA_Y422 = COLOR_YUV2BGRA_UYVY;
const int COLOR_YUV2RGBA_UYNV = COLOR_YUV2RGBA_UYVY;
const int COLOR_YUV2BGRA_UYNV = COLOR_YUV2BGRA_UYVY;
const int COLOR_YUV2RGB_YUY2 = 115;
const int COLOR_YUV2BGR_YUY2 = 116;
const int COLOR_YUV2RGB_YVYU = 117;
const int COLOR_YUV2BGR_YVYU = 118;
const int COLOR_YUV2RGB_YUYV = COLOR_YUV2RGB_YUY2;
const int COLOR_YUV2BGR_YUYV = COLOR_YUV2BGR_YUY2;
const int COLOR_YUV2RGB_YUNV = COLOR_YUV2RGB_YUY2;
const int COLOR_YUV2BGR_YUNV = COLOR_YUV2BGR_YUY2;
const int COLOR_YUV2RGBA_YUY2 = 119;
const int COLOR_YUV2BGRA_YUY2 = 120;
const int COLOR_YUV2RGBA_YVYU = 121;
const int COLOR_YUV2BGRA_YVYU = 122;
const int COLOR_YUV2RGBA_YUYV = COLOR_YUV2RGBA_YUY2;
const int COLOR_YUV2BGRA_YUYV = COLOR_YUV2BGRA_YUY2;
const int COLOR_YUV2RGBA_YUNV = COLOR_YUV2RGBA_YUY2;
const int COLOR_YUV2BGRA_YUNV = COLOR_YUV2BGRA_YUY2;
const int COLOR_YUV2GRAY_UYVY = 123;
const int COLOR_YUV2GRAY_YUY2 = 124;
const int COLOR_YUV2GRAY_Y422 = COLOR_YUV2GRAY_UYVY;
const int COLOR_YUV2GRAY_UYNV = COLOR_YUV2GRAY_UYVY;
const int COLOR_YUV2GRAY_YVYU = COLOR_YUV2GRAY_YUY2;
const int COLOR_YUV2GRAY_YUYV = COLOR_YUV2GRAY_YUY2;
const int COLOR_YUV2GRAY_YUNV = COLOR_YUV2GRAY_YUY2;
const int COLOR_RGBA2mRGBA = 125;
const int COLOR_mRGBA2RGBA = 126;
const int COLOR_RGB2YUV_I420 = 127;
const int COLOR_BGR2YUV_I420 = 128;
const int COLOR_RGB2YUV_IYUV = COLOR_RGB2YUV_I420;
const int COLOR_BGR2YUV_IYUV = COLOR_BGR2YUV_I420;
const int COLOR_RGBA2YUV_I420 = 129;
const int COLOR_BGRA2YUV_I420 = 130;
const int COLOR_RGBA2YUV_IYUV = COLOR_RGBA2YUV_I420;
const int COLOR_BGRA2YUV_IYUV = COLOR_BGRA2YUV_I420;
const int COLOR_RGB2YUV_YV12 = 131;
const int COLOR_BGR2YUV_YV12 = 132;
const int COLOR_RGBA2YUV_YV12 = 133;
const int COLOR_BGRA2YUV_YV12 = 134;
const int COLOR_COLORCVT_MAX = 143;
