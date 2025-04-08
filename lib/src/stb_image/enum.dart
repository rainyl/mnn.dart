import '../g/stbi.g.dart' as c;

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
