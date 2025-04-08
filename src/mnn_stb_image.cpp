#include "mnn_stb_image.h"

int mnn_stbi_rgb_flatten_uint8(stbi_uc *data, int width, int height, int channels) {
  if (!data) return stbi__err("null pointer", "mnn_stbi_rgb_flatten_uint8: data is null");
  const int pixel_count = width * height;
  stbi_uc *temp = new stbi_uc[pixel_count * channels];

  // Copy each channel
  for (int c = 0; c < channels; ++c) {
    for (int i = 0; i < pixel_count; ++i) { temp[c * pixel_count + i] = data[i * channels + c]; }
  }

  // Copy back to original data
  memcpy(data, temp, pixel_count * channels);
  delete[] temp;
  return 1;
}

int mnn_stbi_add_f32(stbi_uc *data, int width, int height, int channels, float *value) {
  if (!data || !value)
    return stbi__err("null pointer", "mnn_stbi_subtract_f32: data or value is null");
  auto ptr = reinterpret_cast<float *>(data);

  for (int i = 0; i < width * height; ++i) {
    for (int c = 0; c < channels; ++c) { ptr[i * channels + c] += value[c]; }
  }
  return 1;
}

int mnn_stbi_sub_f32(stbi_uc *data, int width, int height, int channels, float *value) {
  if (!data || !value)
    return stbi__err("null pointer", "mnn_stbi_subtract_f32: data or value is null");
  auto ptr = reinterpret_cast<float *>(data);

  for (int i = 0; i < width * height; ++i) {
    for (int c = 0; c < channels; ++c) { ptr[i * channels + c] -= value[c]; }
  }
  return 1;
}

int mnn_stbi_div_f32(stbi_uc *data, int width, int height, int channels, float *value) {
  if (!data || !value)
    return stbi__err("null pointer", "mnn_stbi_divide_f32: data or value is null");
  for (int i = 0; i < channels; i++) {
    if (value[i] == 0.0f) {
      return stbi__err("divide by zero", "mnn_stbi_divide_f32: divide by zero");
    }
  }
  auto ptr = reinterpret_cast<float *>(data);

  for (int i = 0; i < width * height; ++i) {
    for (int c = 0; c < channels; ++c) { ptr[i * channels + c] /= value[c]; }
  }
  return 1;
}

int mnn_stbi_mul_f32(stbi_uc *data, int width, int height, int channels, float *value) {
  if (!data || !value)
    return stbi__err("null pointer", "mnn_stbi_multiply_f32: data or value is null");
  auto ptr = reinterpret_cast<float *>(data);
  for (int i = 0; i < width * height; ++i) {
    for (int c = 0; c < channels; ++c) { ptr[i * channels + c] *= value[c]; }
  }
  return 1;
}
