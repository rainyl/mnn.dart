/*
 * tensor.cpp
 * MNN C API for Tensor implementation
 *
 * This file implements the C-style API for MNN's Tensor functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#include "tensor.h"
#include "MNN/HalideRuntime.h"
#include "MNN/Tensor.hpp"
#include "error_code.h"
#include "mnn_type.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

enum DataType {
  DataType_DT_INVALID = 0,
  DataType_DT_FLOAT = 1,
  DataType_DT_DOUBLE = 2,
  DataType_DT_INT32 = 3,
  DataType_DT_UINT8 = 4,
  DataType_DT_INT16 = 5,
  DataType_DT_INT8 = 6,
  DataType_DT_STRING = 7,
  DataType_DT_COMPLEX64 = 8,
  DataType_DT_INT64 = 9,
  DataType_DT_BOOL = 10,
  DataType_DT_QINT8 = 11,
  DataType_DT_QUINT8 = 12,
  DataType_DT_QINT32 = 13,
  DataType_DT_BFLOAT16 = 14,
  DataType_DT_QINT16 = 15,
  DataType_DT_QUINT16 = 16,
  DataType_DT_UINT16 = 17,
  DataType_DT_COMPLEX128 = 18,
  DataType_DT_HALF = 19,
  DataType_DT_RESOURCE = 20,
  DataType_DT_VARIANT = 21,
  DataType_MIN = DataType_DT_INVALID,
  DataType_MAX = DataType_DT_VARIANT
};

halide_type_t mnn_halide_type_from_tensor_data_type(int type) {
  switch (type) {
  case DataType_DT_DOUBLE:
  case DataType_DT_FLOAT: return halide_type_of<float>(); break;
  case DataType_DT_BFLOAT16: return halide_type_t(halide_type_bfloat, 16); break;
  case DataType_DT_QINT32:
  case DataType_DT_INT32:
  case DataType_DT_BOOL:
  case DataType_DT_INT64: return halide_type_of<int32_t>(); break;
  case DataType_DT_QINT8:
  case DataType_DT_INT8: return halide_type_of<int8_t>(); break;
  case DataType_DT_QUINT8:
  case DataType_DT_UINT8: return halide_type_of<uint8_t>(); break;
  case DataType_DT_QUINT16:
  case DataType_DT_UINT16: return halide_type_of<uint16_t>(); break;
  case DataType_DT_QINT16:
  case DataType_DT_INT16: return halide_type_of<int16_t>(); break;
  default:
    MNN_PRINT("Unsupported data type! %d\n", type);
    MNN_ASSERT(false);
    return halide_type_of<float>(); // Return a default type for error handling
    break;
  }
}

int mnn_tensor_data_type_from_halide_type(halide_type_c_t type) {
  switch (type.code) {
  case halide_type_code_t::halide_type_uint:
    switch (type.bits) {
    case 8: return DataType_DT_UINT8; break;
    case 16: return DataType_DT_UINT16; break;
    default: return DataType_DT_INVALID; break;
    }
  case halide_type_code_t::halide_type_int:
    switch (type.bits) {
    case 8: return DataType_DT_INT8; break;
    case 16: return DataType_DT_INT16; break;
    case 32: return DataType_DT_INT32; break;
    case 64: return DataType_DT_INT64; break;
    default: return DataType_DT_INVALID; break;
    }
  case halide_type_code_t::halide_type_float:
    switch (type.bits) {
    case 32: return DataType_DT_FLOAT; break;
    case 64: return DataType_DT_DOUBLE; break;
    default: return DataType_DT_INVALID; break;
    }
  case halide_type_code_t::halide_type_bfloat:
    switch (type.bits) {
    case 16: return DataType_DT_BFLOAT16; break;
    default: return DataType_DT_INVALID; break;
    }
  default: return DataType_DT_INVALID; break;
  }
}

// Tensor creation functions
mnn_tensor_t mnn_tensor_create(int dim_size, mnn_dimension_type_t type) {
  try {
    MNN::Tensor *self = new MNN::Tensor(dim_size, (MNN::Tensor::DimensionType)type);
    return (mnn_tensor_t)self;
  } catch (...) { return nullptr; }
}

mnn_tensor_t
mnn_tensor_create_from_tensor(mnn_tensor_t self, mnn_dimension_type_t type, bool alloc_memory) {
  if (!self) return nullptr;
  try {
    MNN::Tensor *src = (MNN::Tensor *)self;
    MNN::Tensor *newTensor = new MNN::Tensor(src, (MNN::Tensor::DimensionType)type, alloc_memory);
    return (mnn_tensor_t)newTensor;
  } catch (...) { return nullptr; }
}

mnn_tensor_t mnn_tensor_create_device(
    const int *shape, int shape_size, halide_type_c_t type, mnn_dimension_type_t dim_type
) {
  if (!shape || shape_size <= 0) return nullptr;
  try {
    const auto _shape = std::vector<int>(shape, shape + shape_size);
    const auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
    MNN::Tensor *tensor =
        MNN::Tensor::createDevice(_shape, _type, (MNN::Tensor::DimensionType)dim_type);
    return (mnn_tensor_t)tensor;
  } catch (...) { return nullptr; }
}

mnn_tensor_t mnn_tensor_create_with_data(
    const int *shape,
    int shape_size,
    halide_type_c_t type,
    void *data,
    mnn_dimension_type_t dim_type
) {
  if (!shape || shape_size <= 0 || !data) return nullptr;
  try {
    auto _shape = std::vector<int>(shape, shape + shape_size);
    const auto _type = halide_type_t((halide_type_code_t)type.code, type.bits, type.lanes);
    MNN::Tensor *tensor =
        MNN::Tensor::create(_shape, _type, data, (MNN::Tensor::DimensionType)dim_type);
    return (mnn_tensor_t)tensor;
  } catch (...) { return nullptr; }
}

// MNN::Tensor destruction
void mnn_tensor_destroy(mnn_tensor_t self) {
  if (self) { MNN::Tensor::destroy((MNN::Tensor *)self); }
}

// MNN::Tensor operations
mnn_tensor_t mnn_tensor_clone(mnn_tensor_t src, bool deep_copy) {
  if (!src) return nullptr;
  try {
    MNN::Tensor *clone = MNN::Tensor::clone((MNN::Tensor *)src, deep_copy);
    return (mnn_tensor_t)clone;
  } catch (...) { return nullptr; }
}

mnn_error_code_t mnn_tensor_copy_from_host(mnn_tensor_t self, mnn_tensor_t host_tensor) {
  if (!self || !host_tensor) return INVALID_VALUE;
  try {
    MNN::Tensor *dst = (MNN::Tensor *)self;
    MNN::Tensor *src = (MNN::Tensor *)host_tensor;
    auto r = dst->copyFromHostTensor(src);
    return r ? BOOL_TRUE : BOOL_FALSE;
  } catch (...) { return UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_tensor_copy_to_host(mnn_tensor_t self, mnn_tensor_t host_tensor) {
  if (!self || !host_tensor) return INVALID_VALUE;
  try {
    MNN::Tensor *src = (MNN::Tensor *)self;
    MNN::Tensor *dst = (MNN::Tensor *)host_tensor;
    auto r = src->copyToHostTensor(dst);
    return r ? BOOL_TRUE : BOOL_FALSE;
  } catch (...) { return UNKNOWN_ERROR; }
}

// MNN::Tensor properties
int mnn_tensor_dimensions(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->dimensions();
}

mnn_error_code_t mnn_tensor_shape(mnn_tensor_t self, int *shape, int shape_size) {
  if (!self || !shape) return INVALID_VALUE;
  MNN::Tensor *t = (MNN::Tensor *)self;
  if (shape_size < t->dimensions()) return INVALID_VALUE;

  for (int i = 0; i < t->dimensions(); i++) { shape[i] = t->length(i); }
  return NO_ERROR;
}

int mnn_tensor_size(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->size();
}

int mnn_tensor_element_size(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->elementSize();
}

size_t mnn_tensor_usize(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->usize();
}

// Dimension-specific properties
int mnn_tensor_width(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->width();
}

int mnn_tensor_height(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->height();
}

int mnn_tensor_channel(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->channel();
}

int mnn_tensor_batch(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->batch();
}

int mnn_tensor_stride(mnn_tensor_t self, int index) {
  if (!self || index < 0) return -1;
  return ((MNN::Tensor *)self)->stride(index);
}

int mnn_tensor_length(mnn_tensor_t self, int index) {
  if (!self || index < 0) return -1;
  return ((MNN::Tensor *)self)->length(index);
}

void mnn_tensor_set_stride(mnn_tensor_t self, int index, int stride) {
  if (self && index >= 0) { ((MNN::Tensor *)self)->setStride(index, stride); }
}

void mnn_tensor_set_length(mnn_tensor_t self, int index, int length) {
  if (self && index >= 0) { ((MNN::Tensor *)self)->setLength(index, length); }
}

// Data access
void *mnn_tensor_host(mnn_tensor_t self) {
  if (!self) return nullptr;
  return ((MNN::Tensor *)self)->host<void>();
}

uint64_t mnn_tensor_device_id(mnn_tensor_t self) {
  if (!self) return -1;
  return ((MNN::Tensor *)self)->deviceId();
}

halide_buffer_c_t *mnn_tensor_buffer(mnn_tensor_t self) {
  if (!self) return nullptr;
  auto _buf = ((MNN::Tensor *)self)->buffer();
  halide_buffer_c_t *buf = (halide_buffer_c_t *)malloc(sizeof(halide_buffer_c_t));
  buf->device = _buf.device;
  buf->device_interface = _buf.device_interface;
  buf->host = _buf.host;
  buf->flags = _buf.flags;
  buf->type = {(uint8_t)_buf.type.code, _buf.type.bits, _buf.type.lanes};
  buf->dimensions = _buf.dimensions;
  buf->dim = _buf.dim;
  buf->padding = _buf.padding;
  return buf;
}

// Type information
mnn_dimension_type_t mnn_tensor_get_dimension_type(mnn_tensor_t self) {
  if (!self) return MNN_TENSORFLOW;
  return (mnn_dimension_type_t)((MNN::Tensor *)self)->getDimensionType();
}

mnn_handle_data_type_t mnn_tensor_get_handle_data_type(mnn_tensor_t self) {
  if (!self) return MNN_HANDLE_NONE;
  return (mnn_handle_data_type_t)((MNN::Tensor *)self)->getHandleDataType();
}

void mnn_tensor_set_type(mnn_tensor_t self, int type) {
  if (self) { ((MNN::Tensor *)self)->setType(type); }
}

halide_type_c_t *mnn_tensor_get_type(mnn_tensor_t self) {
  if (!self) return nullptr;
  auto _type = ((MNN::Tensor *)self)->getType();
  halide_type_c_t *type = (halide_type_c_t *)malloc(sizeof(halide_type_c_t));
  type->code = (uint8_t)_type.code;
  type->bits = _type.bits;
  type->lanes = _type.lanes;
  return type;
}

// Memory mapping
void *mnn_tensor_map(mnn_tensor_t self, mnn_map_type_t mtype, mnn_dimension_type_t dtype) {
  if (!self) return nullptr;
  return ((MNN::Tensor *)self)->map((MNN::Tensor::MapType)mtype, (MNN::Tensor::DimensionType)dtype);
}

void mnn_tensor_unmap(
    mnn_tensor_t self, mnn_map_type_t mtype, mnn_dimension_type_t dtype, void *map_ptr
) {
  if (self && map_ptr) {
    ((MNN::Tensor *)self)
        ->unmap((MNN::Tensor::MapType)mtype, (MNN::Tensor::DimensionType)dtype, map_ptr);
  }
}

mnn_error_code_t mnn_tensor_wait(mnn_tensor_t self, mnn_map_type_t mtype, bool finish) {
  if (!self) return MNN_INVALID_PTR;
  try {
    ((MNN::Tensor *)self)->wait((MNN::Tensor::MapType)mtype, finish);
    return NO_ERROR;
  } catch (...) { return UNKNOWN_ERROR; }
}

mnn_error_code_t
mnn_tensor_set_device_ptr(mnn_tensor_t self, const void *device_ptr, int memory_type) {
  if (!self || !device_ptr) return MNN_INVALID_PTR;
  try {
    ((MNN::Tensor *)self)->setDevicePtr(device_ptr, memory_type);
    return NO_ERROR;
  } catch (...) { return UNKNOWN_ERROR; }
}

void mnn_tensor_print(mnn_tensor_t self) {
  if (!self) return;
  ((MNN::Tensor *)self)->print();
}

void mnn_tensor_print_shape(mnn_tensor_t self) {
  if (!self) return;
  ((MNN::Tensor *)self)->printShape();
}

MNN_C_API mnn_error_code_t mnn_tensor_set_image_f32(
    mnn_tensor_t self, int index, float *data, int width, int height, int channel
) {
  if (!self || !data) return MNN_INVALID_PTR;
  if (self->width() != width || self->height() != height || self->channel() != channel)
    return INVALID_VALUE;
  auto host = self->host<float>() + index * width * height * channel;
  memcpy(host, data, width * height * channel * sizeof(float));
  return NO_ERROR;
}
