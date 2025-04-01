//
// tensor.cpp
// MNN C API for Tensor implementation
//
// This file implements the C-style API for MNN's Tensor functionality
// All functions use snake_case naming convention
//

#include "tensor.h"
#include "MNN/HalideRuntime.h"
#include "MNN/Tensor.hpp"
#include "error_code.h"
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

struct halide_type_t *mnn_halide_type_create() { return new halide_type_t(); }

struct halide_type_t *
mnn_halide_type_create_1(halide_type_code_t code, uint8_t bits, uint16_t lanes) {
  return new halide_type_t(code, bits, lanes);
}

void mnn_halide_type_destroy(struct halide_type_t *self) {
  if (self) { delete self; }
}

// Tensor creation functions
mnn_tensor_t mnn_tensor_create(int dim_size, mnn_dimension_type_t type) {
  try {
    MNN::Tensor *tensor = new MNN::Tensor(dim_size, (MNN::Tensor::DimensionType)type);
    return (mnn_tensor_t)tensor;
  } catch (...) { return nullptr; }
}

mnn_tensor_t
mnn_tensor_create_from_tensor(mnn_tensor_t tensor, mnn_dimension_type_t type, bool alloc_memory) {
  if (!tensor) return nullptr;
  try {
    MNN::Tensor *src = (MNN::Tensor *)tensor;
    MNN::Tensor *newTensor = new MNN::Tensor(src, (MNN::Tensor::DimensionType)type, alloc_memory);
    return (mnn_tensor_t)newTensor;
  } catch (...) { return nullptr; }
}

mnn_tensor_t mnn_tensor_create_device(
    const int *shape, int shape_size, halide_type_t type, mnn_dimension_type_t dim_type
) {
  if (!shape || shape_size <= 0) return nullptr;
  try {
    const auto _shape = std::vector<int>(shape, shape + shape_size);
    MNN::Tensor *tensor =
        MNN::Tensor::createDevice(_shape, type, (MNN::Tensor::DimensionType)dim_type);
    return (mnn_tensor_t)tensor;
  } catch (...) { return nullptr; }
}

mnn_tensor_t mnn_tensor_create_with_data(
    const int *shape,
    int shape_size,
    struct halide_type_t type,
    void *data,
    mnn_dimension_type_t dim_type
) {
  if (!shape || shape_size <= 0 || !data) return nullptr;
  try {
    auto _shape = std::vector<int>(shape, shape + shape_size);
    MNN::Tensor *tensor =
        MNN::Tensor::create(_shape, type, data, (MNN::Tensor::DimensionType)dim_type);
    return (mnn_tensor_t)tensor;
  } catch (...) { return nullptr; }
}

// MNN::Tensor destruction
void mnn_tensor_destroy(mnn_tensor_t tensor) {
  if (tensor) { delete (MNN::Tensor *)tensor; }
}

// MNN::Tensor operations
mnn_tensor_t mnn_tensor_clone(mnn_tensor_t src, bool deep_copy) {
  if (!src) return nullptr;
  try {
    MNN::Tensor *clone = MNN::Tensor::clone((MNN::Tensor *)src, deep_copy);
    return (mnn_tensor_t)clone;
  } catch (...) { return nullptr; }
}

mnn_error_code_t mnn_tensor_copy_from_host(mnn_tensor_t tensor, mnn_tensor_t host_tensor) {
  if (!tensor || !host_tensor) return INVALID_VALUE;
  try {
    MNN::Tensor *dst = (MNN::Tensor *)tensor;
    MNN::Tensor *src = (MNN::Tensor *)host_tensor;
    auto r = dst->copyFromHostTensor(src);
    return r ? BOOL_TRUE : BOOL_FALSE;
  } catch (...) { return UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_tensor_copy_to_host(mnn_tensor_t tensor, mnn_tensor_t host_tensor) {
  if (!tensor || !host_tensor) return INVALID_VALUE;
  try {
    MNN::Tensor *src = (MNN::Tensor *)tensor;
    MNN::Tensor *dst = (MNN::Tensor *)host_tensor;
    auto r = src->copyToHostTensor(dst);
    return r ? BOOL_TRUE : BOOL_FALSE;
  } catch (...) { return UNKNOWN_ERROR; }
}

// MNN::Tensor properties
int mnn_tensor_dimensions(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->dimensions();
}

mnn_error_code_t mnn_tensor_shape(mnn_tensor_t tensor, int *shape, int shape_size) {
  if (!tensor || !shape) return INVALID_VALUE;
  MNN::Tensor *t = (MNN::Tensor *)tensor;
  if (shape_size < t->dimensions()) return INVALID_VALUE;

  for (int i = 0; i < t->dimensions(); i++) { shape[i] = t->length(i); }
  return NO_ERROR;
}

int mnn_tensor_size(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->size();
}

int mnn_tensor_element_size(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->elementSize();
}

size_t mnn_tensor_usize(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->usize();
}

// Dimension-specific properties
int mnn_tensor_width(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->width();
}

int mnn_tensor_height(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->height();
}

int mnn_tensor_channel(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->channel();
}

int mnn_tensor_batch(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->batch();
}

int mnn_tensor_stride(mnn_tensor_t tensor, int index) {
  if (!tensor || index < 0) return 0;
  return ((MNN::Tensor *)tensor)->stride(index);
}

int mnn_tensor_length(mnn_tensor_t tensor, int index) {
  if (!tensor || index < 0) return 0;
  return ((MNN::Tensor *)tensor)->length(index);
}

void mnn_tensor_set_stride(mnn_tensor_t tensor, int index, int stride) {
  if (tensor && index >= 0) { ((MNN::Tensor *)tensor)->setStride(index, stride); }
}

void mnn_tensor_set_length(mnn_tensor_t tensor, int index, int length) {
  if (tensor && index >= 0) { ((MNN::Tensor *)tensor)->setLength(index, length); }
}

// Data access
void *mnn_tensor_host(mnn_tensor_t tensor) {
  if (!tensor) return nullptr;
  return ((MNN::Tensor *)tensor)->host<void>();
}

uint64_t mnn_tensor_device_id(mnn_tensor_t tensor) {
  if (!tensor) return 0;
  return ((MNN::Tensor *)tensor)->deviceId();
}

void *mnn_tensor_buffer(mnn_tensor_t tensor) {
  if (!tensor) return nullptr;
  return ((MNN::Tensor *)tensor)->buffer().host;
}

// Type information
mnn_dimension_type_t mnn_tensor_get_dimension_type(mnn_tensor_t tensor) {
  if (!tensor) return MNN_TENSORFLOW;
  return (mnn_dimension_type_t)((MNN::Tensor *)tensor)->getDimensionType();
}

mnn_handle_data_type_t mnn_tensor_get_handle_data_type(mnn_tensor_t tensor) {
  if (!tensor) return MNN_HANDLE_NONE;
  return (mnn_handle_data_type_t)((MNN::Tensor *)tensor)->getHandleDataType();
}

void mnn_tensor_set_type(mnn_tensor_t tensor, int type) {
  if (tensor) { ((MNN::Tensor *)tensor)->setType(type); }
}

struct halide_type_t *mnn_tensor_get_type(mnn_tensor_t tensor) {
  if (!tensor) return new halide_type_t();
  auto _type = ((MNN::Tensor *)tensor)->getType();
  return new halide_type_t(_type.code, _type.bits, _type.lanes);
}

// Memory mapping
void *mnn_tensor_map(mnn_tensor_t tensor, mnn_map_type_t mtype, mnn_dimension_type_t dtype) {
  if (!tensor) return nullptr;
  return ((MNN::Tensor *)tensor)
      ->map((MNN::Tensor::MapType)mtype, (MNN::Tensor::DimensionType)dtype);
}

void mnn_tensor_unmap(
    mnn_tensor_t tensor, mnn_map_type_t mtype, mnn_dimension_type_t dtype, void *map_ptr
) {
  if (tensor && map_ptr) {
    ((MNN::Tensor *)tensor)
        ->unmap((MNN::Tensor::MapType)mtype, (MNN::Tensor::DimensionType)dtype, map_ptr);
  }
}

mnn_error_code_t mnn_tensor_wait(mnn_tensor_t tensor, mnn_map_type_t mtype, bool finish) {
  if (!tensor) return MNN_INVALID_PTR;
  try {
    ((MNN::Tensor *)tensor)->wait((MNN::Tensor::MapType)mtype, finish);
    return NO_ERROR;
  } catch (...) { return UNKNOWN_ERROR; }
}

mnn_error_code_t
mnn_tensor_set_device_ptr(mnn_tensor_t tensor, const void *device_ptr, int memory_type) {
  if (!tensor || !device_ptr) return MNN_INVALID_PTR;
  try {
    ((MNN::Tensor *)tensor)->setDevicePtr(device_ptr, memory_type);
    return NO_ERROR;
  } catch (...) { return UNKNOWN_ERROR; }
}
