/*
 * tensor.h
 * MNN C API for Tensor
 *
 * This file implements the C-style API for MNN's Tensor functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#ifndef MNN_TENSOR_H
#define MNN_TENSOR_H

#include "MNN/HalideRuntime.h"
#include "error_code.h"
#include "mnn_type.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include "MNN/Tensor.hpp"
extern "C" {
#endif

/** Opaque pointer types */
#ifdef __cplusplus
typedef MNN::Tensor *mnn_tensor_t;
#else
typedef void *mnn_tensor_t;
#endif

typedef enum { MNN_TENSORFLOW, MNN_CAFFE, MNN_CAFFE_C4 } mnn_dimension_type_t;

typedef enum { MNN_HANDLE_NONE = 0, MNN_HANDLE_STRING = 1 } mnn_handle_data_type_t;

typedef enum { MNN_MAP_TENSOR_WRITE = 0, MNN_MAP_TENSOR_READ = 1 } mnn_map_type_t;

typedef enum {
  MNN_T_D_TYPE_F32_F64 = 0,
  MNN_T_D_TYPE_BF16 = 1,
  MNN_T_D_TYPE_QI32_I32_BOOL_I64 = 2,
  MNN_T_D_TYPE_QI8_I8 = 3,
  MNN_T_D_TYPE_QU8_U8 = 4,
  MNN_T_D_TYPE_QU16_U16 = 5,
  MNN_T_D_TYPE_QI16_I16 = 6,
} mnn_tensor_dtype;

/**
 * @brief Create tensor with dimension size and type
 * @param dim_size Dimension size
 * @param type Dimension type
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t mnn_tensor_create(int dim_size, mnn_dimension_type_t type);

/**
 * @brief Create tensor with same shape as given tensor
 * @param self Shape provider
 * @param type Dimension type
 * @param alloc_memory Whether allocate memory
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t
mnn_tensor_create_from_tensor(mnn_tensor_t self, mnn_dimension_type_t type, bool alloc_memory);

/**
 * @brief Create device tensor
 * @param shape Tensor shape array
 * @param shape_size Shape array size
 * @param type Data type
 * @param dim_type Dimension type
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t mnn_tensor_create_device(
    const int *shape, int shape_size, struct halide_type_t type, mnn_dimension_type_t dim_type
);

/**
 * @brief Create tensor with data
 * @param shape Tensor shape array
 * @param shape_size Shape array size
 * @param type Data type
 * @param data Data pointer
 * @param dim_type Dimension type
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t mnn_tensor_create_with_data(
    const int *shape,
    int shape_size,
    struct halide_type_t type,
    void *data,
    mnn_dimension_type_t dim_type
);

/**
 * @brief Destroy tensor
 * @param tensor Tensor to destroy
 */
MNN_C_API void mnn_tensor_destroy(mnn_tensor_t tensor);

/**
 * @brief Clone tensor
 * @param src Source tensor
 * @param deep_copy Whether to perform deep copy
 * @return Cloned tensor or NULL if failed
 */
MNN_C_API mnn_tensor_t mnn_tensor_clone(mnn_tensor_t src, bool deep_copy);

/**
 * @brief Copy data from host tensor
 * @param self Target tensor
 * @param host_tensor Source tensor
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_tensor_copy_from_host(mnn_tensor_t self, mnn_tensor_t host_tensor);

/**
 * @brief Copy data to host tensor
 * @param self Source tensor
 * @param host_tensor Target tensor
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_tensor_copy_to_host(mnn_tensor_t self, mnn_tensor_t host_tensor);

/**
 * @brief Get tensor dimensions
 * @param self Tensor
 * @return Dimension count
 */
MNN_C_API int mnn_tensor_dimensions(mnn_tensor_t self);

/**
 * @brief Get tensor shape
 * @param self Tensor
 * @param shape Output shape array (must be pre-allocated)
 * @param shape_size Shape array size
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_tensor_shape(mnn_tensor_t self, int *shape, int shape_size);

/**
 * @brief Get tensor data size in bytes
 * @param self Tensor
 * @return Size in bytes
 */
MNN_C_API int mnn_tensor_size(mnn_tensor_t self);

/**
 * @brief Get tensor element count
 * @param self Tensor
 * @return Element count
 */
MNN_C_API int mnn_tensor_element_size(mnn_tensor_t self);

/**
 * @brief Get tensor shape in bytes (unsigned)
 * @param self Tensor
 * @return Size in bytes
 */
MNN_C_API size_t mnn_tensor_usize(mnn_tensor_t self);

/**
 * @brief Get tensor width
 * @param self Tensor
 * @return Width
 */
MNN_C_API int mnn_tensor_width(mnn_tensor_t self);

/**
 * @brief Get tensor height
 * @param self Tensor
 * @return Height
 */
MNN_C_API int mnn_tensor_height(mnn_tensor_t self);

/**
 * @brief Get tensor channel
 * @param self Tensor
 * @return Channel
 */
MNN_C_API int mnn_tensor_channel(mnn_tensor_t self);

/**
 * @brief Get tensor batch
 * @param self Tensor
 * @return Batch
 */
MNN_C_API int mnn_tensor_batch(mnn_tensor_t self);

/**
 * @brief Get tensor stride
 * @param self Tensor
 * @param index Dimension index
 * @return Stride
 */
MNN_C_API int mnn_tensor_stride(mnn_tensor_t self, int index);

/**
 * @brief Get tensor length
 * @param self Tensor
 * @param index Dimension index
 * @return Length
 */
MNN_C_API int mnn_tensor_length(mnn_tensor_t self, int index);

/**
 * @brief Set tensor stride
 * @param self Tensor
 * @param index Dimension index
 * @param stride Stride value
 */
MNN_C_API void mnn_tensor_set_stride(mnn_tensor_t self, int index, int stride);

/**
 * @brief Set tensor length
 * @param self Tensor
 * @param index Dimension index
 * @param length Length value
 */
MNN_C_API void mnn_tensor_set_length(mnn_tensor_t self, int index, int length);

/**
 * @brief Get host data pointer
 * @param self Tensor
 * @return Data pointer or NULL
 */
MNN_C_API void *mnn_tensor_host(mnn_tensor_t self);

/**
 * @brief Get device ID
 * @param self Tensor
 * @return Device ID
 */
MNN_C_API uint64_t mnn_tensor_device_id(mnn_tensor_t self);

/**
 * @brief Get buffer
 * @param self Tensor
 * @return Buffer pointer
 */
MNN_C_API struct halide_buffer_t *mnn_tensor_buffer(mnn_tensor_t self);

/**
 * @brief Get dimension type
 * @param self Tensor
 * @return Dimension type
 */
MNN_C_API mnn_dimension_type_t mnn_tensor_get_dimension_type(mnn_tensor_t self);

/**
 * @brief Get handle data type
 * @param self Tensor
 * @return Handle data type
 */
MNN_C_API mnn_handle_data_type_t mnn_tensor_get_handle_data_type(mnn_tensor_t self);

/**
 * @brief Set data type
 * @param self Tensor
 * @param type Data type
 */
MNN_C_API void mnn_tensor_set_type(mnn_tensor_t self, int type);

/**
 * @brief Get data type
 * @param self Tensor
 * @return Data type
 */
MNN_C_API struct halide_type_t *mnn_tensor_get_type(mnn_tensor_t self);

/**
 * @brief Map tensor for access
 * @param self Tensor
 * @param mtype Map type
 * @param dtype Dimension type
 * @return Mapped pointer or NULL
 */
MNN_C_API void *mnn_tensor_map(mnn_tensor_t self, mnn_map_type_t mtype, mnn_dimension_type_t dtype);

/**
 * @brief Unmap tensor
 * @param self Tensor
 * @param mtype Map type
 * @param dtype Dimension type
 * @param map_ptr Mapped pointer
 */
MNN_C_API void mnn_tensor_unmap(
    mnn_tensor_t self, mnn_map_type_t mtype, mnn_dimension_type_t dtype, void *map_ptr
);

/**
 * @brief Wait for tensor ready
 * @param self Tensor
 * @param mtype Map type
 * @param finish Whether wait for finish
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_tensor_wait(mnn_tensor_t self, mnn_map_type_t mtype, bool finish);

/**
 * @brief Set device pointer
 * @param self Tensor
 * @param device_ptr Device pointer
 * @param memory_type Memory type
 * @return Error code
 */
MNN_C_API mnn_error_code_t
mnn_tensor_set_device_ptr(mnn_tensor_t self, const void *device_ptr, int memory_type);

MNN_C_API void mnn_tensor_print(mnn_tensor_t self);
MNN_C_API void mnn_tensor_print_shape(mnn_tensor_t self);

#ifdef __cplusplus
}
#endif

#endif // MNN_TENSOR_H
