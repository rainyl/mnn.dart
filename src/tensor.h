//
// tensor.h
// MNN C API for Tensor
//
// This file provides C-style API for MNN's Tensor functionality
// All functions use snake_case naming convention
//

#ifndef MNN_TENSOR_H
#define MNN_TENSOR_H

#include "MNN/HalideRuntime.h"
#include "error_code.h"
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

struct halide_type_t *mnn_halide_type_create();
struct halide_type_t *
mnn_halide_type_create_1(halide_type_code_t code, uint8_t bits, uint16_t lanes);
void mnn_halide_type_destroy(struct halide_type_t *self);

/**
 * @brief Create tensor with dimension size and type
 * @param dim_size Dimension size
 * @param type Dimension type
 * @return Tensor instance or NULL if failed
 */
mnn_tensor_t mnn_tensor_create(int dim_size, mnn_dimension_type_t type);

/**
 * @brief Create tensor with same shape as given tensor
 * @param tensor Shape provider
 * @param type Dimension type
 * @param alloc_memory Whether allocate memory
 * @return Tensor instance or NULL if failed
 */
mnn_tensor_t
mnn_tensor_create_from_tensor(mnn_tensor_t tensor, mnn_dimension_type_t type, bool alloc_memory);

/**
 * @brief Create device tensor
 * @param shape Tensor shape array
 * @param shape_size Shape array size
 * @param type Data type
 * @param dim_type Dimension type
 * @return Tensor instance or NULL if failed
 */
mnn_tensor_t mnn_tensor_create_device(
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
mnn_tensor_t mnn_tensor_create_with_data(
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
void mnn_tensor_destroy(mnn_tensor_t tensor);

/**
 * @brief Clone tensor
 * @param src Source tensor
 * @param deep_copy Whether to perform deep copy
 * @return Cloned tensor or NULL if failed
 */
mnn_tensor_t mnn_tensor_clone(mnn_tensor_t src, bool deep_copy);

/**
 * @brief Copy data from host tensor
 * @param tensor Target tensor
 * @param host_tensor Source tensor
 * @return Error code
 */
mnn_error_code_t mnn_tensor_copy_from_host(mnn_tensor_t tensor, mnn_tensor_t host_tensor);

/**
 * @brief Copy data to host tensor
 * @param tensor Source tensor
 * @param host_tensor Target tensor
 * @return Error code
 */
mnn_error_code_t mnn_tensor_copy_to_host(mnn_tensor_t tensor, mnn_tensor_t host_tensor);

/**
 * @brief Get tensor dimensions
 * @param tensor Tensor
 * @return Dimension count
 */
int mnn_tensor_dimensions(mnn_tensor_t tensor);

/**
 * @brief Get tensor shape
 * @param tensor Tensor
 * @param shape Output shape array (must be pre-allocated)
 * @param shape_size Shape array size
 * @return Error code
 */
mnn_error_code_t mnn_tensor_shape(mnn_tensor_t tensor, int *shape, int shape_size);

/**
 * @brief Get tensor data size in bytes
 * @param tensor Tensor
 * @return Size in bytes
 */
int mnn_tensor_size(mnn_tensor_t tensor);

/**
 * @brief Get tensor element count
 * @param tensor Tensor
 * @return Element count
 */
int mnn_tensor_element_size(mnn_tensor_t tensor);

/**
 * @brief Get tensor shape in bytes (unsigned)
 * @param tensor Tensor
 * @return Size in bytes
 */
size_t mnn_tensor_usize(mnn_tensor_t tensor);

/**
 * @brief Get tensor width
 * @param tensor Tensor
 * @return Width
 */
int mnn_tensor_width(mnn_tensor_t tensor);

/**
 * @brief Get tensor height
 * @param tensor Tensor
 * @return Height
 */
int mnn_tensor_height(mnn_tensor_t tensor);

/**
 * @brief Get tensor channel
 * @param tensor Tensor
 * @return Channel
 */
int mnn_tensor_channel(mnn_tensor_t tensor);

/**
 * @brief Get tensor batch
 * @param tensor Tensor
 * @return Batch
 */
int mnn_tensor_batch(mnn_tensor_t tensor);

/**
 * @brief Get tensor stride
 * @param tensor Tensor
 * @param index Dimension index
 * @return Stride
 */
int mnn_tensor_stride(mnn_tensor_t tensor, int index);

/**
 * @brief Get tensor length
 * @param tensor Tensor
 * @param index Dimension index
 * @return Length
 */
int mnn_tensor_length(mnn_tensor_t tensor, int index);

/**
 * @brief Set tensor stride
 * @param tensor Tensor
 * @param index Dimension index
 * @param stride Stride value
 */
void mnn_tensor_set_stride(mnn_tensor_t tensor, int index, int stride);

/**
 * @brief Set tensor length
 * @param tensor Tensor
 * @param index Dimension index
 * @param length Length value
 */
void mnn_tensor_set_length(mnn_tensor_t tensor, int index, int length);

/**
 * @brief Get host data pointer
 * @param tensor Tensor
 * @return Data pointer or NULL
 */
void *mnn_tensor_host(mnn_tensor_t tensor);

/**
 * @brief Get device ID
 * @param tensor Tensor
 * @return Device ID
 */
uint64_t mnn_tensor_device_id(mnn_tensor_t tensor);

/**
 * @brief Get buffer
 * @param tensor Tensor
 * @return Buffer pointer
 */
 struct halide_buffer_t *mnn_tensor_buffer(mnn_tensor_t tensor);

/**
 * @brief Get dimension type
 * @param tensor Tensor
 * @return Dimension type
 */
mnn_dimension_type_t mnn_tensor_get_dimension_type(mnn_tensor_t tensor);

/**
 * @brief Get handle data type
 * @param tensor Tensor
 * @return Handle data type
 */
mnn_handle_data_type_t mnn_tensor_get_handle_data_type(mnn_tensor_t tensor);

/**
 * @brief Set data type
 * @param tensor Tensor
 * @param type Data type
 */
void mnn_tensor_set_type(mnn_tensor_t tensor, int type);

/**
 * @brief Get data type
 * @param tensor Tensor
 * @return Data type
 */
struct halide_type_t *mnn_tensor_get_type(mnn_tensor_t tensor);

/**
 * @brief Map tensor for access
 * @param tensor Tensor
 * @param mtype Map type
 * @param dtype Dimension type
 * @return Mapped pointer or NULL
 */
void *mnn_tensor_map(mnn_tensor_t tensor, mnn_map_type_t mtype, mnn_dimension_type_t dtype);

/**
 * @brief Unmap tensor
 * @param tensor Tensor
 * @param mtype Map type
 * @param dtype Dimension type
 * @param map_ptr Mapped pointer
 */
void mnn_tensor_unmap(
    mnn_tensor_t tensor, mnn_map_type_t mtype, mnn_dimension_type_t dtype, void *map_ptr
);

/**
 * @brief Wait for tensor ready
 * @param tensor Tensor
 * @param mtype Map type
 * @param finish Whether wait for finish
 * @return Error code
 */
mnn_error_code_t mnn_tensor_wait(mnn_tensor_t tensor, mnn_map_type_t mtype, bool finish);

/**
 * @brief Set device pointer
 * @param tensor Tensor
 * @param device_ptr Device pointer
 * @param memory_type Memory type
 * @return Error code
 */
mnn_error_code_t
mnn_tensor_set_device_ptr(mnn_tensor_t tensor, const void *device_ptr, int memory_type);

void mnn_tensor_print(mnn_tensor_t tensor);
void mnn_tensor_print_shape(mnn_tensor_t tensor);

#ifdef __cplusplus
}
#endif

#endif // MNN_TENSOR_H
