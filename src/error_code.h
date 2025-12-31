#ifndef MNN_ERROR_CODE_H
#define MNN_ERROR_CODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Error code enum */
typedef enum {
  MNNC_NO_ERROR = 0,
  MNNC_OUT_OF_MEMORY = 1,
  MNNC_NOT_SUPPORT = 2,
  MNNC_COMPUTE_SIZE_ERROR = 3,
  MNNC_NO_EXECUTION = 4,
  MNNC_INVALID_VALUE = 5,

  // User error
  MNNC_INPUT_DATA_ERROR = 10,
  MNNC_CALL_BACK_STOP = 11,

  // Op Resize Error
  MNNC_TENSOR_NOT_SUPPORT = 20,
  MNNC_TENSOR_NEED_DIVIDE = 21,

  // File error
  MNNC_FILE_CREATE_FAILED = 30,
  MNNC_FILE_REMOVE_FAILED = 31,
  MNNC_FILE_OPEN_FAILED = 32,
  MNNC_FILE_CLOSE_FAILED = 33,
  MNNC_FILE_RESIZE_FAILED = 34,
  MNNC_FILE_SEEK_FAILED = 35,
  MNNC_FILE_NOT_EXIST = 36,
  MNNC_FILE_UNMAP_FAILED = 37,

  // custom
  MNNC_BOOL_TRUE = 100,
  MNNC_BOOL_FALSE = 101,
  MNNC_UNKNOWN_ERROR = 102,
  MNNC_INVALID_PTR = 103
} mnn_error_code_t;

#ifdef __cplusplus
}
#endif

#endif // MNN_ERROR_CODE_H
