/*
 * interpreter.h
 * MNN C API for Interpreter
 *
 * This file provides C-style API for MNN's Interpreter functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#ifndef MNN_INTERPRETER_H
#define MNN_INTERPRETER_H

#include "error_code.h"
#include "mnn_type.h"
#include "tensor.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
  #include "MNN/Interpreter.hpp"
extern "C" {
#endif

/**
 * Opaque pointer types
 */
#ifdef __cplusplus
typedef MNN::Interpreter *mnn_interpreter_t;
typedef MNN::Session     *mnn_session_t;
typedef MNN::RuntimeInfo *mnn_runtime_info_t;
#else
typedef void *mnn_interpreter_t;
typedef void *mnn_session_t;
typedef void *mnn_runtime_info_t;
#endif
typedef void *mnn_backend_t;

/** Forward type enum */
// typedef mnn_forward_type mnn_forward_type_t;
typedef int mnn_forward_type_t;

/** Schedule config structure */
typedef struct mnn_schedule_config_t {
  mnn_forward_type_t type;
  /** CPU:number of threads in parallel , Or GPU: mode setting*/
  union {
    int num_thread;
    int mode;
  };
  mnn_forward_type_t    backupType;
  mnn_backend_config_t *backend_config;
} mnn_schedule_config_t;

/**
 * @brief Create interpreter from file
 * @param file_path Path to model file
 * @param callback Callback function to be called after creation
 * @return Interpreter instance or NULL if failed
 */
MNN_C_API mnn_interpreter_t
mnn_interpreter_create_from_file(const char *file_path, mnn_callback_0 callback);

/**
 * @brief Create interpreter from buffer
 * @param buffer Model data buffer
 * @param size Buffer size
 * @param callback Callback function to be called after creation
 * @return Interpreter instance or NULL if failed
 */
MNN_C_API mnn_interpreter_t
mnn_interpreter_create_from_buffer(const void *buffer, size_t size, mnn_callback_0 callback);

/**
 * @brief Destroy interpreter instance
 * @param self Interpreter to destroy
 */
MNN_C_API void mnn_interpreter_destroy(mnn_interpreter_t self);

/**
 * @brief Create runtime info
 * @param configs Schedule config array
 * @param count Config count
 * @return Runtime info instance
 */
MNN_C_API mnn_runtime_info_t
mnn_interpreter_create_runtime(const mnn_schedule_config_t *configs, size_t count);

/**
 * @brief Destroy runtime info
 * @param runtime Runtime info to destroy
 */
MNN_C_API void mnn_runtime_info_destroy(mnn_runtime_info_t runtime);

/**
 * @brief Create session with config
 * @param self Interpreter instance
 * @param config Schedule config
 * @param callback Callback function to be called after creation
 * @return Session instance or NULL if failed
 */
MNN_C_API mnn_session_t mnn_interpreter_create_session(
    mnn_interpreter_t self, const mnn_schedule_config_t *config, mnn_callback_0 callback
);

/**
 * @brief Create session with runtime info
 * @param self Interpreter instance
 * @param config Schedule config
 * @param runtime Runtime info
 * @param callback Callback function to be called after creation
 * @return Session instance or NULL if failed
 */
MNN_C_API mnn_session_t mnn_interpreter_create_session_with_runtime(
    mnn_interpreter_t            self,
    const mnn_schedule_config_t *config,
    mnn_runtime_info_t           runtime,
    mnn_callback_0               callback
);

/**
 * @brief Release session
 * @param self Interpreter instance
 * @param session Session to release
 * @param callback Callback function to be called after release
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_release_session(
    mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback
);

/**
 * @brief Resize session
 * @param self Interpreter instance
 * @param session Session to resize
 * @param callback Callback function to be called after resize
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_resize_session(
    mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback
);

/**
 * @brief Run session
 * @param self Interpreter instance
 * @param session Session to run
 * @param callback Callback function to be called after run
 * @return Error code
 */
MNN_C_API mnn_error_code_t
mnn_interpreter_run_session(mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback);

/**
 * @brief Get input tensor by name
 * @param self Interpreter instance
 * @param session Session
 * @param name Tensor name (NULL for first input)
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t
mnn_interpreter_get_session_input(mnn_interpreter_t self, mnn_session_t session, const char *name);

/**
 * @brief Get output tensor by name
 * @param self Interpreter instance
 * @param session Session
 * @param name Tensor name (NULL for first output)
 * @return Tensor instance or NULL if failed
 */
MNN_C_API mnn_tensor_t
mnn_interpreter_get_session_output(mnn_interpreter_t self, mnn_session_t session, const char *name);

/**
 * @brief Set session mode
 * @param self Interpreter instance
 * @param mode Session mode
 */
MNN_C_API void mnn_interpreter_set_session_mode(mnn_interpreter_t self, int mode);

/**
 * @brief Get MNN version
 * @return Version string
 */
MNN_C_API const char *mnn_get_version();

/**
 * @brief Get biz code from interpreter
 * @param self Interpreter instance
 * @return Biz code string or NULL if failed
 */
MNN_C_API const char *mnn_interpreter_biz_code(mnn_interpreter_t self);

/**
 * @brief Get uuid from interpreter
 * @param self Interpreter instance
 * @return Uuid string or NULL if failed
 */
MNN_C_API const char *mnn_interpreter_uuid(mnn_interpreter_t self);

/**
 * @brief Set cache file for interpreter
 * @param self Interpreter instance
 * @param cache_file Cache file path
 * @param key_size Key size
 */
MNN_C_API void
mnn_interpreter_set_cache_file(mnn_interpreter_t self, const char *cache_file, size_t key_size);

/**
 * @brief Set external file for interpreter
 * @param self Interpreter instance
 * @param file External file path
 * @param flag Flag value
 */
MNN_C_API void
mnn_interpreter_set_external_file(mnn_interpreter_t self, const char *file, size_t flag);

/**
 * @brief Set session hint
 * @param self Interpreter instance
 * @param mode Hint mode
 * @param value Hint value
 */
MNN_C_API void mnn_interpreter_set_session_hint(mnn_interpreter_t self, int mode, int value);

/**
 * @brief Release model
 * @param self Interpreter instance
 */
MNN_C_API void mnn_interpreter_release_model(mnn_interpreter_t self);

/**
 * @brief Get model buffer
 * @param self Interpreter instance
 * @param buffer Output parameter to receive pointer to model data
 * @return Size of model data in bytes, or 0 if failed
 */
MNN_C_API size_t mnn_interpreter_get_model_buffer(mnn_interpreter_t self, const void **buffer);

/**
 * @brief Get model version
 * @param self Interpreter instance
 * @return Version string or NULL if failed
 */
MNN_C_API const char *mnn_interpreter_get_model_version(mnn_interpreter_t self);

/**
 * @brief Update cache file
 * @param self Interpreter instance
 * @param session Session
 * @param flag Flag value
 * @return Error code
 */
MNN_C_API mnn_error_code_t
mnn_interpreter_update_cache_file(mnn_interpreter_t self, mnn_session_t session, int flag);

/**
 * @brief Update session to model
 * @param self Interpreter instance
 * @param session Session
 * @return Error code
 */
MNN_C_API mnn_error_code_t
mnn_interpreter_update_session_to_model(mnn_interpreter_t self, mnn_session_t session);

/**
 * @brief Get session info
 * @param self Interpreter instance
 * @param session Session
 * @param info Output parameter for session info
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_get_session_info(
    mnn_interpreter_t self, mnn_session_t session, int session_info_code, void *info
);

/**
 * @brief Get all output tensors from session
 * @param self Interpreter instance
 * @param session Session
 * @param tensors Output parameter for tensor array
 * @param count Output parameter for tensor count
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_get_session_output_all(
    mnn_interpreter_t self,
    mnn_session_t     session,
    mnn_tensor_t    **tensors,
    const char     ***names,
    size_t           *count
);

/**
 * @brief Get all input tensors from session
 * @param self Interpreter instance
 * @param session Session
 * @param tensors Output parameter for tensor array
 * @param count Output parameter for tensor count
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_get_session_input_all(
    mnn_interpreter_t self,
    mnn_session_t     session,
    mnn_tensor_t    **tensors,
    const char     ***names,
    size_t           *count
);

/**
 * @brief Resize tensor
 * @param tensor Tensor to resize
 * @param dims New dimensions array
 * @param dim_count Dimension count
 * @return Error code
 */
MNN_C_API mnn_error_code_t mnn_interpreter_resize_tensor(
    mnn_interpreter_t self, mnn_tensor_t tensor, const int *dims, int dim_count
);

MNN_C_API mnn_error_code_t mnn_interpreter_resize_tensor_1(
    mnn_interpreter_t self, mnn_tensor_t tensor, int batch, int channel, int height, int width
);

/**
 * @brief Get backend type
 * @param self Interpreter instance
 * @param session Session
 * @return Backend type
 */
MNN_C_API mnn_backend_t
mnn_interpreter_get_backend(mnn_interpreter_t self, mnn_session_t session, mnn_tensor_t tensor);

// /*
//      * @brief run session.
//      * @param session   given session.
//      * @param before    callback before each op. return true to run the op; return false to skip
//      the op.
//      * @param after     callback after each op. return true to continue running; return false to
//      interrupt the session.
//      * @param sync      synchronously wait for finish of execution or not.
//      * @return result of running.
//      */
//      ErrorCode runSessionWithCallBack(const Session* session, const TensorCallBack& before, const
//      TensorCallBack& end,
//       bool sync = false) const;

// /*
// * @brief run session.
// * @param session   given session.
// * @param before    callback before each op. return true to run the op; return false to skip the
// op.
// * @param after     callback after each op. return true to continue running; return false to
// interrupt the session.
// * @param sync      synchronously wait for finish of execution or not.
// * @return result of running.
// */
// ErrorCode runSessionWithCallBackInfo(const Session* session, const TensorCallBackWithInfo&
// before,
//           const TensorCallBackWithInfo& end, bool sync = false) const;

#ifdef __cplusplus
}
#endif

#endif // MNN_INTERPRETER_H
