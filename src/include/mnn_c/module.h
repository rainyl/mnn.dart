//
// Created by rainy on 2026/1/1.
//

#ifndef MNN_C_API_MODULE_H
#define MNN_C_API_MODULE_H

#include "mnn_c/base.h"
#include "mnn_c/expr.h"
#include "mnn_c/interpreter.h"

#ifdef __cplusplus
  #include "MNN/expr/Executor.hpp"
  #include "MNN/expr/ExecutorScope.hpp"
  #include "MNN/expr/Module.hpp"
extern "C" {
#endif

#ifdef __cplusplus
typedef std::shared_ptr<MNN::Express::Executor> *mnn_executor_t;
typedef MNN::Express::ExecutorScope             *mnn_executor_scope_t;
typedef MNN::Express::Module                    *mnn_module_t;
typedef MNN::Express::Executor::RuntimeManager  *mnn_runtime_manager_t;
typedef MNN::Express::Module::Info              *mnn_module_info_t;
#else
typedef void *mnn_executor_t;
typedef void *mnn_executor_scope_t;
typedef void *mnn_module_t;
typedef void *mnn_runtime_manager_t;
typedef void *mnn_module_info_t;
#endif

// Config struct equivalent for C
typedef struct mnn_module_config_t {
  // Load module as dynamic, default static
  bool dynamic;

  // for static mode, if the shape is mutable, set true, otherwise set false to avoid resizeSession
  // freqencily
  bool shape_mutable;

  // Pre-rearrange weights or not. Disabled by default.
  // The weights will be rearranged in a general way, so the best implementation
  // may not be adopted if `rearrange` is enabled.
  bool rearrange;

  // BackendInfo* backend = nullptr;
  int                   backend_info_type; /*MNN_FORWARD_CPU*/
  mnn_backend_config_t *backend_info_config;

  // base module
  // const Module* base = nullptr;
} mnn_module_config_t;

// MNN::Express::Module::Info
MNN_C_API mnn_module_info_t mnn_module_info_create();
MNN_C_API void              mnn_module_info_destroy(mnn_module_info_t self);
MNN_C_API struct mnn_expr_Variable_Info                  *
mnn_module_info_get_inputs_at(mnn_module_info_t self, int index);
MNN_C_API size_t mnn_module_info_get_inputs_length(mnn_module_info_t self);
MNN_C_API int    mnn_module_info_get_default_format(mnn_module_info_t self);
MNN_C_API size_t
mnn_module_info_get_input_names(mnn_module_info_t self, /*return*/ char ***input_names);
MNN_C_API size_t
mnn_module_info_get_output_names(mnn_module_info_t self, /*return*/ char ***output_names);
MNN_C_API char  *mnn_module_info_get_version(mnn_module_info_t self);
MNN_C_API char  *mnn_module_info_get_bizCode(mnn_module_info_t self);
MNN_C_API char  *mnn_module_info_get_uuid(mnn_module_info_t self);
MNN_C_API size_t mnn_module_info_get_metadata(
    mnn_module_info_t self, /*return*/ char ***keys, /*return*/ char ***values
);

// Executor, ExecutorScope
MNN_C_API mnn_executor_t mnn_executor_static_new_executor(/*MNNForwardType*/ int type,
                                                          mnn_backend_config_t   config,
                                                          int                    num_thread);
MNN_C_API mnn_executor_t mnn_executor_static_get_global_executor();
MNN_C_API void           mnn_executor_destroy(mnn_executor_t self);
MNN_C_API uint32_t       mnn_executor_get_lazy_mode(mnn_executor_t self);
MNN_C_API void           mnn_executor_set_lazy_mode(mnn_executor_t self, uint32_t mode);
MNN_C_API void           mnn_executor_set_lazyEval(mnn_executor_t self, bool enable);
MNN_C_API bool           mnn_executor_get_lazyEval(mnn_executor_t self);
MNN_C_API void           mnn_executor_set_global_executor_config(
              mnn_executor_t self, /*MNNForwardType*/ int type, mnn_backend_config_t config, int num_thread
          );
MNN_C_API int
mnn_executor_get_current_runtime_status(mnn_executor_t self, /*RuntimeStatus*/ int status_enum);
MNN_C_API void               mnn_executor_gc(mnn_executor_t self, /*GCFlag*/ int flag);
MNN_C_API mnn_runtime_info_t mnn_executor_static_get_runtime();

MNN_C_API mnn_executor_scope_t mnn_executor_scope_create(mnn_executor_t current);
MNN_C_API mnn_executor_scope_t
               mnn_executor_scope_create_with_name(const char *name, mnn_executor_t current);
MNN_C_API void mnn_executor_scope_destroy(mnn_executor_scope_t self);
MNN_C_API mnn_executor_t mnn_executor_scope_static_current_executor();

// RuntimeManager API
MNN_C_API mnn_runtime_manager_t mnn_runtime_manager_create(mnn_schedule_config_t config);
MNN_C_API void                  mnn_runtime_manager_destroy(mnn_runtime_manager_t self);
MNN_C_API void mnn_runtime_manager_set_cache(mnn_runtime_manager_t self, const char *cache_path);
MNN_C_API void mnn_runtime_manager_set_external_path(
    mnn_runtime_manager_t self, const char *external_path, int type
);
MNN_C_API void mnn_runtime_manager_set_external_file(mnn_runtime_manager_t self, const char *path);
MNN_C_API void mnn_runtime_manager_update_cache(mnn_runtime_manager_t self);
MNN_C_API bool
mnn_runtime_manager_is_backend_support(mnn_runtime_manager_t self, /*MNNForwardType*/ int backend);
MNN_C_API void
mnn_runtime_manager_set_mode(mnn_runtime_manager_t self, /*Interpreter::SessionMode*/ int mode);
MNN_C_API void mnn_runtime_manager_set_hint(
    mnn_runtime_manager_t self, /*Interpreter::HintMode*/ int mode, int value
);
MNN_C_API bool mnn_runtime_manager_get_info(
    mnn_runtime_manager_t self, /*Interpreter::SessionInfoCode*/ int code, void *ptr
);
MNN_C_API bool mnn_runtime_manager_static_get_device_info(
    const char *device_key, /*MNNForwardType*/ int type, char **device_value
);

// Module API
// inputs and outputs are arrays of strings (char**), length is input_count/output_count
MNN_C_API mnn_module_t mnn_module_load_from_file(
    const char                *file_name,
    const char               **inputs,
    int                        input_count,
    const char               **outputs,
    int                        output_count,
    mnn_runtime_manager_t      mgr,
    const mnn_module_config_t *config
);
MNN_C_API mnn_module_t mnn_module_load_from_bytes(
    const uint8_t             *buffer,
    size_t                     length,
    const char               **inputs,
    int                        input_count,
    const char               **outputs,
    int                        output_count,
    mnn_runtime_manager_t      mgr,
    const mnn_module_config_t *config
);
MNN_C_API mnn_module_t mnn_module_extract(VecVARP_t inputs, VecVARP_t outputs, bool fortrain);
MNN_C_API mnn_module_t mnn_module_clone(mnn_module_t self, bool share_params);
MNN_C_API void         mnn_module_destroy(mnn_module_t self);

MNN_C_API bool mnn_module_load_parameters(mnn_module_t self, VecVARP_t parameters);
MNN_C_API bool mnn_module_get_is_training(mnn_module_t self);
MNN_C_API void mnn_module_set_is_training(mnn_module_t self, bool is_training);
MNN_C_API void mnn_module_clear_cache(mnn_module_t self);

MNN_C_API char *mnn_module_get_name(mnn_module_t self);
MNN_C_API void  mnn_module_set_name(mnn_module_t self, const char *name);
MNN_C_API char *mnn_module_get_type(mnn_module_t self);
MNN_C_API void  mnn_module_set_type(mnn_module_t self, const char *type);
MNN_C_API int   mnn_module_add_parameter(mnn_module_t self, VARP_t parameter);
MNN_C_API void  mnn_module_set_parameter(mnn_module_t self, VARP_t parameter, int index);
MNN_C_API mnn_module_info_t mnn_module_get_info(mnn_module_t self);

MNN_C_API mnn_error_code_t mnn_module_on_forward(
    mnn_module_t self, VecVARP_t inputs, VecVARP_t *outputs, mnn_callback_0 callback
);
MNN_C_API mnn_error_code_t
mnn_module_forward(mnn_module_t self, VARP_t input, VARP_t *output, mnn_callback_0 callback);

// void registerModel(const std::vector<std::shared_ptr<Module>>& children);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_API_MODULE_H
