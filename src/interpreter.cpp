/*
 * interpreter.cpp
 * MNN C API for Interpreter
 *
 * This file provides C-style API for MNN's Interpreter functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#include "interpreter.h"
#include "MNN/Interpreter.hpp"
#include "MNN/MNNForwardType.h"
#include "error_code.h"
#include "tensor.h"
#include <cstddef>
#include <cstring>
#include <string.h>

// Interpreter creation/destruction
mnn_interpreter_t mnn_interpreter_create_from_file(const char *file_path, mnn_callback_0 callback) {
  try {
    auto r = (mnn_interpreter_t)MNN::Interpreter::createFromFile(file_path);
    if (callback) callback();
    return r;
  } catch (...) {
    if (callback) callback();
    return nullptr;
  }
}

mnn_interpreter_t
mnn_interpreter_create_from_buffer(const void *buffer, size_t size, mnn_callback_0 callback) {
  try {
    auto r = (mnn_interpreter_t)MNN::Interpreter::createFromBuffer(buffer, size);
    if (callback) callback();
    return r;
  } catch (...) {
    if (callback) callback();
    return nullptr;
  }
}

void mnn_interpreter_destroy(mnn_interpreter_t self) {
  if (self) { MNN::Interpreter::destroy((MNN::Interpreter *)self); self = nullptr; }
}

const char *mnn_interpreter_biz_code(mnn_interpreter_t self) {
  if (!self) return nullptr;
  try {
    return ((MNN::Interpreter *)self)->bizCode();
  } catch (...) { return nullptr; }
}

const char *mnn_interpreter_uuid(mnn_interpreter_t self) {
  if (!self) return nullptr;
  try {
    return ((MNN::Interpreter *)self)->uuid();
  } catch (...) { return nullptr; }
}

void mnn_interpreter_set_cache_file(
    mnn_interpreter_t self, const char *cache_file, size_t key_size
) {
  if (!self || !cache_file) return;
  try {
    ((MNN::Interpreter *)self)->setCacheFile(cache_file, key_size);
  } catch (...) {}
}

void mnn_interpreter_set_external_file(mnn_interpreter_t self, const char *file, size_t flag) {
  if (!self || !file) return;
  try {
    ((MNN::Interpreter *)self)->setExternalFile(file, flag);
  } catch (...) {}
}

mnn_error_code_t
mnn_interpreter_update_cache_file(mnn_interpreter_t self, mnn_session_t session, int flag) {
  if (!self || !session) return MNNC_INVALID_PTR;
  try {
    return (mnn_error_code_t)((MNN::Interpreter *)self)
        ->updateCacheFile((MNN::Session *)session, flag);
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

void mnn_interpreter_set_session_hint(mnn_interpreter_t self, int mode, int value) {
  if (!self) return;
  try {
    ((MNN::Interpreter *)self)->setSessionHint((MNN::Interpreter::HintMode)mode, value);
  } catch (...) {}
}

void mnn_interpreter_release_model(mnn_interpreter_t self) {
  if (!self) return;
  try {
    ((MNN::Interpreter *)self)->releaseModel();
  } catch (...) {}
}

size_t mnn_interpreter_get_model_buffer(mnn_interpreter_t self, const void **buffer) {
  if (!self || !buffer) return 0;
  try {
    auto pair = ((MNN::Interpreter *)self)->getModelBuffer();
    *buffer = pair.first;
    return pair.second;
  } catch (...) {
    *buffer = nullptr;
    return 0;
  }
}

const char *mnn_interpreter_get_model_version(mnn_interpreter_t self) {
  if (!self) return nullptr;
  try {
    return ((MNN::Interpreter *)self)->getModelVersion();
  } catch (...) { return nullptr; }
}

mnn_error_code_t
mnn_interpreter_update_session_to_model(mnn_interpreter_t self, mnn_session_t session) {
  if (!self || !session) return MNNC_INVALID_PTR;
  try {
    return (mnn_error_code_t)((MNN::Interpreter *)self)
        ->updateSessionToModel((MNN::Session *)session);
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

// Runtime info management
mnn_runtime_info_t
mnn_interpreter_create_runtime(const mnn_schedule_config_t *configs, size_t count) {
  try {
    std::vector<MNN::ScheduleConfig> vecConfigs;
    for (size_t i = 0; i < count; i++) {
      MNN::ScheduleConfig config;
      config.type = static_cast<MNNForwardType>(configs[i].type);
      config.numThread = configs[i].num_thread;
      config.mode = configs[i].mode;
      config.backendConfig = (MNN::BackendConfig *)configs[i].backend_config;
      vecConfigs.push_back(config);
    }
    auto r = MNN::Interpreter::createRuntime(vecConfigs);
    return new MNN::RuntimeInfo(r);
  } catch (...) { return nullptr; }
}

void mnn_runtime_info_destroy(mnn_runtime_info_t runtime) {
  if (runtime) { delete (MNN::RuntimeInfo *)runtime; runtime = nullptr; }
}

// Session management
mnn_session_t mnn_interpreter_create_session(
    mnn_interpreter_t self, const mnn_schedule_config_t *config, mnn_callback_0 callback
) {
  if (!self || !config) {
    if (callback) callback();
    return nullptr;
  }
  try {
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = (MNNForwardType)config->type;
    scheduleConfig.numThread = config->num_thread;
    scheduleConfig.mode = config->mode;

    auto backendConfig = new MNN::BackendConfig();
    if (config->backend_config != nullptr) {
      backendConfig->memory = (MNN::BackendConfig::MemoryMode)config->backend_config->memory;
      backendConfig->power = (MNN::BackendConfig::PowerMode)config->backend_config->power;
      backendConfig->precision =
          (MNN::BackendConfig::PrecisionMode)config->backend_config->precision;
      backendConfig->sharedContext = config->backend_config->sharedContext;
      backendConfig->flags = config->backend_config->flags;

      scheduleConfig.backendConfig = backendConfig;
    }

    auto r = (mnn_session_t)((MNN::Interpreter *)self)->createSession(scheduleConfig);

    delete backendConfig;

    if (callback) callback();
    return r;
  } catch (...) {
    if (callback) callback();
    return nullptr;
  }
}

mnn_session_t mnn_interpreter_create_session_with_runtime(
    mnn_interpreter_t self,
    const mnn_schedule_config_t *config,
    mnn_runtime_info_t runtime,
    mnn_callback_0 callback
) {
  if (!self || !config || !runtime) {
    if (callback) callback();
    return nullptr;
  }
  try {
    MNN::ScheduleConfig scheduleConfig;
    scheduleConfig.type = (MNNForwardType)config->type;
    scheduleConfig.numThread = config->num_thread;
    scheduleConfig.backendConfig = (MNN::BackendConfig *)config->backend_config;
    auto r = (mnn_session_t)((MNN::Interpreter *)self)
                 ->createSession(scheduleConfig, *(MNN::RuntimeInfo *)runtime);
    if (callback) callback();
    return r;
  } catch (...) {
    if (callback) callback();
    return nullptr;
  }
}

mnn_error_code_t mnn_interpreter_release_session(
    mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback
) {
  if (!self || !session) {
    if (callback) callback();
    return MNNC_INVALID_PTR;
  }
  try {
    auto r = ((MNN::Interpreter *)self)->releaseSession((MNN::Session *)session);
    if (callback) callback();
    return r ? MNNC_BOOL_TRUE : MNNC_BOOL_FALSE;
  } catch (...) {
    if (callback) callback();
    return MNNC_UNKNOWN_ERROR;
  }
}

mnn_error_code_t mnn_interpreter_resize_session(
    mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback
) {
  if (!self || !session) {
    if (callback) callback();
    return MNNC_INVALID_PTR;
  }
  try {
    ((MNN::Interpreter *)self)->resizeSession((MNN::Session *)session);
    if (callback) callback();
    return MNNC_NO_ERROR;
  } catch (...) {
    if (callback) callback();
    return MNNC_UNKNOWN_ERROR;
  }
}

mnn_error_code_t mnn_interpreter_run_session(
    mnn_interpreter_t self, mnn_session_t session, mnn_callback_0 callback
) {
  if (!self || !session) {
    if (callback) callback();
    return MNNC_INVALID_PTR;
  }
  try {
    auto code = ((MNN::Interpreter *)self)->runSession((MNN::Session *)session);
    if (callback) callback();
    return (mnn_error_code_t)code;
  } catch (...) {
    if (callback) callback();
    return MNNC_UNKNOWN_ERROR;
  }
}

// Tensor operations
mnn_tensor_t
mnn_interpreter_get_session_input(mnn_interpreter_t self, mnn_session_t session, const char *name) {
  if (!self || !session) return nullptr;
  try {
    return (mnn_tensor_t)((MNN::Interpreter *)self)->getSessionInput((MNN::Session *)session, name);
  } catch (...) { return nullptr; }
}

mnn_tensor_t mnn_interpreter_get_session_output(
    mnn_interpreter_t self, mnn_session_t session, const char *name
) {
  if (!self || !session) return nullptr;
  try {
    return (mnn_tensor_t)((MNN::Interpreter *)self)
        ->getSessionOutput((MNN::Session *)session, name);
  } catch (...) { return nullptr; }
}

// Session configuration
void mnn_interpreter_set_session_mode(mnn_interpreter_t self, int mode) {
  if (!self) return;
  try {
    ((MNN::Interpreter *)self)->setSessionMode((MNN::Interpreter::SessionMode)mode);
  } catch (...) {
    // Ignore errors
  }
}

// Version info
const char *mnn_get_version() { return MNN::getVersion(); }

// Session info and tensor operations
mnn_error_code_t mnn_interpreter_get_session_info(
    mnn_interpreter_t self, mnn_session_t session, int session_info_code, void *info
) {
  if (!self || !session || !info) return MNNC_INVALID_PTR;
  try {
    bool success =
        ((MNN::Interpreter *)self)
            ->getSessionInfo(
                (MNN::Session *)session, (MNN::Interpreter::SessionInfoCode)session_info_code, info
            );
    return success ? MNNC_NO_ERROR : MNNC_UNKNOWN_ERROR;
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_interpreter_get_session_output_all(
    mnn_interpreter_t self,
    mnn_session_t session,
    mnn_tensor_t **tensors,
    const char ***names,
    size_t *count
) {
  if (!self || !session || !tensors || !names || !count) return MNNC_INVALID_PTR;
  try {
    auto outputs = ((MNN::Interpreter *)self)->getSessionOutputAll((MNN::Session *)session);
    *count = outputs.size();
    *tensors = (mnn_tensor_t *)malloc(sizeof(mnn_tensor_t) * (*count));
    *names = (const char **)malloc(sizeof(const char *) * (*count));
    size_t i = 0;
    for (const auto &pair : outputs) {
      (*tensors)[i] = (mnn_tensor_t)pair.second;
      (*names)[i] = strdup(pair.first.c_str());
      i++;
    }
    return MNNC_NO_ERROR;
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_interpreter_get_session_input_all(
    mnn_interpreter_t self,
    mnn_session_t session,
    mnn_tensor_t **tensors,
    const char ***names,
    size_t *count
) {
  if (!self || !session || !tensors || !names || !count) return MNNC_INVALID_PTR;
  try {
    auto inputs = ((MNN::Interpreter *)self)->getSessionInputAll((MNN::Session *)session);
    *count = inputs.size();
    *tensors = (mnn_tensor_t *)malloc(sizeof(mnn_tensor_t) * (*count));
    *names = (const char **)malloc(sizeof(const char *) * (*count));
    size_t i = 0;
    for (const auto &pair : inputs) {
      (*tensors)[i] = (mnn_tensor_t)pair.second;
      (*names)[i] = strdup(pair.first.c_str());
      i++;
    }
    return MNNC_NO_ERROR;
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_interpreter_resize_tensor(
    mnn_interpreter_t self, mnn_tensor_t tensor, const int *dims, int dim_count
) {
  if (!tensor || !dims || dim_count <= 0) return MNNC_INVALID_PTR;
  try {
    std::vector<int> dims_vec(dims, dims + dim_count);
    ((MNN::Interpreter *)self)->resizeTensor((MNN::Tensor *)tensor, dims_vec);
    return MNNC_NO_ERROR;
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

mnn_error_code_t mnn_interpreter_resize_tensor_1(
    mnn_interpreter_t self, mnn_tensor_t tensor, int batch, int channel, int height, int width
) {
  if (!self || !tensor) return MNNC_INVALID_PTR;
  try {
    ((MNN::Interpreter *)self)->resizeTensor((MNN::Tensor *)tensor, batch, channel, height, width);
    return MNNC_NO_ERROR;
  } catch (...) { return MNNC_UNKNOWN_ERROR; }
}

mnn_backend_t
mnn_interpreter_get_backend(mnn_interpreter_t self, mnn_session_t session, mnn_tensor_t tensor) {
  if (!self || !session) return nullptr;
  try {
    return (mnn_backend_t)((MNN::Interpreter *)self)
        ->getBackend((MNN::Session *)session, (MNN::Tensor *)tensor);
  } catch (...) { return nullptr; }
}
