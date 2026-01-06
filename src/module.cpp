#define _GNU_SOURCE
//
// Created by rainy on 2026/1/1.
//

#include "module.h"
#include <cstring>
#include <string>
#include <vector>

using namespace MNN::Express;
using namespace MNN;

// MNN::Express::Module::Info
mnn_module_info_t mnn_module_info_create() { return new MNN::Express::Module::Info(); }

void mnn_module_info_destroy(mnn_module_info_t self) {
  if (self) {
    delete self;
    self = nullptr;
  }
}

mnn_expr_Variable_Info *mnn_module_info_get_inputs_at(mnn_module_info_t self, int index) {
  auto _info = self->inputs[index];

  auto info   = new mnn_expr_Variable_Info();
  info->order = static_cast<int>(_info.order);
  info->ndim  = _info.dim.size();
  auto pdim   = new int[_info.dim.size()];
  memcpy(pdim, _info.dim.data(), sizeof(int) * _info.dim.size());
  info->dim  = pdim;
  info->type = {static_cast<uint8_t>(_info.type.code), _info.type.bits, _info.type.lanes};
  info->size = _info.size;
  return info;
}

size_t mnn_module_info_get_inputs_length(mnn_module_info_t self) { return self->inputs.size(); }

int mnn_module_info_get_default_format(mnn_module_info_t self) {
  return static_cast<int>(self->defaultFormat);
}

size_t mnn_module_info_get_input_names(mnn_module_info_t self, /*return*/ char ***input_names) {
  auto len = self->inputs.size();
  if (input_names) {
    *input_names = new char *[len];
    for (int i = 0; i < len; i++) { (*input_names)[i] = strdup(self->inputNames[i].c_str()); }
  }
  return len;
}

size_t mnn_module_info_get_output_names(mnn_module_info_t self, /*return*/ char ***output_names) {
  auto len = self->outputNames.size();
  if (output_names) {
    *output_names = new char *[len];
    for (int i = 0; i < len; i++) { (*output_names)[i] = strdup(self->outputNames[i].c_str()); }
  }
  return len;
}

char *mnn_module_info_get_version(mnn_module_info_t self) { return strdup(self->version.c_str()); }

char *mnn_module_info_get_bizCode(mnn_module_info_t self) { return strdup(self->bizCode.c_str()); }

char *mnn_module_info_get_uuid(mnn_module_info_t self) { return strdup(self->uuid.c_str()); }

size_t mnn_module_info_get_metadata(
    mnn_module_info_t self, /*return*/ char ***keys, /*return*/ char ***values
) {
  auto len = self->metaData.size();
  if (keys && values) {
    *keys    = new char *[len];
    *values  = new char *[len];
    size_t i = 0;
    for (auto it = self->metaData.begin(); it != self->metaData.end(); ++it) {
      (*keys)[i]   = strdup(it->first.c_str());
      (*values)[i] = strdup(it->second.c_str());
      i++;
    }
  }
  return len;
}

// Executor, ExecutorScope
mnn_executor_t mnn_executor_static_new_executor(/*MNNForwardType*/ int type,
                                                mnn_backend_config_t   config,
                                                int                    num_thread) {
  MNN::BackendConfig _config;
  _config.memory        = static_cast<MNN::BackendConfig::MemoryMode>(config.memory);
  _config.power         = static_cast<MNN::BackendConfig::PowerMode>(config.power);
  _config.precision     = static_cast<MNN::BackendConfig::PrecisionMode>(config.precision);
  _config.sharedContext = config.sharedContext;
  _config.flags         = config.flags;
  return new std::shared_ptr<MNN::Express::Executor>(
      MNN::Express::Executor::newExecutor(static_cast<MNNForwardType>(type), _config, num_thread)
  );
}

mnn_executor_t mnn_executor_static_get_global_executor() {
  return new std::shared_ptr<MNN::Express::Executor>(MNN::Express::Executor::getGlobalExecutor());
}

void mnn_executor_destroy(mnn_executor_t self) {
  if (self == nullptr) return;
  delete self;
  self = nullptr;
}

uint32_t mnn_executor_get_lazy_mode(mnn_executor_t self) { return (*self)->getLazyMode(); }

void mnn_executor_set_lazy_mode(mnn_executor_t self, uint32_t mode) {
  (*self)->setLazyComputeMode(mode);
}

void mnn_executor_set_lazyEval(mnn_executor_t self, bool enable) { (*self)->lazyEval = enable; }

bool mnn_executor_get_lazyEval(mnn_executor_t self) { return (*self)->lazyEval; }

void mnn_executor_set_global_executor_config(
    mnn_executor_t self, /*MNNForwardType*/ int type, mnn_backend_config_t config, int num_thread
) {
  MNN::BackendConfig _config;
  _config.memory        = static_cast<MNN::BackendConfig::MemoryMode>(config.memory);
  _config.power         = static_cast<MNN::BackendConfig::PowerMode>(config.power);
  _config.precision     = static_cast<MNN::BackendConfig::PrecisionMode>(config.precision);
  _config.sharedContext = config.sharedContext;
  _config.flags         = config.flags;
  return (*self)->setGlobalExecutorConfig(static_cast<MNNForwardType>(type), _config, num_thread);
}

int mnn_executor_get_current_runtime_status(
    mnn_executor_t self, /*RuntimeStatus*/ int status_enum
) {
  return (*self)->getCurrentRuntimeStatus(static_cast<RuntimeStatus>(status_enum));
}

void mnn_executor_gc(mnn_executor_t self, /*GCFlag*/ int flag) {
  (*self)->gc(static_cast<Executor::GCFlag>(flag));
}

mnn_runtime_info_t mnn_executor_static_get_runtime() {
  return new MNN::RuntimeInfo(MNN::Express::Executor::getRuntime());
}

mnn_executor_scope_t mnn_executor_scope_create(mnn_executor_t current) {
  return new MNN::Express::ExecutorScope(*current);
}

mnn_executor_scope_t mnn_executor_scope_create_with_name(const char *name, mnn_executor_t current) {
  return new MNN::Express::ExecutorScope(name, *current);
}

void mnn_executor_scope_destroy(mnn_executor_scope_t self) {
  if (self == nullptr) return;
  delete self;
  self = nullptr;
}

mnn_executor_t mnn_executor_scope_static_current_executor() {
  return new std::shared_ptr<MNN::Express::Executor>(MNN::Express::ExecutorScope::Current());
}

// RuntimeManager API
mnn_runtime_manager_t mnn_runtime_manager_create(mnn_schedule_config_t config) {
  MNN::ScheduleConfig _config;
  _config.type          = static_cast<MNNForwardType>(config.type);
  _config.numThread     = config.num_thread;
  _config.mode          = config.mode;
  _config.backendConfig = reinterpret_cast<MNN::BackendConfig *>(config.backend_config);

  return Executor::RuntimeManager::createRuntimeManager(_config);
}

void mnn_runtime_manager_destroy(mnn_runtime_manager_t self) {
  Executor::RuntimeManager::destroy(self);
}

void mnn_runtime_manager_set_cache(mnn_runtime_manager_t self, const char *cache_path) {
  if (self && cache_path) { self->setCache(cache_path); }
}

void mnn_runtime_manager_set_external_path(
    mnn_runtime_manager_t self, const char *external_path, int type
) {
  if (self && external_path) { self->setExternalPath(external_path, type); }
}

void mnn_runtime_manager_set_external_file(mnn_runtime_manager_t self, const char *path) {
  if (self && path) { self->setExternalFile(path); }
}

void mnn_runtime_manager_update_cache(mnn_runtime_manager_t self) {
  if (self) { self->updateCache(); }
}

bool mnn_runtime_manager_is_backend_support(
    mnn_runtime_manager_t self, /*MNNForwardType*/ int backend
) {
  if (self) {
    auto rval = self->isBackendSupport({static_cast<MNNForwardType>(backend)});
    return rval[0];
  }
  return false;
}

void mnn_runtime_manager_set_mode(
    mnn_runtime_manager_t self, /*Interpreter::SessionMode*/ int mode
) {
  self->setMode(static_cast<Interpreter::SessionMode>(mode));
}

void mnn_runtime_manager_set_hint(
    mnn_runtime_manager_t self, /*Interpreter::HintMode*/ int mode, int value
) {
  self->setHint(static_cast<Interpreter::HintMode>(mode), value);
}

bool mnn_runtime_manager_get_info(
    mnn_runtime_manager_t self, /*Interpreter::SessionInfoCode*/ int code, void *ptr
) {
  return self->getInfo(static_cast<Interpreter::SessionInfoCode>(code), ptr);
}

bool mnn_runtime_manager_static_get_device_info(
    const char *device_key, /*MNNForwardType*/ int type, /*return*/ char **device_value
) {
  std::string _device_value;
  auto        rval = MNN::Express::Executor::RuntimeManager::getDeviceInfo(
      device_key, static_cast<MNNForwardType>(type), _device_value
  );
  if (rval) { *device_value = strdup(_device_value.c_str()); }
  return rval;
}

std::vector<std::string> parse_c_strings(const char **s, int count) {
  std::vector<std::string> strings;
  if (s) {
    for (int i = 0; i < count; ++i) {
      if (s[i] != nullptr) { strings.emplace_back(s[i]); }
    }
  }
  return strings;
}

// Module API
mnn_module_t mnn_module_load_from_file(
    const char                *file_name,
    const char               **inputs,
    int                        input_count,
    const char               **outputs,
    int                        output_count,
    mnn_runtime_manager_t      mgr,
    const mnn_module_config_t *config
) {
  std::vector<std::string> input_names  = parse_c_strings(inputs, input_count);
  std::vector<std::string> output_names = parse_c_strings(outputs, output_count);

  Module::Config      mod_config;
  Module::BackendInfo backend_info;
  if (config) {
    mod_config.dynamic      = config->dynamic;
    mod_config.shapeMutable = config->shape_mutable;
    mod_config.rearrange    = config->rearrange;

    backend_info.type   = static_cast<MNNForwardType>(config->backend_info_type);
    backend_info.config = reinterpret_cast<MNN::BackendConfig *>(config->backend_info_config);

    mod_config.backend = &backend_info;
  }

  std::shared_ptr<Executor::RuntimeManager> rt_mgr_shared;
  if (mgr) {
    // Use a null deleter because the C user calls mnn_runtime_manager_destroy explicitly.
    rt_mgr_shared =
        std::shared_ptr<Executor::RuntimeManager>(mgr, [](Executor::RuntimeManager *) {});
  }

  return Module::load(
      input_names,
      output_names,
      file_name,
      mgr ? rt_mgr_shared : nullptr,
      config ? &mod_config : nullptr
  );
}

mnn_module_t mnn_module_load_from_bytes(
    const uint8_t             *buffer,
    size_t                     length,
    const char               **inputs,
    int                        input_count,
    const char               **outputs,
    int                        output_count,
    mnn_runtime_manager_t      mgr,
    const mnn_module_config_t *config
) {
  std::vector<std::string> input_names  = parse_c_strings(inputs, input_count);
  std::vector<std::string> output_names = parse_c_strings(outputs, output_count);

  Module::Config      mod_config;
  Module::BackendInfo backend_info;
  if (config) {
    mod_config.dynamic      = config->dynamic;
    mod_config.shapeMutable = config->shape_mutable;
    mod_config.rearrange    = config->rearrange;

    backend_info.type   = static_cast<MNNForwardType>(config->backend_info_type);
    backend_info.config = reinterpret_cast<MNN::BackendConfig *>(config->backend_info_config);

    mod_config.backend = &backend_info;
  }

  std::shared_ptr<Executor::RuntimeManager> rt_mgr_shared;
  if (mgr) {
    // Use a null deleter because the C user calls mnn_runtime_manager_destroy explicitly.
    rt_mgr_shared =
        std::shared_ptr<Executor::RuntimeManager>(mgr, [](Executor::RuntimeManager *) {});
  }

  return Module::load(
      input_names,
      output_names,
      buffer,
      length,
      mgr ? rt_mgr_shared : nullptr,
      config ? &mod_config : nullptr
  );
}

mnn_module_t mnn_module_extract(VecVARP_t inputs, VecVARP_t outputs, bool fortrain) {
  return Module::extract(*inputs, *outputs, fortrain);
}

mnn_module_t mnn_module_clone(mnn_module_t self, bool share_params) {
  return Module::clone(self, share_params);
}

void mnn_module_destroy(mnn_module_t self) {
  if (self) { Module::destroy(self); }
}

bool mnn_module_load_parameters(mnn_module_t self, VecVARP_t parameters) {
  return self->loadParameters(*parameters);
}

void mnn_module_clear_cache(mnn_module_t self) { self->clearCache(); }

char *mnn_module_get_name(mnn_module_t self) {
  if (self) return strdup(self->name().c_str());
  return nullptr;
}

void mnn_module_set_name(mnn_module_t self, const char *name) {
  if (self) self->setName(name);
}

char *mnn_module_get_type(mnn_module_t self) {
  if (self) return strdup(self->type().c_str());
  return nullptr;
}

void mnn_module_set_type(mnn_module_t self, const char *type) {
  if (self) self->setType(type);
}

int mnn_module_add_parameter(mnn_module_t self, VARP_t parameter) {
  if (!self || !parameter) return -1;
  return self->addParameter(*parameter);
}

void mnn_module_set_parameter(mnn_module_t self, VARP_t parameter, int index) {
  if (!self || !parameter) return;
  return self->setParameter(*parameter, index);
}

mnn_module_info_t mnn_module_get_info(mnn_module_t self) {
  if (!self) return nullptr;
  auto info = self->getInfo();
  if (!info) return nullptr;
  return new MNN::Express::Module::Info(*info);
}

mnn_error_code_t mnn_module_on_forward(
    mnn_module_t self, VecVARP_t inputs, VecVARP_t *outputs, mnn_callback_0 callback
) {
  if (!self || !inputs || !outputs) {
    if (callback) callback();
    return MNNC_INVALID_PTR;
  }
  try {
    auto _outputs = self->onForward(*inputs);
    *outputs      = new std::vector<VARP>(_outputs);
    if (callback) callback();
    return MNNC_NO_ERROR;
  } catch (...) {
    if (callback) callback();
    return MNNC_UNKNOWN_ERROR;
  }
}

mnn_error_code_t
mnn_module_forward(mnn_module_t self, VARP_t input, VARP_t *output, mnn_callback_0 callback) {
  if (!self || !input || !output) {
    if (callback) callback();
    return MNNC_INVALID_PTR;
  }
  try {
    auto _output = self->forward(*input);
    *output      = new VARP(_output);
    if (callback) callback();
    return MNNC_NO_ERROR;
  } catch (...) {
    if (callback) callback();
    return MNNC_UNKNOWN_ERROR;
  }
}

// Info
bool mnn_module_get_is_training(mnn_module_t self) {
  if (self) return self->getIsTraining();
  return false;
}

void mnn_module_set_is_training(mnn_module_t self, bool is_training) {
  if (self) self->setIsTraining(is_training);
}
