#ifndef MNN_FORWARD_TYPE_H
#define MNN_FORWARD_TYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define MNN_C_API __declspec(dllexport)
#else
#define MNN_C_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*mnn_callback_0)();

typedef struct {
  // mnn_memory_mode memory;
  int memory;
  // mnn_power_mode power;
  int power;
  // mnn_precision_mode precision;
  int precision;
  union {
    void *sharedContext;
    size_t flags;
  };
} mnn_backend_config_t;

#ifdef __cplusplus
}
#endif

#endif // MNN_FORWARD_TYPE_H
