#ifndef MNN_FORWARD_TYPE_H
#define MNN_FORWARD_TYPE_H

#include "MNN/HalideRuntime.h"
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

typedef struct {
  uint8_t  code;
  uint8_t  bits;
  uint16_t lanes;
} halide_type_c_t;

/**
 * The raw representation of an image passed around by generated
 * Halide code. It includes some stuff to track whether the image is
 * not actually in main memory, but instead on a device (like a
 * GPU). For a more convenient C++ wrapper, use Halide::Buffer<T>. */
typedef struct {
  /** A device-handle for e.g. GPU memory used to back this buffer. */
  uint64_t device;

  /** The interface used to interpret the above handle. */
  const struct halide_device_interface_t *device_interface;

  /** A pointer to the start of the data in main memory. In terms of
   * the Halide coordinate system, this is the address of the min
   * coordinates (defined below). */
  uint8_t *host;

  /** flags with various meanings. */
  uint64_t flags;

  /** The type of each buffer element. */
  halide_type_c_t type;

  /** The dimensionality of the buffer. */
  int32_t dimensions;

  /** The shape of the buffer. Halide does not own this array - you
   * must manage the memory for it yourself. */
  halide_dimension_t *dim;

  /** Pads the buffer up to a multiple of 8 bytes */
  void *padding;
} halide_buffer_c_t;

typedef void (*mnn_callback_0)();

typedef struct {
  // mnn_memory_mode memory;
  int memory;
  // mnn_power_mode power;
  int power;
  // mnn_precision_mode precision;
  int precision;
  union {
    void  *sharedContext;
    size_t flags;
  };
} mnn_backend_config_t;

#ifdef __cplusplus
}
#endif

#endif // MNN_FORWARD_TYPE_H
