#ifndef MNN_FORWARD_TYPE_H
#define MNN_FORWARD_TYPE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// typedef enum {
//   MNN_FORWARD_CPU = 0,
//   /*
//    Firtly find the first available backends not equal to CPU
//    If no other backends, use cpu
//    */
//   MNN_FORWARD_AUTO = 4,

//   /*Hand write metal*/
//   MNN_FORWARD_METAL = 1,

//   /*NVIDIA GPU API*/
//   MNN_FORWARD_CUDA = 2,

//   /*Android / Common Device GPU API*/
//   MNN_FORWARD_OPENCL = 3,
//   MNN_FORWARD_OPENGL = 6,
//   MNN_FORWARD_VULKAN = 7,

//   /*Android 8.1's NNAPI or CoreML for ios*/
//   MNN_FORWARD_NN = 5,

//   /*User can use API from Backend.hpp to add or search Backend*/
//   MNN_FORWARD_USER_0 = 8,
//   MNN_FORWARD_USER_1 = 9,
//   MNN_FORWARD_USER_2 = 10,
//   MNN_FORWARD_USER_3 = 11,

//   MNN_FORWARD_ALL = 12,

//   /* Apply arm extension instruction set to accelerate some Ops, this forward
//      type is only used in MNN internal, and will be active automatically when
//      user set forward type to be MNN_FORWARD_CPU and extension instruction set
//      is valid on hardware.
//   */
//   MNN_FORWARD_CPU_EXTENSION = 13,
//   // use for shared memory on android device

//   MNN_MEMORY_AHARDWAREBUFFER = 14
// } mnn_forward_type;

// typedef enum {
//   // For the OpenCL backend, all five of the following options are valid. The
//   // user is allowed to enable any one of them.
//   // For the Vulkan backend, only options MNN_GPU_TUNING_NONE,
//   // MNN_GPU_TUNING_HEAVY, and MNN_GPU_TUNING_WIDE are valid. The user is
//   // allowed to enable any one of these three.
//   MNN_GPU_TUNING_NONE = 1 << 0,   /* Forbidden tuning, performance not good.(OpenCL/Vulkan) */
//   MNN_GPU_TUNING_HEAVY = 1 << 1,  /* Heavily tuning, usually not suggested.(OpenCL/Vulkan) */
//   MNN_GPU_TUNING_WIDE = 1 << 2,   /* Widely tuning, performance good. Default.(OpenCL/Vulkan) */
//   MNN_GPU_TUNING_NORMAL = 1 << 3, /* Normal tuning, performance may be ok.(OpenCL) */
//   MNN_GPU_TUNING_FAST = 1 << 4,   /* Fast tuning, performance may not good.(OpenCL) */

//   // For the OpenCL backend, the following two options are both valid. The user
//   // could try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both, and then
//   // choose the better one based on performance.
//   // For the Vulkan backend, neither option is valid. The user uses the CMake
//   // option MNN_VULKAN_IMAGE to select between image memory mode and buffer
//   // memory mode.
//   MNN_GPU_MEMORY_BUFFER = 1 << 6, /* OpenCL_MEMORY_BUFFER */
//   MNN_GPU_MEMORY_IMAGE = 1 << 7,  /* OpenCL_MEMORY_IMAGE */

//   // For the OpenCL backend, the following two options are effective only on
//   // Qualcomm GPUs. When using a Qualcomm GPU, the user could try both options
//   // and choose the better one based on performance.
//   // For the Vulkan backend, only option MNN_GPU_RECORD_BATCH is valid. When
//   // MNN_GPU_RECORD_BATCH is enabled, all ops would share one commandBuffer.
//   MNN_GPU_RECORD_OP = 1 << 8,    /* The kernels in one op execution record into one
//                                     recording.(OpenCL) */
//   MNN_GPU_RECORD_BATCH = 1 << 9, /* 10 kernels record into one recording.(OpenCL) All ops share
//   one
//                                     commandBuffer.(Vulkan) */
// } mnn_gpu_mode;

// typedef enum { MNN_MEMORY_NORMAL = 0, MNN_MEMORY_HIGH, MNN_MEMORY_LOW } mnn_memory_mode;

// typedef enum { MNN_POWER_NORMAL = 0, MNN_POWER_HIGH, MNN_POWER_LOW } mnn_power_mode;

// typedef enum {
//   MNN_PRECISION_NORMAL = 0,
//   MNN_PRECISION_HIGH,
//   MNN_PRECISION_LOW,
//   MNN_PRECISION_LOW_BF16
// } mnn_precision_mode;

// typedef enum {
//   /**
//    * get status whether this runtime support 16-bits float point arithmetic
//    */
//   STATUS_SUPPORT_FP16,
//   /**
//    * get status whether this runtime support dot-product arithmetic
//    */
//   STATUS_SUPPORT_DOT_PRODUCT,
//   /**
//    * get status whether this runtime support power-low (means low priority for
//    * opencl)
//    */
//   STATUS_SUPPORT_POWER_LOW,
//   /**
//    * emum total number
//    */
//   STATUS_COUNT
// } mnn_runtime_status;

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
} mnn_backend_config;

#ifdef __cplusplus
}
#endif

#endif // MNN_FORWARD_TYPE_H
