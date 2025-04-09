/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

// For the OpenCL backend, all five of the following options are valid. The
// user is allowed to enable any one of them.
// For the Vulkan backend, only options MNN_GPU_TUNING_NONE,
// MNN_GPU_TUNING_HEAVY, and MNN_GPU_TUNING_WIDE are valid. The user is
// allowed to enable any one of these three.
///  Forbidden tuning, performance not good.(OpenCL/Vulkan)
const int MNN_GPU_TUNING_NONE = 1 << 0;

/// Heavily tuning, usually not suggested.(OpenCL/Vulkan)
const int MNN_GPU_TUNING_HEAVY = 1 << 1;

/// Widely tuning, performance good. Default.(OpenCL/Vulkan)
const int MNN_GPU_TUNING_WIDE = 1 << 2;

/// Normal tuning, performance may be ok.(OpenCL)
const int MNN_GPU_TUNING_NORMAL = 1 << 3;

/// Fast tuning, performance may not good.(OpenCL)
const int MNN_GPU_TUNING_FAST = 1 << 4;

// For the OpenCL backend, the following two options are both valid. The user
// could try OpenCL_MEMORY_BUFFER and OpenCL_MEMORY_IMAGE both, and then
// choose the better one based on performance.
// For the Vulkan backend, neither option is valid. The user uses the CMake
// option MNN_VULKAN_IMAGE to select between image memory mode and buffer
// memory mode.
/// OpenCL_MEMORY_BUFFER
const int MNN_GPU_MEMORY_BUFFER = 1 << 6;

/// OpenCL_MEMORY_IMAGE
const int MNN_GPU_MEMORY_IMAGE = 1 << 7;

// For the OpenCL backend, the following two options are effective only on
// Qualcomm GPUs. When using a Qualcomm GPU, the user could try both options
// and choose the better one based on performance.
// For the Vulkan backend, only option MNN_GPU_RECORD_BATCH is valid. When
// MNN_GPU_RECORD_BATCH is enabled, all ops would share one commandBuffer.
/// The kernels in one op execution record into one recording.(OpenCL)
const int MNN_GPU_RECORD_OP = 1 << 8;

/// 10 kernels record into one recording.(OpenCL) All ops share one commandBuffer.(Vulkan)
const int MNN_GPU_RECORD_BATCH = 1 << 9;

const int MNN_MEMORY_NORMAL = 0;
const int MNN_MEMORY_HIGH = 1;
const int MNN_MEMORY_LOW = 2;

const int MNN_POWER_NORMAL = 0;
const int MNN_POWER_HIGH = 1;
const int MNN_POWER_LOW = 2;

const int MNN_PRECISION_NORMAL = 0;
const int MNN_PRECISION_HIGH = 1;
const int MNN_PRECISION_LOW = 2;
const int MNN_PRECISION_LOW_BF16 = 3;
