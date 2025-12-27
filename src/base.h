//
// Created by rainy on 2025/9/15.
//

#ifndef MNN_C_API_BASE_H
#define MNN_C_API_BASE_H

#include "mnn_type.h"

#include <stddef.h>

#if WIN32
#include <Windows.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

MNN_C_API void *dart_malloc(size_t size);

MNN_C_API void dart_free(void *ptr);

#ifdef __cplusplus
}
#endif

#endif // MNN_C_API_BASE_H
