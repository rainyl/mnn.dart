//
// Created by rainy on 2025/9/15.
//

#include "base.h"
#include <cstdlib>

void *dart_malloc(size_t size) {
#if WIN32
  return CoTaskMemAlloc(size);
#else
  return malloc(size);
#endif
}

void dart_free(void *ptr) {
#if WIN32
  return CoTaskMemFree(ptr);
#else
  return free(ptr);
#endif
}
