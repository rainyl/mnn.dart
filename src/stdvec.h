//
// Created by rainy on 2024/12/16.
//

#ifndef STDVEC_H
#define STDVEC_H

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include "mnn_type.h"

#ifdef __cplusplus
#include <vector>
extern "C" {

#define CVD_STD_VEC_FUNC_IMPL(TYPE, ELEM)                                                          \
  CVD_STD_VEC_FUNC_IMPL_COMMON(TYPE);                                                              \
  TYPE std_##TYPE##_new(size_t length) { return new std::vector<ELEM>(length); }                   \
  TYPE std_##TYPE##_new_1(size_t length, ELEM val) { return new std::vector<ELEM>(length, val); }  \
  TYPE std_##TYPE##_new_2(size_t length, ELEM *val_ptr) {                                          \
    return new std::vector<ELEM>(val_ptr, val_ptr + length);                                       \
  }                                                                                                \
  void std_##TYPE##_push_back(TYPE self, ELEM val) { self->push_back(val); }                       \
  ELEM std_##TYPE##_get(TYPE self, size_t index) { return self->at(index); }                       \
  void std_##TYPE##_set(TYPE self, size_t index, ELEM val) { self->at(index) = val; }              \
  ELEM *std_##TYPE##_data(TYPE self) { return self->data(); }                                      \
  TYPE std_##TYPE##_clone(TYPE self) { return new std::vector<ELEM>(*(self)); }

#define CVD_STD_VEC_FUNC_IMPL_COMMON(TYPE)                                                         \
  void std_##TYPE##_free(TYPE self) { delete self; }                                               \
  size_t std_##TYPE##_length(TYPE self) { return self->size(); }                                   \
  void std_##TYPE##_resize(TYPE self, size_t new_len) { self->resize(new_len); }                   \
  void std_##TYPE##_reserve(TYPE self, size_t new_len) { self->reserve(new_len); }                 \
  void std_##TYPE##_clear(TYPE self) { self->clear(); }                                            \
  void std_##TYPE##_shrink_to_fit(TYPE self) { self->shrink_to_fit(); }                            \
  void std_##TYPE##_extend(TYPE self, TYPE other) {                                                \
    self->insert(self->end(), other->begin(), other->end());                                       \
  }
#endif

#define CVD_STD_VEC_FUNC_DEF(TYPE, ELEM)                                                           \
  MNN_C_API TYPE std_##TYPE##_new(size_t length);                                                            \
  MNN_C_API TYPE std_##TYPE##_new_1(size_t length, ELEM val);                                                \
  MNN_C_API TYPE std_##TYPE##_new_2(size_t length, ELEM *val_ptr);                                           \
  MNN_C_API void std_##TYPE##_free(TYPE self);                                                               \
  MNN_C_API void std_##TYPE##_push_back(TYPE self, ELEM val);                                                \
  MNN_C_API ELEM std_##TYPE##_get(TYPE self, size_t index);                                                  \
  MNN_C_API void std_##TYPE##_set(TYPE self, size_t index, ELEM val);                                        \
  MNN_C_API size_t std_##TYPE##_length(TYPE self);                                                           \
  MNN_C_API ELEM *std_##TYPE##_data(TYPE self);                                                              \
  MNN_C_API void std_##TYPE##_resize(TYPE self, size_t new_len);                                             \
  MNN_C_API void std_##TYPE##_reserve(TYPE self, size_t new_len);                                            \
  MNN_C_API void std_##TYPE##_clear(TYPE self);                                                              \
  MNN_C_API void std_##TYPE##_shrink_to_fit(TYPE self);                                                      \
  MNN_C_API void std_##TYPE##_extend(TYPE self, TYPE other);                                                 \
  MNN_C_API TYPE std_##TYPE##_clone(TYPE self);

#ifdef __cplusplus
#define CVD_TYPEDEF_STD_VEC(TYPE, NAME) typedef std::vector<TYPE> *NAME
#else
#define CVD_TYPEDEF_STD_VEC(TYPE, NAME) typedef void *NAME
#endif

typedef unsigned char uchar;
CVD_TYPEDEF_STD_VEC(uchar, VecUChar);
CVD_TYPEDEF_STD_VEC(char, VecChar);
CVD_TYPEDEF_STD_VEC(uint8_t, VecU8);
CVD_TYPEDEF_STD_VEC(int8_t, VecI8);
CVD_TYPEDEF_STD_VEC(uint16_t, VecU16);
CVD_TYPEDEF_STD_VEC(int16_t, VecI16);
CVD_TYPEDEF_STD_VEC(uint32_t, VecU32);
CVD_TYPEDEF_STD_VEC(int32_t, VecI32);
CVD_TYPEDEF_STD_VEC(int64_t, VecI64);
CVD_TYPEDEF_STD_VEC(uint64_t, VecU64);
CVD_TYPEDEF_STD_VEC(float_t, VecF32);
CVD_TYPEDEF_STD_VEC(double_t, VecF64);
CVD_TYPEDEF_STD_VEC(uint16_t, VecF16); // TODO: change to native float16

CVD_STD_VEC_FUNC_DEF(VecU8, uint8_t);
CVD_STD_VEC_FUNC_DEF(VecI8, int8_t);
CVD_STD_VEC_FUNC_DEF(VecU16, uint16_t);
CVD_STD_VEC_FUNC_DEF(VecI16, int16_t);
CVD_STD_VEC_FUNC_DEF(VecU32, uint32_t);
CVD_STD_VEC_FUNC_DEF(VecI32, int32_t);
CVD_STD_VEC_FUNC_DEF(VecU64, uint64_t);
CVD_STD_VEC_FUNC_DEF(VecI64, int64_t);
CVD_STD_VEC_FUNC_DEF(VecF32, float_t);
CVD_STD_VEC_FUNC_DEF(VecF64, double_t);
CVD_STD_VEC_FUNC_DEF(VecF16, uint16_t);

// CVD_STD_VEC_FUNC_DEF(VecVecChar, VecChar);
// VecVecChar* std_VecVecChar_new(size_t length);
// VecVecChar* std_VecVecChar_new_1(size_t length, VecChar* val);
// VecVecChar* std_VecVecChar_new_2(VecVecChar* val_ptr);
// VecVecChar* std_VecVecChar_new_3(char** val, VecI32 sizes);
// void std_VecVecChar_free(VecVecChar* self);
// size_t std_VecVecChar_length(VecVecChar* self);
// void std_VecVecChar_push_back(VecVecChar* self, VecChar val);
// VecChar* std_VecVecChar_get(VecVecChar* self, int index);
// void std_VecVecChar_set(VecVecChar* self, int index, VecChar* val);
// VecChar* std_VecVecChar_data(VecVecChar* self);
// void std_VecVecChar_reserve(VecVecChar* self, size_t size);
// void std_VecVecChar_resize(VecVecChar* self, size_t size);
// void std_VecVecChar_clear(VecVecChar* self);
// void std_VecVecChar_shrink_to_fit(VecVecChar* self);
// VecVecChar* std_VecVecChar_clone(VecVecChar* self);

#ifdef __cplusplus
}
#endif

#endif // STDVEC_H
