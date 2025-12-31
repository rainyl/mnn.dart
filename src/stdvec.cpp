//
// Created by rainy on 2024/12/16.
//

#include "stdvec.h"

#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

CVD_STD_VEC_FUNC_IMPL(VecU8, uint8_t);
CVD_STD_VEC_FUNC_IMPL(VecI8, int8_t);
CVD_STD_VEC_FUNC_IMPL(VecU16, uint16_t);
CVD_STD_VEC_FUNC_IMPL(VecI16, int16_t);
CVD_STD_VEC_FUNC_IMPL(VecU32, uint32_t);
CVD_STD_VEC_FUNC_IMPL(VecI32, int32_t);
CVD_STD_VEC_FUNC_IMPL(VecU64, uint64_t);
CVD_STD_VEC_FUNC_IMPL(VecI64, int64_t);
CVD_STD_VEC_FUNC_IMPL(VecF32, float_t);
CVD_STD_VEC_FUNC_IMPL(VecF64, double_t);
CVD_STD_VEC_FUNC_IMPL(VecF16, uint16_t);

// CVD_STD_VEC_FUNC_IMPL_COMMON(VecVecChar)
// VecVecChar* std_VecVecChar_new(size_t length) {
//     return new VecVecChar{new std::vector<std::vector<char>>(length)};
// }
//
// VecVecChar* std_VecVecChar_new_1(size_t length, VecChar* val) {
//     const auto v = static_cast<std::vector<char>*>(val);
//     return new VecVecChar{new std::vector<std::vector<char>>(length, *v)};
// }
//
// VecVecChar* std_VecVecChar_new_2(VecVecChar* val_ptr) {
//     const auto v = static_cast<std::vector<std::vector<char>>*>(val_ptr);
//     return new VecVecChar{new std::vector<std::vector<char>>(*v)};
// }
//
// VecVecChar* std_VecVecChar_new_3(char** val, VecI32 sizes) {
//     if (!sizes.ptr || sizes.ptr->empty()) {
//         return new VecVecChar{new std::vector<std::vector<char>>()};
//     }
//     auto rv = new std::vector<std::vector<char>>(sizes.ptr->size());
//     for (size_t i = 0; i < sizes.ptr->size(); i++) {
//         auto t = std::vector<char>(val[i], val[i] + sizes.ptr->at(i));
//         rv->push_back(t);
//     }
//     return new VecVecChar{rv};
// }
//
// void std_VecVecChar_push_back(VecVecChar self, VecChar val) {
//     const auto v = static_cast<std::vector<char>*>(val.ptr);
//     self->push_back(*v);
// }
//
// VecChar* std_VecVecChar_get(VecVecChar self, int index) {
//     return new VecChar{new std::vector<char>(self->at(index))};
// }
//
// void std_VecVecChar_set(VecVecChar self, int index, VecChar* val) {
//     self->at(index) = *(val);
// }
//
// VecChar* std_VecVecChar_data(VecVecChar self) {
//     return new VecChar{self->data()};
// }
//
// VecVecChar* std_VecVecChar_clone(VecVecChar self) {
//     return new VecVecChar{new std::vector<std::vector<char>>(*self)};
// }
