#define _GNU_SOURCE
//
// Created by rainy on 2025/9/10.
//

#include "expr.h"

#include "MNN/expr/Expr.hpp"
#include "base.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

VecVARP_t mnn_expr_VecVARP_create(size_t length, VARP_t value) {
  if (value) return new std::vector<MNN::Express::VARP>(length, *value);
  return new std::vector<MNN::Express::VARP>(length);
}

void mnn_expr_VecVARP_free(VecVARP_t self) {
  if (self == nullptr) return;
  delete static_cast<VecVARP_t>(self);
  self = nullptr;
}
VARP_t mnn_expr_VecVARP_at(VecVARP_t self, int i) { return new MNN::Express::VARP(self->at(i)); }
VARP_t mnn_expr_VecVARP_at_ref(VecVARP_t self, int i) { return &self->at(i); }
void mnn_expr_VecVARP_set(VecVARP_t self, int i, VARP_t value) { self->at(i) = *value; }
void mnn_expr_VecVARP_push_back(VecVARP_t self, VARP_t value) { return self->push_back(*value); }
size_t mnn_expr_VecVARP_size(VecVARP_t self) { return self->size(); }

void mnn_expr_VecWeakEXPRP_free(VecWeakEXPRP_t self) {
  if (self == nullptr) return;
  delete static_cast<VecWeakEXPRP_t>(self);
  self = nullptr;
}
MNN_C_API EXPRP_t mnn_expr_VecWeakEXPRP_at(VecWeakEXPRP_t self, int i) {
  auto _varp = self->at(i);
  if (_varp.expired()) return nullptr;
  return new MNN::Express::EXPRP(_varp.lock());
}
MNN_C_API void mnn_expr_VecWeakEXPRP_set(VecWeakEXPRP_t self, int i, EXPRP_t value) {
  self->at(i) = *value;
}
MNN_C_API void mnn_expr_VecWeakEXPRP_push_back(VecWeakEXPRP_t self, EXPRP_t value) {
  self->push_back(*value);
}
MNN_C_API size_t mnn_expr_VecWeakEXPRP_size(VecWeakEXPRP_t self) { return self->size(); }

void mnn_expr_Variable_Info_free(void *self) {
  auto p = static_cast<mnn_expr_Variable_Info *>(self);
  if (p != nullptr) {
    delete [] p->dim;
    delete p;
    p = nullptr;
  }
}

MNN_C_API VARMAP_t mnn_expr_VARMAP_create() {
  return new std::map<std::string, MNN::Express::VARP>();
}
MNN_C_API void mnn_expr_VARMAP_free(VARMAP_t self) {
  if (self == nullptr) return;
  delete static_cast<VARMAP_t>(self);
  self = nullptr;
}
MNN_C_API size_t mnn_expr_VARMAP_size(VARMAP_t self) { return self->size(); }
MNN_C_API char **mnn_expr_VARMAP_keys(VARMAP_t self) {
  char **keys = static_cast<char **>(malloc(self->size() * sizeof(char *)));
  int idx = 0;
  for (const auto &pair : *self) {
    keys[idx] = strdup(pair.first.c_str());
    idx++;
  }
  return keys;
}
MNN_C_API VARP_t mnn_expr_VARMAP_get(VARMAP_t self, char *key) {
  auto it = self->find(key);
  if (it != self->end()) return new MNN::Express::VARP(self->at(key));
  return nullptr;
}
MNN_C_API VARP_t mnn_expr_VARMAP_get_ref(VARMAP_t self, char *key) {
  auto it = self->find(key);
  if (it != self->end()) return &(*self)[key];
  return nullptr;
}
MNN_C_API void mnn_expr_VARMAP_set(VARMAP_t self, char *key, VARP_t value) {
  (*self)[key] = *value;
}

VARP_t mnn_expr_VARP_create_empty() { return new MNN::Express::VARP(); }
VARP_t mnn_expr_VARP_create_VARP(VARP_t other) { return new MNN::Express::VARP(*other); }
void mnn_expr_VARP_free(VARP_t self) {
  // std::cout << "Releasing VARP at " << self << std::endl;
  if (self == nullptr) return;
  delete self;
  self = nullptr;
}
VARP_t mnn_expr_VARP_op_add(VARP_t self, VARP_t other) {
  return new MNN::Express::VARP((*self) + (*other));
}
VARP_t mnn_expr_VARP_op_sub(VARP_t self, VARP_t other) {
  return new MNN::Express::VARP((*self) - (*other));
}
VARP_t mnn_expr_VARP_op_mul(VARP_t self, VARP_t other) {
  return new MNN::Express::VARP((*self) * (*other));
}
VARP_t mnn_expr_VARP_op_div(VARP_t self, VARP_t other) {
  return new MNN::Express::VARP((*self) / (*other));
}
VARP_t mnn_expr_VARP_mean(VARP_t self, VecI32 dims) {
  return new MNN::Express::VARP(self->mean(*dims));
}
VARP_t mnn_expr_VARP_sum(VARP_t self, VecI32 dims) {
  return new MNN::Express::VARP(self->sum(*dims));
}
bool mnn_expr_VARP_op_eqeq(VARP_t self, VARP_t other) { return (*self) == (*other); }
bool mnn_expr_VARP_op_less(VARP_t self, VARP_t other) { return (*self) < (*other); }
bool mnn_expr_VARP_op_lessequal(VARP_t self, VARP_t other) { return (*self) <= (*other); }
void mnn_expr_VARP_op_assign(VARP_t self, VARP_t other) { (*self) = (*other); }
bool mnn_expr_VARP_fix(VARP_t self, int type) {
  return self->fix(static_cast<MNN::Express::VARP::InputType>(type));
}
void mnn_expr_VARP_setOrder(VARP_t self, int format) {
  self->setOrder(static_cast<MNN::Express::Dimensionformat>(format));
}

struct mnn_expr_Variable_Info *mnn_expr_VARP_getInfo(VARP_t self) {
  if (self==nullptr) return nullptr;
  auto _info = (*self)->getInfo();
  if (_info == nullptr) { return nullptr; }

  auto info = new mnn_expr_Variable_Info();
  info->order = static_cast<int>(_info->order);
  info->ndim = _info->dim.size();
  auto pdim = new int[_info->dim.size()];
  memcpy(pdim, _info->dim.data(), sizeof(int) * _info->dim.size());
  info->dim = pdim;
  info->type = {static_cast<uint8_t>(_info->type.code), _info->type.bits, _info->type.lanes};
  info->size = _info->size;
  return info;
}

const void *mnn_expr_VARP_readMap(VARP_t self) { return (*self)->readMap<void>(); }
void *mnn_expr_VARP_writeMap(VARP_t self) { return (*self)->writeMap<void>(); }
void mnn_expr_VARP_unMap(VARP_t self) { (*self)->unMap(); }

// MNN::Express::Variable
void mnn_expr_VARP_setName(VARP_t self, char *name) { (*self)->setName(name); }
char *mnn_expr_VARP_getName(VARP_t self) { return strdup((*self)->name().c_str()); }
bool mnn_expr_VARP_setDevicePtr(VARP_t self, const void *devicePtr, int memoryType) {
  return (*self)->setDevicePtr(devicePtr, memoryType);
}
bool mnn_expr_VARP_copyToDevicePtr(VARP_t self, void *devicePtr, int memoryType) {
  return (*self)->copyToDevicePtr(devicePtr, memoryType);
}

struct Variable_expr_pair *mnn_expr_VARP_getExpr(VARP_t self) {
  auto _expr = (*self)->expr();
  auto pair = static_cast<Variable_expr_pair *>(malloc(sizeof(Variable_expr_pair)));
  pair->index = _expr.second;
  pair->expr = new MNN::Express::EXPRP{_expr.first};
  return pair;
}

bool mnn_expr_VARP_resize(VARP_t self, VecI32 dims) { return (*self)->resize(*dims); }
void mnn_expr_VARP_writeScaleMap(VARP_t self, float scaleValue, float zeroPoint) {
  (*self)->writeScaleMap(scaleValue, zeroPoint);
}
bool mnn_expr_VARP_input(VARP_t self, VARP_t src) { return (*self)->input(*src); }
void mnn_expr_VARP_static_replace(VARP_t dst, VARP_t src) {
  MNN::Express::Variable::replace(*dst, *src);
}
VARP_t mnn_expr_VARP_static_create_EXPRP(EXPRP_t expr, int index) {
  return new MNN::Express::VARP(MNN::Express::Variable::create(*expr, index));
}
VecVARP_t mnn_expr_VARP_static_load(const char *fileName) {
  auto v = MNN::Express::Variable::load(fileName);
  auto rval = new std::vector<MNN::Express::VARP>(v);
  return rval;
}

VARMAP_t mnn_expr_VARP_static_loadMap(const char *fileName) {
  auto varmap = MNN::Express::Variable::loadMap(fileName);
  return new std::map<std::string, MNN::Express::VARP>(varmap);
}

VecVARP_t mnn_expr_VARP_static_loadBuffer(const uint8_t *buffer, size_t length) {
  auto vec = MNN::Express::Variable::load(buffer, length);
  return new std::vector<MNN::Express::VARP>(vec);
}

VARMAP_t mnn_expr_VARP_static_loadMapBuffer(const uint8_t *buffer, size_t length) {
  auto varmap = MNN::Express::Variable::loadMap(buffer, length);
  return new std::map<std::string, MNN::Express::VARP>(varmap);
}

void mnn_expr_VARP_static_save(VecVARP_t vars, const char *fileName) {
  MNN::Express::Variable::save(*vars, fileName);
}

VecI8 mnn_expr_VARP_static_saveBytes(VecVARP_t vars) {
  return new std::vector<int8_t>(MNN::Express::Variable::save(*vars));
}

void mnn_expr_VARP_static_saveNet(VecVARP_t vars, Net_t dest) {
  MNN::Express::Variable::save(*vars, dest);
}

void mnn_expr_VARP_static_prepareCompute(VecVARP_t vars, bool forceCPU) {
  MNN::Express::Variable::prepareCompute(*vars, forceCPU);
}
void mnn_expr_VARP_static_compute(VecVARP_t vars, bool forceCPU) {
  MNN::Express::Variable::compute(*vars, forceCPU);
}
size_t mnn_expr_VARP_linkNumber(VARP_t self) { return (*self)->linkNumber(); }
const VecWeakEXPRP_t mnn_expr_VARP_toExprs(VARP_t self) {
  auto exprs = (*self)->toExprs();
  return new std::vector<std::weak_ptr<MNN::Express::Expr>>(exprs);
}
void mnn_expr_VARP_setExpr(VARP_t self, EXPRP_t expr, int index) { (*self)->setExpr(*expr, index); }
const mnn_tensor_t mnn_expr_VARP_getTensor(VARP_t self) {
  return new MNN::Tensor((*self)->getTensor());
}

EXPRP_t mnn_expr_Expr_create_empty() { return new std::shared_ptr<MNN::Express::Expr>(); }
EXPRP_t mnn_expr_Expr_static_create(mnn_tensor_t tensor, bool own) {
  return new MNN::Express::EXPRP(MNN::Express::Expr::create(tensor, own));
}
EXPRP_t mnn_expr_Expr_static_create_1(
    struct mnn_expr_Variable_Info *info, const void *ptr, int type, int memoryType
) {
  MNN::Express::Variable::Info _info;
  _info.order = static_cast<MNN::Express::Dimensionformat>(info->order);
  _info.dim = std::vector<int>(info->dim, info->dim + info->ndim);
  _info.type =
      halide_type_t((halide_type_code_t)info->type.code, info->type.bits, info->type.lanes);
  _info.size = info->size;
  return new MNN::Express::EXPRP(
      MNN::Express::Expr::create(
          std::move(_info),
          ptr,
          static_cast<MNN::Express::VARP::InputType>(type),
          static_cast<MNN::Express::Expr::MemoryType>(memoryType)
      )
  );
}
EXPRP_t mnn_expr_Expr_static_create_2(OpT_t op, VecVARP_t inputs, int outputSize) {
  return new MNN::Express::EXPRP(MNN::Express::Expr::create(op, *inputs, outputSize));
}

void mnn_expr_Expr_free(EXPRP_t self) {
  if (self == nullptr) return;
  delete self;
  self = nullptr;
}

void mnn_expr_Expr_setName(EXPRP_t self, const char *name) { self->get()->setName(name); }
const char *mnn_expr_Expr_getName(EXPRP_t self) { return strdup(self->get()->name().c_str()); }
const Op_t mnn_expr_Expr_getOp(EXPRP_t self) { return const_cast<Op_t>(self->get()->get()); }
VecVARP_t mnn_expr_Expr_getInputs(EXPRP_t self) {
  auto _inputs = self->get()->inputs();
  return new std::vector<MNN::Express::VARP>(_inputs);
}

VecWeakEXPRP_t mnn_expr_Expr_getOutputs(EXPRP_t self) {
  auto vec = self->get()->outputs();
  return new std::vector<std::weak_ptr<MNN::Express::Expr>>(vec);
}

int mnn_expr_Expr_getOutputSize(EXPRP_t self) { return self->get()->outputSize(); }
const char *mnn_expr_Expr_getOutputName(EXPRP_t self, int index) {
  return strdup(self->get()->outputName(index).c_str());
}
void mnn_expr_Expr_static_replace(EXPRP_t oldExpr, EXPRP_t newExpr) {
  MNN::Express::Expr::replace(*oldExpr, *newExpr);
}
bool mnn_expr_Expr_requireInfo(EXPRP_t self) { return self->get()->requireInfo(); }

int mnn_expr_Expr_inputType(EXPRP_t self) { return static_cast<int>(self->get()->inputType()); }

struct mnn_expr_Variable_Info *mnn_expr_Expr_outputInfo(EXPRP_t self, int index) {
  auto _info = self->get()->outputInfo(index);
  auto info = new mnn_expr_Variable_Info();
  info->order = static_cast<int>(_info->order);
  info->ndim = _info->dim.size();
  auto pdim = new int32_t[info->ndim];
  for (int i = 0; i < info->ndim; i++) { pdim[i] = _info->dim[i]; }
  info->dim = pdim;
  info->type = {static_cast<uint8_t>(_info->type.code), _info->type.bits, _info->type.lanes};
  info->size = _info->size;
  return info;
}
