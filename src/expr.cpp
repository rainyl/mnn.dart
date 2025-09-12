//
// Created by rainy on 2025/9/10.
//

#include "expr.h"
#include "MNN/expr/Expr.hpp"
#include <vector>

MNN_C_API void mnn_expr_VecVARP_free(void *self) {
  if (self != nullptr) {
    delete static_cast<VecVARP *>(self)->ptr;
    delete static_cast<VecVARP *>(self);
  }
}
MNN_C_API void mnn_expr_VecWeakEXPRP_free(void *self) {
  if (self != nullptr) {
    delete static_cast<VecWeakEXPRP *>(self)->ptr;
    delete static_cast<VecWeakEXPRP *>(self);
  }
}

mnn_expr_VARP_t mnn_expr_VARP_create_empty() { return new MNN::Express::VARP(); }
mnn_expr_VARP_t mnn_expr_VARP_create_variable(mnn_expr_Variable_t var) {
  return new MNN::Express::VARP(var);
}
mnn_expr_VARP_t mnn_expr_VARP_create_variableP(mnn_expr_VariableP_t var) {
  return new MNN::Express::VARP(*var);
}
mnn_expr_VARP_t mnn_expr_VARP_create_VARP(mnn_expr_VARP_t other) {
  return new MNN::Express::VARP(*other);
}
void mnn_expr_VARP_free(mnn_expr_VARP_t self) { delete self; }
mnn_expr_Variable_t mnn_expr_VARP_get(mnn_expr_VARP_t self) { return self->get(); }
mnn_expr_VARP_t mnn_expr_VARP_op_add(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return new MNN::Express::VARP((*self) + (*other));
}
mnn_expr_VARP_t mnn_expr_VARP_op_sub(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return new MNN::Express::VARP((*self) - (*other));
}
mnn_expr_VARP_t mnn_expr_VARP_op_mul(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return new MNN::Express::VARP((*self) * (*other));
}
mnn_expr_VARP_t mnn_expr_VARP_op_div(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return new MNN::Express::VARP((*self) / (*other));
}
mnn_expr_VARP_t mnn_expr_VARP_mean(mnn_expr_VARP_t self, VecI32 dims) {
  return new MNN::Express::VARP(self->mean(*dims));
}
mnn_expr_VARP_t mnn_expr_VARP_sum(mnn_expr_VARP_t self, VecI32 dims) {
  return new MNN::Express::VARP(self->sum(*dims));
}
bool mnn_expr_VARP_op_eqeq(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return (*self) == (*other);
}
bool mnn_expr_VARP_op_less(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return (*self) < (*other);
}
bool mnn_expr_VARP_op_lessequal(mnn_expr_VARP_t self, mnn_expr_VARP_t other) {
  return (*self) <= (*other);
}
void mnn_expr_VARP_op_assign(mnn_expr_VARP_t self, mnn_expr_VARP_t other) { (*self) = (*other); }
void mnn_expr_VARP_op_assign_variable(mnn_expr_VARP_t self, mnn_expr_Variable_t other) {
  (*self) = other;
}
bool mnn_expr_VARP_fix(mnn_expr_VARP_t self, int type) {
  return self->fix(static_cast<MNN::Express::VARP::InputType>(type));
}
void mnn_expr_VARP_setOrder(mnn_expr_VARP_t self, int format) {
  self->setOrder(static_cast<MNN::Express::Dimensionformat>(format));
}

struct mnn_expr_Variable_Info *mnn_expr_VARP_getInfo(mnn_expr_VARP_t self) {
  auto _info = self->get()->getInfo();
  auto info = new mnn_expr_Variable_Info{};
  info->order = static_cast<int>(_info->order);
  info->ndim = _info->dim.size();
  auto pdim = static_cast<int32_t *>(malloc(info->ndim * sizeof(int32_t)));
  for (int i = 0; i < info->ndim; i++) { pdim[i] = _info->dim[i]; }
  info->dim = pdim;
  info->type = {static_cast<uint8_t>(_info->type.code), _info->type.bits, _info->type.lanes};
  info->size = _info->size;
  return info;
}

MNN_C_API const void *mnn_expr_VARP_readMap(mnn_expr_VARP_t self) {
  return self->get()->readMap<void>();
}
MNN_C_API void *mnn_expr_VARP_writeMap(mnn_expr_VARP_t self) {
  return self->get()->writeMap<void>();
}
MNN_C_API void mnn_expr_VARP_unMap(mnn_expr_VARP_t self) { self->get()->unMap(); }

// MNN::Express::Variable
void mnn_expr_Variable_free(mnn_expr_Variable_t self) { delete self; }
void mnn_expr_Variable_setName(mnn_expr_Variable_t self, char *name) { self->setName(name); }
char *mnn_expr_Variable_getName(mnn_expr_Variable_t self) { return strdup(self->name().c_str()); }
bool mnn_expr_Variable_setDevicePtr(
    mnn_expr_Variable_t self, const void *devicePtr, int memoryType
) {
  return self->setDevicePtr(devicePtr, memoryType);
}
bool mnn_expr_Variable_copyToDevicePtr(mnn_expr_Variable_t self, void *devicePtr, int memoryType) {
  return self->copyToDevicePtr(devicePtr, memoryType);
}

struct mnn_expr_Variable_expr_pair *mnn_expr_Variable_getExpr(mnn_expr_Variable_t self) {
  auto _expr = self->expr();
  return new mnn_expr_Variable_expr_pair{
      .expr = new MNN::Express::EXPRP{_expr.first}, .index = _expr.second
  };
}

MNN_C_API void mnn_expr_Variable_Info_free(void *self) {
  auto p = static_cast<mnn_expr_Variable_Info *>(self);
  free(p->dim);
  delete p;
}

mnn_expr_Variable_Info *mnn_expr_Variable_getInfo(mnn_expr_Variable_t self) {
  auto _info = self->getInfo();
  auto info = new mnn_expr_Variable_Info{};
  info->order = static_cast<int>(_info->order);
  info->ndim = _info->dim.size();
  auto pdim = static_cast<int32_t *>(malloc(info->ndim * sizeof(int32_t)));
  for (int i = 0; i < info->ndim; i++) { pdim[i] = _info->dim[i]; }
  info->dim = pdim;
  info->type = {static_cast<uint8_t>(_info->type.code), _info->type.bits, _info->type.lanes};
  info->size = _info->size;
  return info;
}
bool mnn_expr_Variable_resize(mnn_expr_Variable_t self, VecI32 dims) { return self->resize(*dims); }
const void *mnn_expr_Variable_readMap(mnn_expr_Variable_t self) { return self->readMap<void>(); }
void *mnn_expr_Variable_writeMap(mnn_expr_Variable_t self) { return self->writeMap<void>(); }
void mnn_expr_Variable_writeScaleMap(mnn_expr_Variable_t self, float scaleValue, float zeroPoint) {
  self->writeScaleMap(scaleValue, zeroPoint);
}
void mnn_expr_Variable_unMap(mnn_expr_Variable_t self) { self->unMap(); }
bool mnn_expr_Variable_input(mnn_expr_Variable_t self, mnn_expr_VARP_t src) {
  return self->input(*src);
}
void mnn_expr_Variable_static_replace(mnn_expr_VARP_t dst, mnn_expr_VARP_t src) {
  MNN::Express::Variable::replace(*dst, *src);
}
mnn_expr_VARP_t mnn_expr_Variable_static_create_EXPRP(mnn_expr_EXPRP_t expr, int index) {
  return new MNN::Express::VARP(MNN::Express::Variable::create(*expr, index));
}
VecVARP *mnn_expr_Variable_static_load(const char *fileName) {
  auto p = new MNN::Express::VARPS(MNN::Express::Variable::load(fileName));
  return new VecVARP{p->data(), p->size()};
}

mnn_expr_VARMAP_t mnn_expr_Variable_static_loadMap(const char *fileName) {
  return new std::map<std::string, MNN::Express::VARP>(MNN::Express::Variable::loadMap(fileName));
}

VecVARP *mnn_expr_Variable_static_loadBuffer(const uint8_t *buffer, size_t length) {
  auto p = new MNN::Express::VARPS(MNN::Express::Variable::load(buffer, length));
  return new VecVARP{p->data(), p->size()};
}

mnn_expr_VARMAP_t mnn_expr_Variable_static_loadMapBuffer(const uint8_t *buffer, size_t length) {
  return new std::map<std::string, MNN::Express::VARP>(
      MNN::Express::Variable::loadMap(buffer, length)
  );
}

void mnn_expr_Variable_static_save(VecVARP vars, const char *fileName) {
  MNN::Express::Variable::save(MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size), fileName);
}

VecI8 mnn_expr_Variable_static_saveBytes(VecVARP vars) {
  return new std::vector<int8_t>(
      MNN::Express::Variable::save(MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size))
  );
}

void mnn_expr_Variable_static_saveNet(VecVARP vars, mnn_net_t dest) {
  MNN::Express::Variable::save(MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size), dest);
}

void mnn_expr_Variable_static_prepareCompute(VecVARP vars, bool forceCPU) {
  MNN::Express::Variable::prepareCompute(
      MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size), forceCPU
  );
}
void mnn_expr_Variable_static_compute(VecVARP vars, bool forceCPU) {
  MNN::Express::Variable::compute(MNN::Express::VARPS(vars.ptr, vars.ptr + vars.size), forceCPU);
}
size_t mnn_expr_Variable_linkNumber(mnn_expr_Variable_t self) { return self->linkNumber(); }
const VecWeakEXPRP *mnn_expr_Variable_toExprs(mnn_expr_Variable_t self) {
  auto p = new std::vector<MNN::Express::WeakEXPRP>(self->toExprs());
  return new VecWeakEXPRP{p->data(), p->size()};
}
void mnn_expr_Variable_setExpr(mnn_expr_Variable_t self, mnn_expr_EXPRP_t expr, int index) {
  self->setExpr(*expr, index);
}
const mnn_tensor_t mnn_expr_Variable_getTensor(mnn_expr_Variable_t self) {
  return new MNN::Tensor(self->getTensor());
}

mnn_expr_EXPRP_t mnn_expr_Expr_create_empty() { return new std::shared_ptr<MNN::Express::Expr>(); }
mnn_expr_EXPRP_t mnn_expr_Expr_static_create(mnn_tensor_t tensor, bool own) {
  return new MNN::Express::EXPRP(MNN::Express::Expr::create(tensor, own));
}
mnn_expr_EXPRP_t mnn_expr_Expr_static_create_1(
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
mnn_expr_EXPRP_t mnn_expr_Expr_static_create_2(mnn_OpT_t op, VecVARP inputs, int outputSize) {
  return new MNN::Express::EXPRP(
      MNN::Express::Expr::create(
          op, MNN::Express::VARPS(inputs.ptr, inputs.ptr + inputs.size), outputSize
      )
  );
}

void mnn_expr_Expr_free(mnn_expr_EXPRP_t self) { delete self; }

void mnn_expr_Expr_setName(mnn_expr_EXPRP_t self, const char *name) { self->get()->setName(name); }
const char *mnn_expr_Expr_getName(mnn_expr_EXPRP_t self) {
  return strdup(self->get()->name().c_str());
}
const mnn_Op_t mnn_expr_Expr_getOp(mnn_expr_EXPRP_t self) {
  return const_cast<mnn_Op_t>(self->get()->get());
}
VecVARP *mnn_expr_Expr_getInputs(mnn_expr_EXPRP_t self) {
  auto p = new std::vector<MNN::Express::VARP>(self->get()->inputs());
  return new VecVARP{p->data(), p->size()};
}
int mnn_expr_Expr_getOutputSize(mnn_expr_EXPRP_t self) { return self->get()->outputSize(); }
const char *mnn_expr_Expr_getOutputName(mnn_expr_EXPRP_t self, int index) {
  return strdup(self->get()->outputName(index).c_str());
}
void mnn_expr_Expr_static_replace(mnn_expr_EXPRP_t oldExpr, mnn_expr_EXPRP_t newExpr) {
  MNN::Express::Expr::replace(*oldExpr, *newExpr);
}
bool mnn_expr_Expr_requireInfo(mnn_expr_EXPRP_t self) { return self->get()->requireInfo(); }

int mnn_expr_Expr_inputType(mnn_expr_EXPRP_t self) {
  return static_cast<int>(self->get()->inputType());
}

MNN_C_API struct mnn_expr_Variable_Info *
mnn_expr_Expr_outputInfo(mnn_expr_EXPRP_t self, int index) {
  auto _info = self->get()->outputInfo(index);
  auto info = new mnn_expr_Variable_Info{};
  info->order = static_cast<int>(_info->order);
  info->ndim = _info->dim.size();
  auto pdim = static_cast<int32_t *>(malloc(info->ndim * sizeof(int32_t)));
  for (int i = 0; i < info->ndim; i++) { pdim[i] = _info->dim[i]; }
  info->dim = pdim;
  info->type = {static_cast<uint8_t>(_info->type.code), _info->type.bits, _info->type.lanes};
  info->size = _info->size;
  return info;
}
