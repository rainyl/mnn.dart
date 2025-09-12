#ifndef MNN_EXPR_H
#define MNN_EXPR_H

#include "error_code.h"
#include "mnn_type.h"
#include "stdvec.h"
#include "tensor.h"

#ifdef __cplusplus
#include "MNN/expr/Expr.hpp"
extern "C" {
#endif

/** Opaque pointer types */
#ifdef __cplusplus
typedef MNN::Express::VARP *mnn_expr_VARP_t;
typedef MNN::Express::EXPRP *mnn_expr_EXPRP_t;
typedef MNN::Express::Variable *mnn_expr_Variable_t;
typedef std::shared_ptr<MNN::Express::Variable> *mnn_expr_VariableP_t;
typedef MNN::OpT *mnn_OpT_t;
typedef MNN::Op *mnn_Op_t;
typedef std::map<std::string, MNN::Express::VARP> *mnn_expr_VARMAP_t;
typedef std::pair<
    std::map<std::string, MNN::Express::VARP>,
    std::map<std::string, MNN::Express::VARP>> *mnn_expr_VARMAP_PAIR_t;
typedef MNN::NetT *mnn_net_t;
#else
typedef void *mnn_expr_VARP_t;
typedef void *mnn_expr_EXPRP_t;
typedef void *mnn_expr_Variable_t;
typedef void *mnn_expr_VariableP_t;
typedef void *mnn_expr_Expr_t;
typedef void *mnn_OpT_t;
typedef void *mnn_Op_t;
typedef void *mnn_expr_VARMAP_t;
typedef void *mnn_expr_VARMAP_PAIR_t;
typedef void *mnn_net_t;
#endif

// typedef bool(*mnn_expr_Expr_visit_callback)(mnn_expr_EXPRP_t, int);
typedef struct VecVARP {
#ifdef __cplusplus
  MNN::Express::VARP *ptr;
#else
  void *ptr;
#endif
  size_t size;
} VecVARP;

typedef struct VecWeakEXPRP {
#ifdef __cplusplus
  std::weak_ptr<MNN::Express::Expr> *ptr;
#else
  void *ptr;
#endif
  size_t size;
} VecWeakEXPRP;

MNN_C_API void mnn_expr_VecVARP_free(void *self);
MNN_C_API void mnn_expr_VecWeakEXPRP_free(void *self);

// MNN::Express::VARP
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_create_empty();
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_create_variable(mnn_expr_Variable_t var);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_create_variableP(mnn_expr_VariableP_t var);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_create_VARP(mnn_expr_VARP_t other);
MNN_C_API void mnn_expr_VARP_free(mnn_expr_VARP_t self);
MNN_C_API mnn_expr_Variable_t mnn_expr_VARP_get(mnn_expr_VARP_t self);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_op_add(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_op_sub(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_op_mul(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_op_div(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_mean(mnn_expr_VARP_t self, VecI32 dims);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_sum(mnn_expr_VARP_t self, VecI32 dims);
MNN_C_API bool mnn_expr_VARP_op_eqeq(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_less(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_lessequal(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API void mnn_expr_VARP_op_assign(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API void mnn_expr_VARP_op_assign_variable(mnn_expr_VARP_t self, mnn_expr_Variable_t other);
MNN_C_API bool mnn_expr_VARP_fix(mnn_expr_VARP_t self, int type);        // InputType type
MNN_C_API void mnn_expr_VARP_setOrder(mnn_expr_VARP_t self, int format); // Dimensionformat format
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_VARP_getInfo(mnn_expr_VARP_t self);
MNN_C_API const void *mnn_expr_VARP_readMap(mnn_expr_VARP_t self);
MNN_C_API void *mnn_expr_VARP_writeMap(mnn_expr_VARP_t self);
MNN_C_API void mnn_expr_VARP_unMap(mnn_expr_VARP_t self);

// MNN::Express::Variable
MNN_C_API void mnn_expr_Variable_free(mnn_expr_Variable_t self);
MNN_C_API void mnn_expr_Variable_setName(mnn_expr_Variable_t self, char *name);
MNN_C_API char *mnn_expr_Variable_getName(mnn_expr_Variable_t self);
MNN_C_API bool
mnn_expr_Variable_setDevicePtr(mnn_expr_Variable_t self, const void *devicePtr, int memoryType);
MNN_C_API bool
mnn_expr_Variable_copyToDevicePtr(mnn_expr_Variable_t self, void *devicePtr, int memoryType);
struct mnn_expr_Variable_expr_pair {
  mnn_expr_EXPRP_t expr;
  int index;
};
MNN_C_API struct mnn_expr_Variable_expr_pair *mnn_expr_Variable_getExpr(mnn_expr_Variable_t self);
struct mnn_expr_Variable_Info {
  int order;
  int32_t *dim;
  size_t ndim;
  halide_type_c_t type;
  size_t size;
};
MNN_C_API void mnn_expr_Variable_Info_free(void *self);
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_Variable_getInfo(mnn_expr_Variable_t self);
MNN_C_API bool mnn_expr_Variable_resize(mnn_expr_Variable_t self, VecI32 dims);
MNN_C_API const void *mnn_expr_Variable_readMap(mnn_expr_Variable_t self);
MNN_C_API void *mnn_expr_Variable_writeMap(mnn_expr_Variable_t self);
MNN_C_API void
mnn_expr_Variable_writeScaleMap(mnn_expr_Variable_t self, float scaleValue, float zeroPoint);
MNN_C_API void mnn_expr_Variable_unMap(mnn_expr_Variable_t self);
MNN_C_API bool mnn_expr_Variable_input(mnn_expr_Variable_t self, mnn_expr_VARP_t src);
MNN_C_API void mnn_expr_Variable_static_replace(mnn_expr_VARP_t dst, mnn_expr_VARP_t src);
MNN_C_API mnn_expr_VARP_t mnn_expr_Variable_static_create_EXPRP(mnn_expr_EXPRP_t expr, int index);

MNN_C_API VecVARP *mnn_expr_Variable_static_load(const char *fileName);
MNN_C_API mnn_expr_VARMAP_t mnn_expr_Variable_static_loadMap(const char *fileName);
MNN_C_API VecVARP *mnn_expr_Variable_static_loadBuffer(const uint8_t *buffer, size_t length);
MNN_C_API mnn_expr_VARMAP_t
mnn_expr_Variable_static_loadMapBuffer(const uint8_t *buffer, size_t length);
// MNN_C_API void mnn_expr_Variable_static_getExecuteOrder(VecVARP output);
MNN_C_API void mnn_expr_Variable_static_save(VecVARP vars, const char *fileName);
MNN_C_API VecI8 mnn_expr_Variable_static_saveBytes(VecVARP vars);
MNN_C_API void mnn_expr_Variable_static_saveNet(VecVARP vars, mnn_net_t dest);

MNN_C_API void mnn_expr_Variable_static_prepareCompute(VecVARP vars, bool forceCPU);
MNN_C_API void mnn_expr_Variable_static_compute(VecVARP vars, bool forceCPU);
MNN_C_API size_t mnn_expr_Variable_linkNumber(mnn_expr_Variable_t self);
MNN_C_API const VecWeakEXPRP *mnn_expr_Variable_toExprs(mnn_expr_Variable_t self);
MNN_C_API void
mnn_expr_Variable_setExpr(mnn_expr_Variable_t self, mnn_expr_EXPRP_t expr, int index);
MNN_C_API const mnn_tensor_t mnn_expr_Variable_getTensor(mnn_expr_Variable_t self);

// MNN::Express::Expr
MNN_C_API mnn_expr_EXPRP_t mnn_expr_Expr_create_empty();
MNN_C_API mnn_expr_EXPRP_t mnn_expr_Expr_static_create(mnn_tensor_t tensor, bool own);
MNN_C_API mnn_expr_EXPRP_t mnn_expr_Expr_static_create_1(
    struct mnn_expr_Variable_Info *info, const void *ptr, int type, int memoryType
);
MNN_C_API mnn_expr_EXPRP_t
mnn_expr_Expr_static_create_2(mnn_OpT_t op, VecVARP inputs, int outputSize);
MNN_C_API void mnn_expr_Expr_free(mnn_expr_EXPRP_t self);

MNN_C_API void mnn_expr_Expr_setName(mnn_expr_EXPRP_t self, const char *name);
MNN_C_API const mnn_Op_t mnn_expr_Expr_getOp(mnn_expr_EXPRP_t self);
MNN_C_API VecVARP *mnn_expr_Expr_getInputs(mnn_expr_EXPRP_t self);
MNN_C_API int mnn_expr_Expr_getOutputSize(mnn_expr_EXPRP_t self);
MNN_C_API void mnn_expr_Expr_static_replace(mnn_expr_EXPRP_t oldExpr, mnn_expr_EXPRP_t newExpr);
MNN_C_API bool mnn_expr_Expr_requireInfo(mnn_expr_EXPRP_t self);
MNN_C_API const char *mnn_expr_Expr_getName(mnn_expr_EXPRP_t self);
MNN_C_API const char *mnn_expr_Expr_getOutputName(mnn_expr_EXPRP_t self, int index);
MNN_C_API int mnn_expr_Expr_inputType(mnn_expr_EXPRP_t self);
// MNN_C_API void mnn_expr_Expr_visitOutputs();
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_Expr_outputInfo(mnn_expr_EXPRP_t self, int index);

#ifdef __cplusplus
}
#endif

#endif // MNN_EXPR_H
