#ifndef MNN_EXPR_H
#define MNN_EXPR_H

#include "base.h"
#include "mnn_type.h"
#include "stdvec.h"
#include "tensor.h"
#include <stddef.h>

#ifdef __cplusplus
#include "MNN/expr/Expr.hpp"
extern "C" {
#endif

/** Opaque pointer types */
#ifdef __cplusplus
typedef MNN::Express::VARP *VARP_t;
typedef MNN::Express::EXPRP *EXPRP_t;
typedef MNN::Express::VARPS *VecVARP_t;
typedef std::vector<MNN::Express::WeakEXPRP> *VecWeakEXPRP_t;
typedef MNN::OpT *OpT_t;
typedef MNN::Op *Op_t;
typedef std::pair<
    std::map<std::string, MNN::Express::VARP>,
    std::map<std::string, MNN::Express::VARP>> *VARMAP_PAIR_t;
typedef std::map<std::string, MNN::Express::VARP> *VARMAP_t;
typedef MNN::NetT *Net_t;
#else
typedef void *VARP_t;
typedef void *EXPRP_t;
typedef void *VecVARP_t;
typedef void *VecWeakEXPRP_t;
typedef void *mnn_expr_Expr_t;
typedef void *OpT_t;
typedef void *Op_t;
typedef void *VARMAP_PAIR_t;
typedef void *VARMAP_t;
typedef void *Net_t;
#endif

struct mnn_expr_Variable_Info {
  int order;
  int32_t *dim;
  size_t ndim;
  halide_type_c_t type;
  size_t size;
};

struct Variable_expr_pair {
  EXPRP_t expr;
  int index;
};

MNN_C_API VecVARP_t mnn_expr_VecVARP_create(size_t length, VARP_t value);
MNN_C_API void mnn_expr_VecVARP_free(void *self);
MNN_C_API VARP_t mnn_expr_VecVARP_at(VecVARP_t self, int i);
MNN_C_API VARP_t mnn_expr_VecVARP_at_ref(VecVARP_t self, int i);
MNN_C_API void mnn_expr_VecVARP_set(VecVARP_t self, int i, VARP_t value);
MNN_C_API void mnn_expr_VecVARP_push_back(VecVARP_t self, VARP_t value);
MNN_C_API size_t mnn_expr_VecVARP_size(VecVARP_t self);

MNN_C_API void mnn_expr_VecWeakEXPRP_free(void *self);
MNN_C_API EXPRP_t mnn_expr_VecWeakEXPRP_at(VecWeakEXPRP_t self, int i);
MNN_C_API void mnn_expr_VecWeakEXPRP_set(VecWeakEXPRP_t self, int i, EXPRP_t value);
MNN_C_API void mnn_expr_VecWeakEXPRP_push_back(VecWeakEXPRP_t self, EXPRP_t value);
MNN_C_API size_t mnn_expr_VecWeakEXPRP_size(VecWeakEXPRP_t self);

MNN_C_API void mnn_expr_Variable_Info_free(void *self);

MNN_C_API VARMAP_t mnn_expr_VARMAP_create();
MNN_C_API void mnn_expr_VARMAP_free(void *self);
MNN_C_API size_t mnn_expr_VARMAP_size(VARMAP_t self);
MNN_C_API char **mnn_expr_VARMAP_keys(VARMAP_t self);
MNN_C_API VARP_t mnn_expr_VARMAP_get(VARMAP_t self, char *key);
MNN_C_API VARP_t mnn_expr_VARMAP_get_ref(VARMAP_t self, char *key);
MNN_C_API void mnn_expr_VARMAP_set(VARMAP_t self, char *key, VARP_t value);

// MNN::Express::VARP
MNN_C_API VARP_t mnn_expr_VARP_create_empty();
MNN_C_API VARP_t mnn_expr_VARP_create_VARP(VARP_t other);
MNN_C_API void mnn_expr_VARP_free(void *self);
MNN_C_API VARP_t mnn_expr_VARP_op_add(VARP_t self, VARP_t other);
MNN_C_API VARP_t mnn_expr_VARP_op_sub(VARP_t self, VARP_t other);
MNN_C_API VARP_t mnn_expr_VARP_op_mul(VARP_t self, VARP_t other);
MNN_C_API VARP_t mnn_expr_VARP_op_div(VARP_t self, VARP_t other);
MNN_C_API VARP_t mnn_expr_VARP_mean(VARP_t self, VecI32 dims);
MNN_C_API VARP_t mnn_expr_VARP_sum(VARP_t self, VecI32 dims);
MNN_C_API bool mnn_expr_VARP_op_eqeq(VARP_t self, VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_less(VARP_t self, VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_lessequal(VARP_t self, VARP_t other);
MNN_C_API void mnn_expr_VARP_op_assign(VARP_t self, VARP_t other);
MNN_C_API bool mnn_expr_VARP_fix(VARP_t self, int type);        // InputType type
MNN_C_API void mnn_expr_VARP_setOrder(VARP_t self, int format); // Dimensionformat format
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_VARP_getInfo(VARP_t self);
MNN_C_API const void *mnn_expr_VARP_readMap(VARP_t self);
MNN_C_API void *mnn_expr_VARP_writeMap(VARP_t self);
MNN_C_API void mnn_expr_VARP_unMap(VARP_t self);

// MNN::Express::Variable
MNN_C_API void mnn_expr_VARP_setName(VARP_t self, char *name);
MNN_C_API char *mnn_expr_VARP_getName(VARP_t self);
MNN_C_API bool mnn_expr_VARP_setDevicePtr(VARP_t self, const void *devicePtr, int memoryType);
MNN_C_API bool mnn_expr_VARP_copyToDevicePtr(VARP_t self, void *devicePtr, int memoryType);
MNN_C_API struct Variable_expr_pair *mnn_expr_VARP_getExpr(VARP_t self);
MNN_C_API bool mnn_expr_VARP_resize(VARP_t self, VecI32 dims);
MNN_C_API void mnn_expr_VARP_writeScaleMap(VARP_t self, float scaleValue, float zeroPoint);
MNN_C_API bool mnn_expr_VARP_input(VARP_t self, VARP_t src);
MNN_C_API void mnn_expr_VARP_static_replace(VARP_t dst, VARP_t src);
MNN_C_API VARP_t mnn_expr_VARP_static_create_EXPRP(EXPRP_t expr, int index);

MNN_C_API VecVARP_t mnn_expr_VARP_static_load(const char *fileName);
MNN_C_API VARMAP_t mnn_expr_VARP_static_loadMap(const char *fileName);
MNN_C_API VecVARP_t mnn_expr_VARP_static_loadBuffer(const uint8_t *buffer, size_t length);
MNN_C_API VARMAP_t mnn_expr_VARP_static_loadMapBuffer(const uint8_t *buffer, size_t length);
// MNN_C_API void mnn_expr_VARP_static_getExecuteOrder(VecVARP_t output);
MNN_C_API void mnn_expr_VARP_static_save(VecVARP_t vars, const char *fileName);
MNN_C_API VecI8 mnn_expr_VARP_static_saveBytes(VecVARP_t vars);
MNN_C_API void mnn_expr_VARP_static_saveNet(VecVARP_t vars, Net_t dest);

MNN_C_API void mnn_expr_VARP_static_prepareCompute(VecVARP_t vars, bool forceCPU);
MNN_C_API void mnn_expr_VARP_static_compute(VecVARP_t vars, bool forceCPU);
MNN_C_API size_t mnn_expr_VARP_linkNumber(VARP_t self);
MNN_C_API const VecWeakEXPRP_t mnn_expr_VARP_toExprs(VARP_t self);
MNN_C_API void mnn_expr_VARP_setExpr(VARP_t self, EXPRP_t expr, int index);
MNN_C_API const mnn_tensor_t mnn_expr_VARP_getTensor(VARP_t self);

// MNN::Express::Expr
MNN_C_API EXPRP_t mnn_expr_Expr_create_empty();
MNN_C_API EXPRP_t mnn_expr_Expr_static_create(mnn_tensor_t tensor, bool own);
MNN_C_API EXPRP_t mnn_expr_Expr_static_create_1(
    struct mnn_expr_Variable_Info *info, const void *ptr, int type, int memoryType
);
MNN_C_API EXPRP_t mnn_expr_Expr_static_create_2(OpT_t op, VecVARP_t inputs, int outputSize);
MNN_C_API void mnn_expr_Expr_free(EXPRP_t self);

MNN_C_API void mnn_expr_Expr_setName(EXPRP_t self, const char *name);
MNN_C_API const Op_t mnn_expr_Expr_getOp(EXPRP_t self);
MNN_C_API VecVARP_t mnn_expr_Expr_getInputs(EXPRP_t self);
MNN_C_API VecWeakEXPRP_t mnn_expr_Expr_getOutputs(EXPRP_t self);
MNN_C_API int mnn_expr_Expr_getOutputSize(EXPRP_t self);
MNN_C_API void mnn_expr_Expr_static_replace(EXPRP_t oldExpr, EXPRP_t newExpr);
MNN_C_API bool mnn_expr_Expr_requireInfo(EXPRP_t self);
MNN_C_API const char *mnn_expr_Expr_getName(EXPRP_t self);
MNN_C_API const char *mnn_expr_Expr_getOutputName(EXPRP_t self, int index);
MNN_C_API int mnn_expr_Expr_inputType(EXPRP_t self);
// MNN_C_API void mnn_expr_Expr_visitOutputs();
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_Expr_outputInfo(EXPRP_t self, int index);

#ifdef __cplusplus
}
#endif

#endif // MNN_EXPR_H
