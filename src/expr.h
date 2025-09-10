#ifndef MNN_EXPR_H
#define MNN_EXPR_H

#include "error_code.h"
#include "mnn_type.h"
#include "tensor.h"

#ifdef __cplusplus
#include "MNN/expr/Expr.hpp"
extern "C" {
#endif

/** Opaque pointer types */
#ifdef __cplusplus
typedef MNN::Express::VARP *mnn_expr_VARP_t;
typedef MNN::Express::EXPRP *mnn_expr_EXPRP_t;
typedef MNN::Express::INTS *mnn_expr_INTS_t;
typedef MNN::Express::Variable *mnn_expr_Variable_t;
typedef std::shared_ptr<MNN::Express::Variable> *mnn_expr_VariableP_t;
typedef MNN::Express::VARPS *mnn_expr_VARPS_t;
typedef std::vector<MNN::Express::WeakEXPRP> *mnn_expr_WeakEXPRPS_t;
typedef MNN::Express::Expr *mnn_expr_Expr_t;
typedef MNN::OpT *mnn_OpT_t;
#else
typedef void *mnn_expr_VARP_t;
typedef void *mnn_expr_EXPRP_t;
typedef void *mnn_expr_INTS_t;
typedef void *mnn_expr_Variable_t;
typedef void *mnn_expr_VariableP_t;
typedef void *mnn_expr_VARPS_t;
typedef void *mnn_expr_WeakEXPRPS_t;
typedef void *mnn_expr_Expr_t;
typedef void *mnn_OpT_t;
#endif

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
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_mean(mnn_expr_VARP_t self, mnn_expr_INTS_t dims);
MNN_C_API mnn_expr_VARP_t mnn_expr_VARP_sum(mnn_expr_VARP_t self, mnn_expr_INTS_t dims);
MNN_C_API bool mnn_expr_VARP_op_eqeq(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_less(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API bool mnn_expr_VARP_op_lessequal(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API void mnn_expr_VARP_op_assign(mnn_expr_VARP_t self, mnn_expr_VARP_t other);
MNN_C_API void mnn_expr_VARP_op_assign_variable(mnn_expr_VARP_t self, mnn_expr_Variable_t other);
MNN_C_API bool mnn_expr_VARP_fix(mnn_expr_VARP_t self, int type);        // InputType type
MNN_C_API void mnn_expr_VARP_setOrder(mnn_expr_VARP_t self, int format); // Dimensionformat format

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
// MNN_C_API struct mnn_expr_Variable_expr_pair *mnn_expr_Variable_getExpr(mnn_expr_Variable_t
// self); struct Info {
//   Dimensionformat order = NHWC;
//   INTS dim;
//   halide_type_t type;
//   size_t size;
//   void syncSize();
// };
struct mnn_expr_Variable_Info {
  int order;
  mnn_expr_INTS_t dim;
  halide_type_c_t type;
  size_t size;
};
MNN_C_API struct mnn_expr_Variable_Info *mnn_expr_Variable_getInfo(mnn_expr_Variable_t self);
MNN_C_API bool mnn_expr_Variable_resize(mnn_expr_Variable_t self, mnn_expr_INTS_t dims);
MNN_C_API const void *mnn_expr_Variable_readMap(mnn_expr_Variable_t self);
MNN_C_API void *mnn_expr_Variable_writeMap(mnn_expr_Variable_t self);
MNN_C_API void
mnn_expr_Variable_writeScaleMap(mnn_expr_Variable_t self, float scaleValue, float zeroPoint);
MNN_C_API void mnn_expr_Variable_unMap(mnn_expr_Variable_t self);
MNN_C_API bool mnn_expr_Variable_input(mnn_expr_Variable_t self, mnn_expr_VARP_t src);
MNN_C_API void mnn_expr_Variable_static_replace(mnn_expr_VARP_t dst, mnn_expr_VARP_t src);
MNN_C_API mnn_expr_VARP_t mnn_expr_Variable_static_create_EXPRP(mnn_expr_EXPRP_t expr, int index);
MNN_C_API mnn_expr_VARPS_t mnn_expr_Variable_static_load(const char *fileName);
MNN_C_API mnn_expr_VARPS_t
mnn_expr_Variable_static_loadBuffer(const uint8_t *buffer, size_t length);
/*
 * static std::map<std::string, VARP> loadMap(const char* fileName);
 * static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
 * static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>>
 * getInputAndOutput(const std::map<std::string, VARP>& allVariable); static std::vector<VARP>
 * mapToSequence(const std::map<std::string, VARP>& source); static std::vector<EXPRP>
 * getExecuteOrder(const std::vector<VARP>& output);
 */
MNN_C_API void mnn_expr_Variable_static_save(mnn_expr_VARPS_t vars, const char *fileName);
MNN_C_API void mnn_expr_Variable_static_prepareCompute(mnn_expr_VARPS_t vars, bool forceCPU);
MNN_C_API void mnn_expr_Variable_static_compute(mnn_expr_VARPS_t vars, bool forceCPU);
MNN_C_API size_t mnn_expr_Variable_linkNumber(mnn_expr_Variable_t self);
MNN_C_API const mnn_expr_WeakEXPRPS_t mnn_expr_Variable_toExprs(mnn_expr_Variable_t self);
MNN_C_API void
mnn_expr_Variable_setExpr(mnn_expr_Variable_t self, mnn_expr_EXPRP_t expr, int index);
MNN_C_API const mnn_tensor_t mnn_expr_Variable_getTensor(mnn_expr_Variable_t self);

// MNN::Express::Expr
MNN_C_API mnn_expr_EXPRP_t mnn_expr_Expr_static_create(mnn_tensor_t tensor, bool own);
MNN_C_API mnn_expr_EXPRP_t mnn_expr_Expr_static_create_1(
    struct mnn_expr_Variable_Info *info, const void *ptr, int type, int memoryType
);
MNN_C_API mnn_expr_EXPRP_t
mnn_expr_Expr_static_create_2(mnn_OpT_t op, mnn_expr_VARPS_t inputs, int outputSize);
// static EXPRP create(std::shared_ptr<BufferStorage> extra, std::vector<VARP>&& inputs, int
// outputSize = 1);
MNN_C_API void mnn_expr_Expr_setName(mnn_expr_Expr_t self, char *name);

// Math Op
// BinaryOPs
MNN_C_API mnn_expr_VARP_t mnn_expr_Add(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Subtract(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Multiply(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Divide(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Pow(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Minimum(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Maximum(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_BiasAdd(mnn_expr_VARP_t value, mnn_expr_VARP_t bias);
MNN_C_API mnn_expr_VARP_t mnn_expr_Greater(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_GreaterEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Less(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_FloorDiv(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_SquaredDifference(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Equal(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_LessEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_FloorMod(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_Atan2(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_LogicalOr(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_NotEqual(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_BitwiseAnd(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_BitwiseOr(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_BitwiseXor(mnn_expr_VARP_t x, mnn_expr_VARP_t y);

// UnaryOPs
MNN_C_API mnn_expr_VARP_t mnn_expr_Sign(mnn_expr_VARP_t a);
MNN_C_API mnn_expr_VARP_t mnn_expr_Abs(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Negative(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Floor(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Round(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Ceil(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Square(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Sqrt(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Rsqrt(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Exp(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Log(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Sin(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Sinh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Cos(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Cosh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Tan(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Asin(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Asinh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Acos(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Acosh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Atan(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Atanh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Reciprocal(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Log1p(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Gelu(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Tanh(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Sigmoid(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Erf(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Erfc(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Erfinv(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Expm1(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Hardswish(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Silu(mnn_expr_VARP_t x);

// ReduceOPs
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceSum(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMean(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMax(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMin(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceProd(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceAny(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceAll(mnn_expr_VARP_t input_variable, mnn_expr_INTS_t axis, bool keepDims);

MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceSumMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMeanMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMaxMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceMinMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceProdMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceAnyMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReduceAllMutable(mnn_expr_VARP_t input_variable, mnn_expr_VARP_t axis, bool keepDims);

// EltwiseOPs
MNN_C_API mnn_expr_VARP_t
mnn_expr_Prod(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Sum(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Max(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Sub(mnn_expr_VARP_t a, mnn_expr_VARP_t b, float *coeff, size_t coeffSize);
// MNN_PUBLIC VARP _EltwiseProdInt8(VARP x, VARP y,
//                     std::vector<int8_t> x_weight, std::vector<int32_t> x_bias, std::vector<float>
//                     x_scale, std::vector<float> x_tensorScale, std::vector<int8_t> y_weight,
//                     std::vector<int32_t> y_bias, std::vector<float> y_scale, std::vector<float>
//                     y_tensorScale, std::vector<int8_t> output_weight, std::vector<int32_t>
//                     output_bias, std::vector<float> output_scale, std::vector<float>
//                     output_tensorScale);
// MNN_PUBLIC VARP _EltwiseSumInt8(VARP x, VARP y,
//                      std::vector<int8_t> x_weight, std::vector<int32_t> x_bias,
//                      std::vector<float> x_scale, std::vector<float> x_tensorScale,
//                     std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float>
//                     y_scale, std::vector<float> y_tensorScale, std::vector<int8_t> output_weight,
//                     std::vector<int32_t> output_bias, std::vector<float> output_scale,
//                     std::vector<float> output_tensorScale);
// MNN_PUBLIC VARP _EltwiseSubInt8(VARP x, VARP y,
//                      std::vector<int8_t> x_weight, std::vector<int32_t> x_bias,
//                      std::vector<float> x_scale, std::vector<float> x_tensorScale,
//                     std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float>
//                     y_scale, std::vector<float> y_tensorScale, std::vector<int8_t> output_weight,
//                     std::vector<int32_t> output_bias, std::vector<float> output_scale,
//                     std::vector<float> output_tensorScale);
// MNN_PUBLIC VARP _EltwiseMaxInt8(VARP x, VARP y,
//                       std::vector<int8_t> x_weight, std::vector<int32_t> x_bias,
//                       std::vector<float> x_scale, std::vector<float> x_tensorScale,
//                     std::vector<int8_t> y_weight, std::vector<int32_t> y_bias, std::vector<float>
//                     y_scale, std::vector<float> y_tensorScale, std::vector<int8_t> output_weight,
//                     std::vector<int32_t> output_bias, std::vector<float> output_scale,
//                     std::vector<float> output_tensorScale);
MNN_C_API mnn_expr_VARP_t mnn_expr_Mod(mnn_expr_VARP_t x, mnn_expr_VARP_t y);

// OtherOPs
//  template<typename T>
//  VARP _Cast(VARP x) {
//    return _Cast(x, halide_type_of<T>());
//  }
MNN_C_API mnn_expr_VARP_t mnn_expr_Cast(mnn_expr_VARP_t x, halide_type_c_t dtype);
MNN_C_API mnn_expr_VARP_t
mnn_expr_MatMul(mnn_expr_VARP_t a, mnn_expr_VARP_t b, bool tranposeA, bool tranposeB);
MNN_C_API mnn_expr_VARP_t mnn_expr_Normalize(
    mnn_expr_VARP_t x,
    int32_t acrossSpatial,
    int32_t channelShared,
    float eps,
    const float *scale,
    size_t scaleLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ArgMax(mnn_expr_VARP_t input, int axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_ArgMin(mnn_expr_VARP_t input, int axis);
MNN_C_API mnn_expr_VARP_t
mnn_expr_BatchMatMul(mnn_expr_VARP_t x, mnn_expr_VARP_t y, bool adj_x, bool adj_y);
MNN_C_API mnn_expr_VARP_t mnn_expr_UnravelIndex(mnn_expr_VARP_t indices, mnn_expr_VARP_t dims);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ScatterNd(mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape);
MNN_C_API mnn_expr_VARP_t mnn_expr_ScatterNd_1(
    mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape, mnn_expr_VARP_t input
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ScatterNd_2(
    mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, mnn_expr_VARP_t shape, int reduction
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ScatterNd_3(
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t updates,
    mnn_expr_VARP_t shape,
    mnn_expr_VARP_t input,
    int reduction
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ScatterElements(
    mnn_expr_VARP_t data, mnn_expr_VARP_t indices, mnn_expr_VARP_t updates, int reduction
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ScatterElements_1(
    mnn_expr_VARP_t data,
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t updates,
    mnn_expr_VARP_t axis,
    int reduction
);
MNN_C_API mnn_expr_VARP_t mnn_expr_OneHot(
    mnn_expr_VARP_t indices,
    mnn_expr_VARP_t depth,
    mnn_expr_VARP_t onValue,
    mnn_expr_VARP_t offValue,
    int axis
);
MNN_C_API mnn_expr_VARP_t mnn_expr_BroadcastTo(mnn_expr_VARP_t a, mnn_expr_VARP_t shape);
MNN_C_API mnn_expr_VARP_t
mnn_expr_LinSpace(mnn_expr_VARP_t start, mnn_expr_VARP_t stop, mnn_expr_VARP_t num);

MNN_C_API mnn_expr_VARP_t mnn_expr_RandomUnifom(
    mnn_expr_VARP_t shape, halide_type_c_t dtype, float low, float high, int seed0, int seed1
);
MNN_C_API mnn_expr_VARP_t
mnn_expr_CumSum(mnn_expr_VARP_t x, int axis, bool exclusive, bool reverse);
MNN_C_API mnn_expr_VARP_t mnn_expr_CumProd(mnn_expr_VARP_t x, int axis);
MNN_C_API mnn_expr_VARPS_t mnn_expr_Svd(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Histogram(mnn_expr_VARP_t x, int bin, int min, int max, int channel);

// Neural Network Ops
MNN_C_API mnn_expr_VARP_t
mnn_expr_Input(const int *shape, size_t shapeLength, int data_format, halide_type_c_t dtype);
MNN_C_API mnn_expr_VARP_t mnn_expr_Clone(mnn_expr_VARP_t source, bool deepCopy);
MNN_C_API mnn_expr_VARP_t mnn_expr_Scalar(const void *ptr, halide_type_c_t type);

MNN_C_API mnn_expr_VARP_t
mnn_expr_Const(float value, const int *shape, size_t shapeLength, int format);
// MNN_PUBLIC VARP _Const(const void* ptr, INTS shape = {}, Dimensionformat format = NHWC,
//                   halide_type_t type = halide_type_of<float>());
// MNN_PUBLIC VARP _InnerProduct(std::vector<float>&& weight, std::vector<float>&& bias, VARP x,
// INTS outputShape);
MNN_C_API mnn_expr_VARP_t mnn_expr_Conv(
    mnn_expr_VARP_t weight,
    mnn_expr_VARP_t bias,
    mnn_expr_VARP_t x,
    int pad,
    const int *stride,
    size_t strideLength,
    const int *dilate,
    size_t dilateLength,
    int group,
    const int *pads,
    size_t padsLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Deconv(
    mnn_expr_VARP_t weight,
    mnn_expr_VARP_t bias,
    mnn_expr_VARP_t x,
    int pad,
    const int *stride,
    size_t strideLength,
    const int *dilate,
    size_t dilateLength,
    int group,
    const int *pads,
    size_t padsLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_MaxPool(
    mnn_expr_VARP_t x,
    const int *kernel,
    size_t kernelLength,
    const int *stride,
    size_t strideLength,
    int pad,
    const int *pads,
    size_t padsLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_AvePool(
    mnn_expr_VARP_t x,
    const int *kernel,
    size_t kernelLength,
    const int *stride,
    size_t strideLength,
    int pad,
    const int *pads,
    size_t padsLength
);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Reshape(mnn_expr_VARP_t x, const int *shape, size_t shapeLength, int original_format);
MNN_C_API mnn_expr_VARP_t mnn_expr_Reshape_1(mnn_expr_VARP_t x, mnn_expr_VARP_t shape);
MNN_C_API mnn_expr_VARP_t mnn_expr_Scale(
    mnn_expr_VARP_t x,
    int channels,
    float *scales,
    size_t scaleLength,
    float *bias,
    size_t biasLength
);

MNN_C_API mnn_expr_VARP_t mnn_expr_Relu(mnn_expr_VARP_t x, float slope);
MNN_C_API mnn_expr_VARP_t mnn_expr_Relu6(mnn_expr_VARP_t x, float minValue, float maxValue);
MNN_C_API mnn_expr_VARP_t mnn_expr_PRelu(mnn_expr_VARP_t x, float *slopes, size_t slopeLength);
MNN_C_API mnn_expr_VARP_t mnn_expr_Softmax(mnn_expr_VARP_t logits, int axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_Softplus(mnn_expr_VARP_t features);
MNN_C_API mnn_expr_VARP_t mnn_expr_Softsign(mnn_expr_VARP_t features);
MNN_C_API mnn_expr_VARPS_t
mnn_expr_Split(mnn_expr_VARP_t value, const int *size_splits, size_t size_splitsLength, int axis);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Slice(mnn_expr_VARP_t x, mnn_expr_VARP_t starts, mnn_expr_VARP_t sizes);
MNN_C_API mnn_expr_VARP_t mnn_expr_StridedSlice(
    mnn_expr_VARP_t input,
    mnn_expr_VARP_t begin,
    mnn_expr_VARP_t end,
    mnn_expr_VARP_t strided,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
);
MNN_C_API mnn_expr_VARP_t mnn_expr_StridedSliceWrite(
    mnn_expr_VARP_t input,
    mnn_expr_VARP_t begin,
    mnn_expr_VARP_t end,
    mnn_expr_VARP_t strided,
    mnn_expr_VARP_t write,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Concat(mnn_expr_VARPS_t values, int axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_Convert(mnn_expr_VARP_t input, int format);
MNN_C_API mnn_expr_VARP_t mnn_expr_Transpose(mnn_expr_VARP_t x, const int *perm, size_t permLength);
MNN_C_API mnn_expr_VARP_t mnn_expr_Transpose_1(mnn_expr_VARP_t x, mnn_expr_VARP_t perm);
MNN_C_API mnn_expr_VARP_t mnn_expr_ChannelShuffle(mnn_expr_VARP_t x, int group);
MNN_C_API mnn_expr_VARP_t mnn_expr_ChangeInputFormat(mnn_expr_VARP_t input, int format);
MNN_C_API mnn_expr_VARP_t mnn_expr_Reverse(mnn_expr_VARP_t x, mnn_expr_VARP_t axis);
MNN_C_API mnn_expr_VARP_t
mnn_expr_ReverseSequence(mnn_expr_VARP_t x, mnn_expr_VARP_t y, int batchDim, int seqDim);
MNN_C_API mnn_expr_VARP_t mnn_expr_Crop(
    mnn_expr_VARP_t images, mnn_expr_VARP_t size, int axis, const int *offset, size_t offsetLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Resize(mnn_expr_VARP_t images, float xScale, float yScale);
MNN_C_API mnn_expr_VARP_t mnn_expr_Pad(mnn_expr_VARP_t x, mnn_expr_VARP_t paddings, int mode);
MNN_C_API mnn_expr_VARP_t mnn_expr_ExpandDims(mnn_expr_VARP_t input, int axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_ExpandDims_1(mnn_expr_VARP_t input, mnn_expr_VARP_t axis);

MNN_C_API mnn_expr_VARP_t mnn_expr_Shape(mnn_expr_VARP_t input, bool nchw);
MNN_C_API mnn_expr_VARP_t mnn_expr_Stack(mnn_expr_VARPS_t values, int axis);
// enum InterpolationMethod {BILINEAR, NEAREST};
MNN_C_API mnn_expr_VARP_t mnn_expr_CropAndResize(
    mnn_expr_VARP_t image,
    mnn_expr_VARP_t boxes,
    mnn_expr_VARP_t box_ind,
    mnn_expr_VARP_t crop_size,
    int method,
    float extrapolation_value
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Fill(mnn_expr_VARP_t dims, mnn_expr_VARP_t value);
MNN_C_API mnn_expr_VARP_t mnn_expr_Tile(mnn_expr_VARP_t input, mnn_expr_VARP_t multiples);
MNN_C_API mnn_expr_VARP_t mnn_expr_Gather(mnn_expr_VARP_t params, mnn_expr_VARP_t indices);
MNN_C_API mnn_expr_VARP_t
mnn_expr_GatherV2(mnn_expr_VARP_t params, mnn_expr_VARP_t indices, mnn_expr_VARP_t axis);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Squeeze(mnn_expr_VARP_t input, const int *axis, size_t axisLength);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Unsqueeze(mnn_expr_VARP_t input, const int *axis, size_t axisLength);
MNN_C_API mnn_expr_VARP_t
mnn_expr_BatchToSpaceND(mnn_expr_VARP_t input, mnn_expr_VARP_t block_shape, mnn_expr_VARP_t crops);
MNN_C_API mnn_expr_VARP_t mnn_expr_GatherND(mnn_expr_VARP_t params, mnn_expr_VARP_t indices);
MNN_C_API mnn_expr_VARP_t mnn_expr_GatherElements(mnn_expr_VARP_t params, mnn_expr_VARP_t indices);
MNN_C_API mnn_expr_VARP_t
mnn_expr_GatherElements_1(mnn_expr_VARP_t params, mnn_expr_VARP_t indices, mnn_expr_VARP_t axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_Selu(mnn_expr_VARP_t features, float scale, float alpha);
MNN_C_API mnn_expr_VARP_t mnn_expr_Size(mnn_expr_VARP_t input);
MNN_C_API mnn_expr_VARP_t mnn_expr_Elu(mnn_expr_VARP_t features, float alpha);
MNN_C_API mnn_expr_VARP_t mnn_expr_Threshold(mnn_expr_VARP_t features, float alpha);
MNN_C_API mnn_expr_VARP_t mnn_expr_MatrixBandPart(
    mnn_expr_VARP_t input, mnn_expr_VARP_t num_lower, mnn_expr_VARP_t num_upper
);
MNN_C_API mnn_expr_VARPS_t mnn_expr_Moments(
    mnn_expr_VARP_t x, const int *axis, size_t axisLength, mnn_expr_VARP_t shift, bool keepDims
);
MNN_C_API mnn_expr_VARP_t mnn_expr_SetDiff1D(mnn_expr_VARP_t x, mnn_expr_VARP_t y);
MNN_C_API mnn_expr_VARP_t mnn_expr_SpaceToDepth(mnn_expr_VARP_t input, int block_size);
MNN_C_API mnn_expr_VARP_t mnn_expr_SpaceToBatchND(
    mnn_expr_VARP_t input, mnn_expr_VARP_t block_shape, mnn_expr_VARP_t paddings
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ZerosLike(mnn_expr_VARP_t input);
MNN_C_API mnn_expr_VARPS_t mnn_expr_Unstack(mnn_expr_VARP_t value, int axis);
MNN_C_API mnn_expr_VARP_t mnn_expr_Rank(mnn_expr_VARP_t input);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Range(mnn_expr_VARP_t start, mnn_expr_VARP_t limit, mnn_expr_VARP_t delta);
MNN_C_API mnn_expr_VARP_t mnn_expr_DepthToSpace(mnn_expr_VARP_t input, int block_size);
// MNN_PUBLIC VARP _PriorBox(VARP feature, VARP image,
//                             std::vector<float> min_size, std::vector<float> max_size,
//                             std::vector<float>aspect_ratio, bool flip, bool clip,
//                             std::vector<float>variance, unsigned int img_h, unsigned int img_w,
//                             float step_h, float step_w, float offset = 0.5);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Permute(mnn_expr_VARP_t input, const int *dims, size_t dimsLength);
// MNN_PUBLIC VARP _DetectionOutput(VARP location, VARP confidence, VARP priorbox,
//                         unsigned int num_classes, bool share_location, int background_label_id,
//                         float nms_threshhold, int nms_topk, int code_type,
//                         bool variance_encoded_in_target,
//                         int keep_top_k, float confidence_threshold, float visualize_threshold);
//   MNN_PUBLIC  std::vector<VARP> _DetectionPostProcess(VARP encode_boxes, VARP class_predictions,
//   VARP anchors,
//                           int num_classes, int max_detections,
//                           int max_class_per_detection, int detections_per_class,
//                           float nms_threshold, float iou_threshold,
//                           bool use_regular_nms, std::vector<float> centersize_encoding);
MNN_C_API mnn_expr_VARP_t mnn_expr_Interp(
    mnn_expr_VARPS_t xs,
    float widthScale,
    float heightScale,
    int outputWidth,
    int outputHeight,
    int resizeType,
    bool alignCorners
);
MNN_C_API mnn_expr_VARP_t mnn_expr_ZeroGrad(mnn_expr_VARP_t x);

// Int8 Inference
// MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&&
// scale, VARP x, INTS channel, INTS kernelSize,
//                       PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
//                       int nbits = 8);
// MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<int>&& bias, std::vector<float>&&
// scale,
//                       VARP x, INTS channel, INTS kernelSize,
//                       PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
//                       int8_t inputZeroPoint, int8_t outputZeroPoint,
//                       int8_t minValue, int8_t maxValue, bool accumulateToInt16);
// MNN_PUBLIC VARP _Conv(std::vector<int8_t>&& weight, std::vector<float>&& bias,
// std::vector<float>&& weightScale,
//                       VARP x, INTS channel, INTS kernelSize,
//                       PaddingMode pad, INTS stride, INTS dilate, int group, INTS pads, bool relu,
//                       float scaleIn, float scaleOut,
//                       int8_t inputZeroPoint, int8_t outputZeroPoint,
//                       int8_t minValue, int8_t maxValue, float weightClampValue, bool
//                       accumulateToInt16);
MNN_C_API mnn_expr_VARP_t
mnn_expr_CosineSimilarity(mnn_expr_VARP_t input0, mnn_expr_VARP_t input1, mnn_expr_VARP_t inputDim);
// enum GridSamplePaddingMode {GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER,
// GRID_SAMPLE_PADDING_REFLECTION};
MNN_C_API mnn_expr_VARP_t mnn_expr_GridSample(
    mnn_expr_VARP_t input, mnn_expr_VARP_t grid, int mode, int paddingMode, bool alignCorners
);
MNN_C_API mnn_expr_VARP_t
mnn_expr_FloatToInt8(mnn_expr_VARP_t x, mnn_expr_VARP_t scale, char minValue, char maxValue);
MNN_C_API mnn_expr_VARP_t mnn_expr_FloatToInt8_1(
    mnn_expr_VARP_t x, mnn_expr_VARP_t scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Int8ToFloat(mnn_expr_VARP_t x, mnn_expr_VARP_t scale);
MNN_C_API mnn_expr_VARP_t
mnn_expr_Int8ToFloat_1(mnn_expr_VARP_t x, mnn_expr_VARP_t scale, int8_t zeroPoint);

MNN_C_API mnn_expr_VARP_t
mnn_expr_Select(mnn_expr_VARP_t select, mnn_expr_VARP_t input0, mnn_expr_VARP_t input1);
MNN_C_API mnn_expr_VARPS_t mnn_expr_TopKV2(mnn_expr_VARP_t input0, mnn_expr_VARP_t input1);
// MNN_PUBLIC VARP _ImageProcess(VARP input, CV::ImageProcess::Config config, CV::Matrix matrix, int
// oh, int ow, int oc, int dtype, uint8_t padVal = 0); mnn_expr_VARP_t
// mnn_expr_ImageProcess(mnn_expr_VARP_t input, CV::ImageProcess::Config config, CV::Matrix matrix,
// int oh, int ow, int oc, int dtype, uint8_t padVal);
MNN_C_API mnn_expr_VARP_t mnn_expr_Where(mnn_expr_VARP_t x);
MNN_C_API mnn_expr_VARP_t mnn_expr_Sort(mnn_expr_VARP_t x, int axis, bool arg, bool descend);
MNN_C_API mnn_expr_VARP_t mnn_expr_Raster(
    mnn_expr_VARPS_t vars,
    const int *regions,
    size_t regionsLength,
    const int *shape,
    size_t shapeLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_RasterRaw(
    mnn_expr_VARPS_t vars,
    const int *region,
    size_t regionLength,
    const int *shape,
    size_t shapeLength,
    struct halide_type_t dataType,
    int format
);

MNN_C_API mnn_expr_VARP_t mnn_expr_Nms(
    mnn_expr_VARP_t boxes,
    mnn_expr_VARP_t scores,
    int maxDetections,
    float iouThreshold,
    float scoreThreshold
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Im2Col(
    mnn_expr_VARP_t x,
    const int *kernelSize,
    size_t kernelSizeLength,
    const int *dilate,
    size_t dilateLength,
    const int *pads,
    size_t padsLength,
    const int *stride,
    size_t strideLength
);
MNN_C_API mnn_expr_VARP_t mnn_expr_Col2Im(
    mnn_expr_VARP_t x,
    mnn_expr_VARP_t outputShape,
    const int *kernelSize,
    size_t kernelSizeLength,
    const int *dilate,
    size_t dilateLength,
    const int *pads,
    size_t padsLength,
    const int *stride,
    size_t strideLength
);

#ifdef __cplusplus
}
#endif

#endif // MNN_EXPR_H
