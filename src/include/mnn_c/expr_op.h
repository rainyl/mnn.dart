//
// Created by rainy on 2025/9/11.
//

#ifndef MNN_EXPR_OP_H
#define MNN_EXPR_OP_H

#include "mnn_c/base.h"
#include "mnn_c/expr.h"

#ifdef __cplusplus
extern "C" {
#endif
// Math Op
// BinaryOPs
MNN_C_API VARP_t mnn_expr_Add(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Subtract(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Multiply(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Divide(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Pow(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Minimum(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Maximum(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_BiasAdd(VARP_t value, VARP_t bias);
MNN_C_API VARP_t mnn_expr_Greater(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_GreaterEqual(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Less(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_FloorDiv(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_SquaredDifference(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Equal(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_LessEqual(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_FloorMod(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_Atan2(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_LogicalOr(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_NotEqual(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_BitwiseAnd(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_BitwiseOr(VARP_t x, VARP_t y);
MNN_C_API VARP_t mnn_expr_BitwiseXor(VARP_t x, VARP_t y);

// UnaryOPs
MNN_C_API VARP_t mnn_expr_Sign(VARP_t a);
MNN_C_API VARP_t mnn_expr_Abs(VARP_t x);
MNN_C_API VARP_t mnn_expr_Negative(VARP_t x);
MNN_C_API VARP_t mnn_expr_Floor(VARP_t x);
MNN_C_API VARP_t mnn_expr_Round(VARP_t x);
MNN_C_API VARP_t mnn_expr_Ceil(VARP_t x);
MNN_C_API VARP_t mnn_expr_Square(VARP_t x);
MNN_C_API VARP_t mnn_expr_Sqrt(VARP_t x);
MNN_C_API VARP_t mnn_expr_Rsqrt(VARP_t x);
MNN_C_API VARP_t mnn_expr_Exp(VARP_t x);
MNN_C_API VARP_t mnn_expr_Log(VARP_t x);
MNN_C_API VARP_t mnn_expr_Sin(VARP_t x);
MNN_C_API VARP_t mnn_expr_Sinh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Cos(VARP_t x);
MNN_C_API VARP_t mnn_expr_Cosh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Tan(VARP_t x);
MNN_C_API VARP_t mnn_expr_Asin(VARP_t x);
MNN_C_API VARP_t mnn_expr_Asinh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Acos(VARP_t x);
MNN_C_API VARP_t mnn_expr_Acosh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Atan(VARP_t x);
MNN_C_API VARP_t mnn_expr_Atanh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Reciprocal(VARP_t x);
MNN_C_API VARP_t mnn_expr_Log1p(VARP_t x);
MNN_C_API VARP_t mnn_expr_Gelu(VARP_t x);
MNN_C_API VARP_t mnn_expr_Tanh(VARP_t x);
MNN_C_API VARP_t mnn_expr_Sigmoid(VARP_t x);
MNN_C_API VARP_t mnn_expr_Erf(VARP_t x);
MNN_C_API VARP_t mnn_expr_Erfc(VARP_t x);
MNN_C_API VARP_t mnn_expr_Erfinv(VARP_t x);
MNN_C_API VARP_t mnn_expr_Expm1(VARP_t x);
MNN_C_API VARP_t mnn_expr_Hardswish(VARP_t x);
MNN_C_API VARP_t mnn_expr_Silu(VARP_t x);

// ReduceOPs
MNN_C_API VARP_t mnn_expr_ReduceSum(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMean(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceVariance(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMax(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMin(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceProd(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceAny(VARP_t input_variable, VecI32 axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceAll(VARP_t input_variable, VecI32 axis, bool keepDims);

MNN_C_API VARP_t mnn_expr_ReduceSumMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMeanMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMaxMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceMinMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceProdMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceAnyMutable(VARP_t input_variable, VARP_t axis, bool keepDims);
MNN_C_API VARP_t mnn_expr_ReduceAllMutable(VARP_t input_variable, VARP_t axis, bool keepDims);

// EltwiseOPs
MNN_C_API VARP_t mnn_expr_Prod(VARP_t a, VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API VARP_t mnn_expr_Sum(VARP_t a, VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API VARP_t mnn_expr_Max(VARP_t a, VARP_t b, float *coeff, size_t coeffSize);
MNN_C_API VARP_t mnn_expr_Sub(VARP_t a, VARP_t b, float *coeff, size_t coeffSize);
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
MNN_C_API VARP_t mnn_expr_Mod(VARP_t x, VARP_t y);

// OtherOPs
//  template<typename T>
//  VARP _Cast(VARP x) {
//    return _Cast(x, halide_type_of<T>());
//  }
MNN_C_API VARP_t mnn_expr_Cast(VARP_t x, halide_type_c_t dtype);
MNN_C_API VARP_t mnn_expr_MatMul(VARP_t a, VARP_t b, bool tranposeA, bool tranposeB);
MNN_C_API VARP_t mnn_expr_Normalize(
    VARP_t       x,
    int32_t      acrossSpatial,
    int32_t      channelShared,
    float        eps,
    const float *scale,
    size_t       scaleLength
);
MNN_C_API VARP_t mnn_expr_ArgMax(VARP_t input, int axis);
MNN_C_API VARP_t mnn_expr_ArgMin(VARP_t input, int axis);
MNN_C_API VARP_t mnn_expr_BatchMatMul(VARP_t x, VARP_t y, bool adj_x, bool adj_y);
MNN_C_API VARP_t mnn_expr_UnravelIndex(VARP_t indices, VARP_t dims);
MNN_C_API VARP_t mnn_expr_ScatterNd(VARP_t indices, VARP_t updates, VARP_t shape);
MNN_C_API VARP_t mnn_expr_ScatterNd_1(VARP_t indices, VARP_t updates, VARP_t shape, VARP_t input);
MNN_C_API VARP_t mnn_expr_ScatterNd_2(VARP_t indices, VARP_t updates, VARP_t shape, int reduction);
MNN_C_API VARP_t
mnn_expr_ScatterNd_3(VARP_t indices, VARP_t updates, VARP_t shape, VARP_t input, int reduction);
MNN_C_API VARP_t
mnn_expr_ScatterElements(VARP_t data, VARP_t indices, VARP_t updates, int reduction);
MNN_C_API VARP_t
mnn_expr_ScatterElements_1(VARP_t data, VARP_t indices, VARP_t updates, VARP_t axis, int reduction);
MNN_C_API VARP_t
mnn_expr_OneHot(VARP_t indices, VARP_t depth, VARP_t onValue, VARP_t offValue, int axis);
MNN_C_API VARP_t mnn_expr_BroadcastTo(VARP_t a, VARP_t shape);
MNN_C_API VARP_t mnn_expr_LinSpace(VARP_t start, VARP_t stop, VARP_t num);

MNN_C_API VARP_t mnn_expr_RandomUniform(
    VARP_t shape, halide_type_c_t dtype, float low, float high, int seed0, int seed1
);
MNN_C_API VARP_t    mnn_expr_CumSum(VARP_t x, int axis, bool exclusive, bool reverse);
MNN_C_API VARP_t    mnn_expr_CumProd(VARP_t x, int axis);
MNN_C_API VecVARP_t mnn_expr_Svd(VARP_t x);
MNN_C_API VARP_t    mnn_expr_Histogram(VARP_t x, int bin, int min, int max, int channel);

// Neural Network Ops
MNN_C_API VARP_t
mnn_expr_Input(const int *shape, size_t shapeLength, int data_format, halide_type_c_t dtype);
MNN_C_API VARP_t mnn_expr_Clone(VARP_t source, bool deepCopy);
MNN_C_API VARP_t mnn_expr_Scalar(const void *ptr, halide_type_c_t type);

MNN_C_API VARP_t
mnn_expr_Const(void *value, const int *shape, size_t shapeLength, int format, halide_type_c_t type);
// MNN_PUBLIC VARP _InnerProduct(std::vector<float>&& weight, std::vector<float>&& bias, VARP x,
// INTS outputShape);
MNN_C_API VARP_t mnn_expr_Conv(
    VARP_t     weight,
    VARP_t     bias,
    VARP_t     x,
    int        pad,
    const int *stride,
    size_t     strideLength,
    const int *dilate,
    size_t     dilateLength,
    int        group,
    const int *pads,
    size_t     padsLength
);
MNN_C_API VARP_t mnn_expr_Deconv(
    VARP_t     weight,
    VARP_t     bias,
    VARP_t     x,
    int        pad,
    const int *stride,
    size_t     strideLength,
    const int *dilate,
    size_t     dilateLength,
    int        group,
    const int *pads,
    size_t     padsLength
);
MNN_C_API VARP_t mnn_expr_MaxPool(
    VARP_t     x,
    const int *kernel,
    size_t     kernelLength,
    const int *stride,
    size_t     strideLength,
    int        pad,
    const int *pads,
    size_t     padsLength
);
MNN_C_API VARP_t mnn_expr_AvePool(
    VARP_t     x,
    const int *kernel,
    size_t     kernelLength,
    const int *stride,
    size_t     strideLength,
    int        pad,
    const int *pads,
    size_t     padsLength
);
MNN_C_API VARP_t
mnn_expr_Reshape(VARP_t x, const int *shape, size_t shapeLength, int original_format);
MNN_C_API VARP_t mnn_expr_Reshape_1(VARP_t x, VARP_t shape);
MNN_C_API VARP_t mnn_expr_Scale(
    VARP_t x, int channels, float *scales, size_t scaleLength, float *bias, size_t biasLength
);

MNN_C_API VARP_t mnn_expr_Relu(VARP_t x, float slope);
MNN_C_API VARP_t mnn_expr_Relu6(VARP_t x, float minValue, float maxValue);
MNN_C_API VARP_t mnn_expr_PRelu(VARP_t x, float *slopes, size_t slopeLength);
MNN_C_API VARP_t mnn_expr_Softmax(VARP_t logits, int axis);
MNN_C_API VARP_t mnn_expr_Softplus(VARP_t features);
MNN_C_API VARP_t mnn_expr_Softsign(VARP_t features);
MNN_C_API VecVARP_t
mnn_expr_Split(VARP_t value, const int *size_splits, size_t size_splitsLength, int axis);
MNN_C_API VARP_t mnn_expr_Slice(VARP_t x, VARP_t starts, VARP_t sizes);
MNN_C_API VARP_t mnn_expr_StridedSlice(
    VARP_t  input,
    VARP_t  begin,
    VARP_t  end,
    VARP_t  strided,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
);
MNN_C_API VARP_t mnn_expr_StridedSliceWrite(
    VARP_t  input,
    VARP_t  begin,
    VARP_t  end,
    VARP_t  strided,
    VARP_t  write,
    int32_t beginMask,
    int32_t endMask,
    int32_t ellipsisMask,
    int32_t newAxisMask,
    int32_t shrinkAxisMask
);
MNN_C_API VARP_t mnn_expr_Concat(VecVARP_t values, int axis);
MNN_C_API VARP_t mnn_expr_Convert(VARP_t input, int format);
MNN_C_API VARP_t mnn_expr_Transpose(VARP_t x, const int *perm, size_t permLength);
MNN_C_API VARP_t mnn_expr_Transpose_1(VARP_t x, VARP_t perm);
MNN_C_API VARP_t mnn_expr_ChannelShuffle(VARP_t x, int group);
MNN_C_API VARP_t mnn_expr_ChangeInputFormat(VARP_t input, int format);
MNN_C_API VARP_t mnn_expr_Reverse(VARP_t x, VARP_t axis);
MNN_C_API VARP_t mnn_expr_ReverseSequence(VARP_t x, VARP_t y, int batchDim, int seqDim);
MNN_C_API VARP_t
mnn_expr_Crop(VARP_t images, VARP_t size, int axis, const int *offset, size_t offsetLength);
MNN_C_API VARP_t mnn_expr_Resize(VARP_t images, float xScale, float yScale);
MNN_C_API VARP_t mnn_expr_Pad(VARP_t x, VARP_t paddings, int mode);
MNN_C_API VARP_t mnn_expr_ExpandDims(VARP_t input, int axis);
MNN_C_API VARP_t mnn_expr_ExpandDims_1(VARP_t input, VARP_t axis);

MNN_C_API VARP_t mnn_expr_Shape(VARP_t input, bool nchw);
MNN_C_API VARP_t mnn_expr_Stack(VecVARP_t values, int axis);
// enum InterpolationMethod {BILINEAR, NEAREST};
MNN_C_API VARP_t mnn_expr_CropAndResize(
    VARP_t image,
    VARP_t boxes,
    VARP_t box_ind,
    VARP_t crop_size,
    int    method,
    float  extrapolation_value
);
MNN_C_API VARP_t mnn_expr_Fill(VARP_t dims, VARP_t value);
MNN_C_API VARP_t mnn_expr_Tile(VARP_t input, VARP_t multiples);
MNN_C_API VARP_t mnn_expr_Gather(VARP_t params, VARP_t indices);
MNN_C_API VARP_t mnn_expr_GatherV2(VARP_t params, VARP_t indices, VARP_t axis);
MNN_C_API VARP_t mnn_expr_Squeeze(VARP_t input, const int *axis, size_t axisLength);
MNN_C_API VARP_t mnn_expr_Unsqueeze(VARP_t input, const int *axis, size_t axisLength);
MNN_C_API VARP_t mnn_expr_BatchToSpaceND(VARP_t input, VARP_t block_shape, VARP_t crops);
MNN_C_API VARP_t mnn_expr_GatherND(VARP_t params, VARP_t indices);
MNN_C_API VARP_t mnn_expr_GatherElements(VARP_t params, VARP_t indices);
MNN_C_API VARP_t mnn_expr_GatherElements_1(VARP_t params, VARP_t indices, VARP_t axis);
MNN_C_API VARP_t mnn_expr_Selu(VARP_t features, float scale, float alpha);
MNN_C_API VARP_t mnn_expr_Size(VARP_t input);
MNN_C_API VARP_t mnn_expr_Elu(VARP_t features, float alpha);
MNN_C_API VARP_t mnn_expr_Threshold(VARP_t features, float alpha);
MNN_C_API VARP_t mnn_expr_MatrixBandPart(VARP_t input, VARP_t num_lower, VARP_t num_upper);
MNN_C_API VecVARP_t
mnn_expr_Moments(VARP_t x, const int *axis, size_t axisLength, VARP_t shift, bool keepDims);
MNN_C_API VARP_t    mnn_expr_SetDiff1D(VARP_t x, VARP_t y);
MNN_C_API VARP_t    mnn_expr_SpaceToDepth(VARP_t input, int block_size);
MNN_C_API VARP_t    mnn_expr_SpaceToBatchND(VARP_t input, VARP_t block_shape, VARP_t paddings);
MNN_C_API VARP_t    mnn_expr_ZerosLike(VARP_t input);
MNN_C_API VecVARP_t mnn_expr_Unstack(VARP_t value, int axis);
MNN_C_API VARP_t    mnn_expr_Rank(VARP_t input);
MNN_C_API VARP_t    mnn_expr_Range(VARP_t start, VARP_t limit, VARP_t delta);
MNN_C_API VARP_t    mnn_expr_DepthToSpace(VARP_t input, int block_size);
// MNN_PUBLIC VARP _PriorBox(VARP feature, VARP image,
//                             std::vector<float> min_size, std::vector<float> max_size,
//                             std::vector<float>aspect_ratio, bool flip, bool clip,
//                             std::vector<float>variance, unsigned int img_h, unsigned int img_w,
//                             float step_h, float step_w, float offset = 0.5);
MNN_C_API VARP_t mnn_expr_Permute(VARP_t input, const int *dims, size_t dimsLength);
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
MNN_C_API VARP_t mnn_expr_Interp(
    VecVARP_t xs,
    float     widthScale,
    float     heightScale,
    int       outputWidth,
    int       outputHeight,
    int       resizeType,
    bool      alignCorners
);
MNN_C_API VARP_t mnn_expr_ZeroGrad(VARP_t x);

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
MNN_C_API VARP_t mnn_expr_CosineSimilarity(VARP_t input0, VARP_t input1, VARP_t inputDim);
// enum GridSamplePaddingMode {GRID_SAMPLE_PADDING_ZEROS, GRID_SAMPLE_PADDING_BORDER,
// GRID_SAMPLE_PADDING_REFLECTION};
MNN_C_API VARP_t
mnn_expr_GridSample(VARP_t input, VARP_t grid, int mode, int paddingMode, bool alignCorners);
MNN_C_API VARP_t mnn_expr_FloatToInt8(VARP_t x, VARP_t scale, char minValue, char maxValue);
MNN_C_API VARP_t
mnn_expr_FloatToInt8_1(VARP_t x, VARP_t scale, int8_t minValue, int8_t maxValue, int8_t zeroPoint);
MNN_C_API VARP_t mnn_expr_Int8ToFloat(VARP_t x, VARP_t scale);
MNN_C_API VARP_t mnn_expr_Int8ToFloat_1(VARP_t x, VARP_t scale, int8_t zeroPoint);

MNN_C_API VARP_t    mnn_expr_Select(VARP_t select, VARP_t input0, VARP_t input1);
MNN_C_API VecVARP_t mnn_expr_TopKV2(VARP_t input0, VARP_t input1);
// MNN_PUBLIC VARP _ImageProcess(VARP input, CV::ImageProcess::Config config, CV::Matrix matrix, int
// oh, int ow, int oc, int dtype, uint8_t padVal = 0); mnn_expr_VARP_t
// mnn_expr_ImageProcess(mnn_expr_VARP_t input, CV::ImageProcess::Config config, CV::Matrix matrix,
// int oh, int ow, int oc, int dtype, uint8_t padVal);
MNN_C_API VARP_t mnn_expr_Where(VARP_t x);
MNN_C_API VARP_t mnn_expr_Sort(VARP_t x, int axis, bool arg, bool descend);
MNN_C_API VARP_t mnn_expr_Raster(
    VecVARP_t vars, const int *regions, size_t regionsLength, const int *shape, size_t shapeLength
);
MNN_C_API VARP_t mnn_expr_RasterRaw(
    VecVARP_t            vars,
    const int           *region,
    size_t               regionLength,
    const int           *shape,
    size_t               shapeLength,
    struct halide_type_t dataType,
    int                  format
);

MNN_C_API VARP_t mnn_expr_Nms(
    VARP_t boxes, VARP_t scores, int maxDetections, float iouThreshold, float scoreThreshold
);
MNN_C_API VARP_t mnn_expr_Im2Col(
    VARP_t     x,
    const int *kernelSize,
    size_t     kernelSizeLength,
    const int *dilate,
    size_t     dilateLength,
    const int *pads,
    size_t     padsLength,
    const int *stride,
    size_t     strideLength
);
MNN_C_API VARP_t mnn_expr_Col2Im(
    VARP_t     x,
    VARP_t     outputShape,
    const int *kernelSize,
    size_t     kernelSizeLength,
    const int *dilate,
    size_t     dilateLength,
    const int *pads,
    size_t     padsLength,
    const int *stride,
    size_t     strideLength
);

#ifdef __cplusplus
}
#endif

#endif // MNN_EXPR_OP_H
