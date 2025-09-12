/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';
import 'package:mnn/src/base.dart';
import 'package:mnn/src/expr/op.dart' as op;
import 'package:mnn/src/expr/utils.dart';
import 'package:mnn/src/g/mnn.g.dart' as C;
import 'package:mnn/src/halide_runtime.dart';
import 'package:mnn/src/tensor.dart';
import 'package:mnn/src/vec.dart';

enum MemoryType {
  COPY(0),
  MOVE(1),
  REF(2);

  final int value;
  const MemoryType(this.value);

  factory MemoryType.fromValue(int value) => switch (value) {
        0 => COPY,
        1 => MOVE,
        2 => REF,
        _ => throw ArgumentError.value(value, 'value', 'Invalid ExprMemoryType value')
      };
}

enum DimensionFormat {
  NHWC(0),
  NC4HW4(1),
  NCHW(2);

  final int value;
  const DimensionFormat(this.value);

  factory DimensionFormat.fromValue(int value) => switch (value) {
        0 => NHWC,
        1 => NC4HW4,
        2 => NCHW,
        _ => throw ArgumentError.value(value, 'value', 'Invalid DimensionFormat value')
      };
}

enum InputType {
  INPUT(0),
  CONSTANT(1),
  TRAINABLE(2);

  final int value;
  const InputType(this.value);

  factory InputType.fromValue(int value) => switch (value) {
        0 => INPUT,
        1 => CONSTANT,
        2 => TRAINABLE,
        _ => throw ArgumentError.value(value, 'value', 'Invalid VARPInputType value')
      };
}

class VariableInfo extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_Variable_Info_free);

  VariableInfo.fromPointer(ffi.Pointer<C.mnn_expr_Variable_Info> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  factory VariableInfo.create({
    DimensionFormat order = DimensionFormat.NHWC,
    List<int>? dim,
    HalideType type = HalideType.f32,
    int size = 0,
  }) {
    final (pdim, ndim) = dim?.toNativeArrayI32() ?? (ffi.nullptr, 0);
    final info = calloc<C.mnn_expr_Variable_Info>()
      ..ref.order = order.value
      ..ref.dim = pdim
      ..ref.ndim = ndim
      ..ref.type = type.native.ref
      ..ref.size = size;
    return VariableInfo.fromPointer(info);
  }

  DimensionFormat get order => DimensionFormat.fromValue(ref.order);
  List<int> get dim => ptr.cast<C.mnn_expr_Variable_Info>().ref.dim.asTypedList(ref.ndim);
  int get size => ref.size;
  HalideType get type => HalideType.fromNative(ref.type);
  int get ndim => ref.ndim;

  C.mnn_expr_Variable_Info get ref => ptr.cast<C.mnn_expr_Variable_Info>().ref;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_Variable_Info_free(ptr);
  }

  @override
  String toString() {
    return "VariableInfo(order=$order, dim=$dim, size=$size, type=$type, ndim=$ndim)";
  }
}

class Expr extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_Expr_free);

  Expr.fromPointer(C.mnn_expr_Expr_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory Expr.empty() => Expr.fromPointer(C.mnn_expr_Expr_create_empty());

  factory Expr.fromTensor(Tensor tensor, {bool own = false}) =>
      Expr.fromPointer(C.mnn_expr_Expr_static_create(tensor.ptr, own));

  factory Expr.fromVariableInfo(
    VariableInfo info,
    ffi.Pointer<ffi.Void> ptr,
    InputType type, {
    MemoryType copy = MemoryType.COPY,
  }) {
    return Expr.fromPointer(C.mnn_expr_Expr_static_create_1(info.ptr.cast(), ptr, type.value, copy.value));
  }

  String get name => C.mnn_expr_Expr_getName(ptr).cast<Utf8>().toDartString();
  set name(String name) {
    final cname = name.toNativeUtf8().cast<ffi.Char>();
    C.mnn_expr_Expr_setName(ptr, cname);
    calloc.free(cname);
  }

  //  const Op* get() const {
  //       return mOp;
  //   }
  //   const std::vector<VARP>& inputs() const {
  //       return mInputs;
  //   }

  int get outputSize => C.mnn_expr_Expr_getOutputSize(ptr);
  bool get requireInfo => C.mnn_expr_Expr_requireInfo(ptr);

  InputType get inputType => InputType.fromValue(C.mnn_expr_Expr_inputType(ptr));
  String outputName(int index) => C.mnn_expr_Expr_getOutputName(ptr, index).cast<Utf8>().toDartString();

  // static void replace(Expr oldExpr, Expr newExpr) {
  //   c.mnn_expr_Expr_static_replace(oldExpr.ptr, newExpr.ptr);
  // }

  //   void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
  //   static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);

  //   const std::vector<WeakEXPRP>& outputs() const {
  //       return mTo;
  //   }

  VariableInfo outputInfo(int index) => VariableInfo.fromPointer(C.mnn_expr_Expr_outputInfo(ptr, index));

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_Expr_free(ptr);
  }
}

class Variable extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_Variable_free);

  Variable.fromPointer(C.mnn_expr_Variable_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  static VARP create(Expr expr, {int index = 0}) =>
      VARP.fromPointer(C.mnn_expr_Variable_static_create_EXPRP(expr.ptr, index));

  String get name => C.mnn_expr_Variable_getName(ptr).cast<Utf8>().toDartString();
  set name(String name) {
    final cname = name.toNativeUtf8().cast<ffi.Char>();
    C.mnn_expr_Variable_setName(ptr, cname);
    calloc.free(cname);
  }

  bool setDevicePtr(ffi.Pointer<ffi.Void> devicePtr, int memoryType) =>
      C.mnn_expr_Variable_setDevicePtr(ptr, devicePtr, memoryType);
  bool copyToDevicePtr(ffi.Pointer<ffi.Void> devicePtr, int memoryType) =>
      C.mnn_expr_Variable_copyToDevicePtr(ptr, devicePtr, memoryType);

  // std::pair<EXPRP, int> expr() const {
  //     return std::make_pair(mFrom, mFromIndex);
  // }

  VariableInfo getInfo() {
    return VariableInfo.fromPointer(C.mnn_expr_Variable_getInfo(ptr));
  }

  bool resize(List<int> dims) {
    final cdims = dims.i32;
    final rval = C.mnn_expr_Variable_resize(ptr, cdims.ptr);
    cdims.dispose();
    return rval;
  }

  ffi.Pointer<ffi.Void> readMap() => C.mnn_expr_Variable_readMap(ptr);
  ffi.Pointer<ffi.Void> writeMap() => C.mnn_expr_Variable_writeMap(ptr);

  void unMap() => C.mnn_expr_Variable_unMap(ptr);

  bool input(VARP src) => C.mnn_expr_Variable_input(ptr, src.ptr);

  static void replace(VARP dst, VARP src) => C.mnn_expr_Variable_static_replace(dst.ptr, src.ptr);

  // static std::vector<VARP> load(const char* fileName);
  // static std::map<std::string, VARP> loadMap(const char* fileName);
  // static std::vector<VARP> load(const uint8_t* buffer, size_t length);
  // static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
  // static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);
  // static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);
  // static std::vector<EXPRP> getExecuteOrder(const std::vector<VARP>& output);
  // static void save(const std::vector<VARP>& vars, const char* fileName);
  // static std::vector<int8_t> save(const std::vector<VARP>& vars);
  // static void save(const std::vector<VARP>& vars, NetT* dest);

  // Pack a few Variable to compute in one pipeline
  // static void prepareCompute(const std::vector<VARP>& vars, bool forceCPU = false);
  // static void compute(const std::vector<VARP>& vars, bool forceCPU = false);

  int linkNumber() => C.mnn_expr_Variable_linkNumber(ptr);

  // const std::vector<WeakEXPRP>& toExprs() const;

  void setExpr(Expr expr, int index) => C.mnn_expr_Variable_setExpr(ptr, expr.ptr, index);

  Tensor getTensor() => Tensor.fromPointer(C.mnn_expr_Variable_getTensor(ptr), attach: false);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_VARP_free(ptr);
  }
}

class VARP extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_VARP_free);

  VARP.fromPointer(C.mnn_expr_VARP_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory VARP.empty() => VARP.fromPointer(C.mnn_expr_VARP_create_empty());

  factory VARP.copyFrom(VARP varp) => VARP.fromPointer(C.mnn_expr_VARP_create_VARP(varp.ptr));

  factory VARP.fromVariable(Variable variable) =>
      VARP.fromPointer(C.mnn_expr_VARP_create_variable(variable.ptr));

  static VARP scalar<T extends ffi.SizedNativeType>(num value) => op.scalar<T>(value);

  static VARP list<T extends ffi.SizedNativeType>(
    Iterable<num> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) =>
      op.constant<T>(data, [data.length], format: format);

  static VARP list2D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<num>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) =>
      op.constant<T>(data.expand((e) => e), [data.length, data.first.length], format: format);

  static VARP list3D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<Iterable<num>>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) =>
      op.constant<T>(
        data.expand((e) => e.expand((e) => e)),
        [data.length, data.first.length, data.first.first.length],
        format: format,
      );

  static VARP list4D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<Iterable<Iterable<num>>>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) =>
      op.constant<T>(
        data.expand((e) => e.expand((e) => e.expand((e) => e))),
        [data.length, data.first.length, data.first.first.length, data.first.first.first.length],
        format: format,
      );

  static VARP listND<T extends ffi.SizedNativeType>(
    Iterable<num> data,
    Iterable<int> shape, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) =>
      op.constant<T>(data, shape, format: format);

  VARP astype<T extends ffi.SizedNativeType>() => op.cast<T>(this);

  static HalideType dtypeOf<T extends ffi.SizedNativeType>() {
    if (T == ffi.Float) {
      return HalideType.f32;
    } else if (T == ffi.Double) {
      return HalideType.f64;
    } else if (T == ffi.Bool) {
      return HalideType.bool_;
    } else if (T == ffi.Uint8) {
      return HalideType.u8;
    } else if (T == ffi.Uint16) {
      return HalideType.u16;
    } else if (T == ffi.Uint32) {
      return HalideType.u32;
    } else if (T == ffi.Uint64) {
      return HalideType.u64;
    } else if (T == ffi.Int8) {
      return HalideType.i8;
    } else if (T == ffi.Int16) {
      return HalideType.i16;
    } else if (T == ffi.Int32) {
      return HalideType.i32;
    } else if (T == ffi.Int64) {
      return HalideType.i64;
    } else {
      throw ArgumentError('Unsupported type $T');
    }
  }

  ffi.Pointer<ffi.Void> readMap() => C.mnn_expr_VARP_readMap(ptr);
  ffi.Pointer<ffi.Void> writeMap() => C.mnn_expr_VARP_writeMap(ptr);
  void unMap() => C.mnn_expr_VARP_unMap(ptr);

  VariableInfo get info => VariableInfo.fromPointer(C.mnn_expr_VARP_getInfo(ptr));

  DimensionFormat get order => info.order;
  List<int> get dim => info.dim;
  int get ndim => info.ndim;
  HalideType get dtype => info.type;
  int get size => info.size;

  List<num> get data {
    final List<num> dataList = switch (dtype) {
      HalideType.f32 => readMap().cast<ffi.Float>().asTypedList(size),
      HalideType.f64 => readMap().cast<ffi.Double>().asTypedList(size),
      HalideType.i32 => readMap().cast<ffi.Int32>().asTypedList(size),
      HalideType.u8 => readMap().cast<ffi.Uint8>().asTypedList(size),
      HalideType.i8 => readMap().cast<ffi.Int8>().asTypedList(size),
      HalideType.i16 => readMap().cast<ffi.Int16>().asTypedList(size),
      HalideType.u16 => readMap().cast<ffi.Uint16>().asTypedList(size),
      HalideType.i64 => readMap().cast<ffi.Int64>().asTypedList(size),
      HalideType.u64 => readMap().cast<ffi.Uint64>().asTypedList(size),
      _ => throw UnimplementedError('Data type $dtype not supported'),
    };
    return dataList;
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_VARP_free(ptr);
  }

  VARP operator +(VARP other) => VARP.fromPointer(C.mnn_expr_VARP_op_add(ptr, other.ptr));

  VARP operator -(VARP other) => VARP.fromPointer(C.mnn_expr_VARP_op_sub(ptr, other.ptr));

  VARP operator *(VARP other) => VARP.fromPointer(C.mnn_expr_VARP_op_mul(ptr, other.ptr));

  VARP operator /(VARP other) => VARP.fromPointer(C.mnn_expr_VARP_op_div(ptr, other.ptr));

  VARP mean(List<int> dims) {
    final cdims = dims.i32;
    return VARP.fromPointer(C.mnn_expr_VARP_mean(ptr, cdims.ptr));
  }

  VARP sum(List<int> dims) {
    final cdims = dims.i32;
    return VARP.fromPointer(C.mnn_expr_VARP_sum(ptr, cdims.ptr));
  }

  Variable get variable => Variable.fromPointer(C.mnn_expr_VARP_get(ptr), attach: false);

  @override
  bool operator ==(Object other) => other is VARP && C.mnn_expr_VARP_op_eqeq(ptr, other.ptr);
  @override
  int get hashCode => ptr.address.hashCode;

  bool operator <(VARP other) => C.mnn_expr_VARP_op_less(ptr, other.ptr);
  bool operator <=(VARP other) => C.mnn_expr_VARP_op_lessequal(ptr, other.ptr);

  bool fix(InputType type) => C.mnn_expr_VARP_fix(ptr, type.value);

  void setOrder(DimensionFormat format) => C.mnn_expr_VARP_setOrder(ptr, format.value);
}
