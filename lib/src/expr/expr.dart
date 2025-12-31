/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:mnn/src/base.dart';
import 'package:mnn/src/expr/op.dart' as op;
import 'package:mnn/src/g/mnn.g.dart' as C;
import 'package:mnn/src/halide_runtime.dart';
import 'package:mnn/src/tensor.dart';
import 'package:mnn/src/vec.dart';

import '../exception.dart';

enum MemoryType {
  COPY(0),
  MOVE(1),
  REF(2)
  ;

  final int value;
  const MemoryType(this.value);

  factory MemoryType.fromValue(int value) => switch (value) {
    0 => COPY,
    1 => MOVE,
    2 => REF,
    _ => throw ArgumentError.value(value, 'value', 'Invalid ExprMemoryType value'),
  };
}

enum DimensionFormat {
  NHWC(0),
  NC4HW4(1),
  NCHW(2)
  ;

  final int value;
  const DimensionFormat(this.value);

  factory DimensionFormat.fromValue(int value) => switch (value) {
    0 => NHWC,
    1 => NC4HW4,
    2 => NCHW,
    _ => throw ArgumentError.value(value, 'value', 'Invalid DimensionFormat value'),
  };
}

enum InputType {
  INPUT(0),
  CONSTANT(1),
  TRAINABLE(2)
  ;

  final int value;
  const InputType(this.value);

  factory InputType.fromValue(int value) => switch (value) {
    0 => INPUT,
    1 => CONSTANT,
    2 => TRAINABLE,
    _ => throw ArgumentError.value(value, 'value', 'Invalid VARPInputType value'),
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
    dim ??= [];
    final pdim = malloc<ffi.Int32>(dim.length);
    final ndim = dim.length;
    pdim.asTypedList(ndim).setAll(0, dim);
    final info = calloc<C.mnn_expr_Variable_Info>()
      ..ref.order = order.value
      ..ref.dim = pdim
      ..ref.ndim = ndim
      ..ref.type = type.native.ref
      ..ref.size = size;
    return VariableInfo.fromPointer(info);
  }

  DimensionFormat get order => DimensionFormat.fromValue(ref.order);
  List<int> get dim => ref.dim.asTypedList(ref.ndim);
  int get size => ref.size;
  HalideType get type => HalideType.fromNative(ref.type);
  int get ndim => ref.ndim;

  void syncSize() {
    var size_ = 1;
    final dim_ = dim;
    final order_ = order;
    for (int i = 0; i < dim_.length; ++i) {
      if (dim_[i] <= 0) {
        // Not valid
        size_ = 0;
        return;
      }
      if (order_ == DimensionFormat.NC4HW4 && i == 1) {
        size_ *= (dim_[1] + 4 - 1) ~/ 4 * 4;
      } else {
        size_ *= dim_[i];
      }
    }
    ref.size = size_;
  }

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
    InputType type, {
    ffi.Pointer<ffi.Void>? ptr,
    MemoryType copy = MemoryType.COPY,
  }) {
    return Expr.fromPointer(
      C.mnn_expr_Expr_static_create_1(
        info.ptr.cast(),
        ptr ?? ffi.nullptr,
        type.value,
        copy.value,
      ),
    );
  }

  String get name => C.mnn_expr_Expr_getName(ptr).cast<Utf8>().toDartString();
  set name(String name) {
    final cname = name.toNativeUtf8().cast<ffi.Char>();
    C.mnn_expr_Expr_setName(ptr, cname);
    malloc.free(cname);
  }

  //  const Op* get() const {
  //       return mOp;
  //   }

  List<VARP> get inputs {
    final p = C.mnn_expr_Expr_getInputs(ptr);
    final size = C.mnn_expr_VecVARP_size(p);
    final rval = List.generate(size, (index) => VARP.fromPointer(C.mnn_expr_VecVARP_at(p, index)));
    C.mnn_expr_VecVARP_free(p);
    return rval;
  }

  int get outputSize => C.mnn_expr_Expr_getOutputSize(ptr);
  bool get requireInfo => C.mnn_expr_Expr_requireInfo(ptr);

  InputType get inputType => InputType.fromValue(C.mnn_expr_Expr_inputType(ptr));
  String outputName(int index) => C.mnn_expr_Expr_getOutputName(ptr, index).cast<Utf8>().toDartString();

  static void replace(Expr old, Expr newExpr) {
    C.mnn_expr_Expr_static_replace(old.ptr, newExpr.ptr);
  }

  //   void visitOutputs(const std::function<bool(EXPRP, int)>& visit);
  //   static void visit(EXPRP expr, const std::function<bool(EXPRP)>& before, const std::function<bool(EXPRP)>& after);

  //   const std::vector<WeakEXPRP>& outputs() const {
  //       return mTo;
  //   }
  // List<WeakReference<Expr>> get outputs {
  //   final p = C.mnn_expr_Expr_getOutputs(ptr);
  //   final rval = List.generate(p.ref.size, (index) => WeakReference(Expr.fromPointer(p.ref.ptr.cast<C.mnn_expr_Expr_t>()[index])));
  //   malloc.free(p);
  //   return rval;
  // }

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

  @override
  String toString() {
    return "Expr(name=$name, inputType=$inputType, outputSize=$outputSize, requireInfo=$requireInfo)";
  }
}

class VARP extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_VARP_free);

  VARP.fromPointer(C.VARP_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory VARP.empty() => VARP.fromPointer(C.mnn_expr_VARP_create_empty());

  factory VARP.copyFrom(VARP varp) => VARP.fromPointer(C.mnn_expr_VARP_create_VARP(varp.ptr));

  factory VARP.create(Expr expr, {int index = 0}) =>
      VARP.fromPointer(C.mnn_expr_VARP_static_create_EXPRP(expr.ptr, index));

  static VARP scalar<T extends ffi.SizedNativeType>(num value) => op.scalar<T>(value);

  static VARP fromList1D<T extends ffi.SizedNativeType>(
    Iterable<num> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) => op.constant<T>(data, [data.length], format: format);

  static VARP fromList2D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<num>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) => op.constant<T>(data.expand((e) => e), [data.length, data.first.length], format: format);

  static VARP fromList3D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<Iterable<num>>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) => op.constant<T>(
    data.expand((e) => e.expand((e) => e)),
    [data.length, data.first.length, data.first.first.length],
    format: format,
  );

  static VARP fromList4D<T extends ffi.SizedNativeType>(
    Iterable<Iterable<Iterable<Iterable<num>>>> data, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) => op.constant<T>(
    data.expand((e) => e.expand((e) => e.expand((e) => e))),
    [data.length, data.first.length, data.first.first.length, data.first.first.first.length],
    format: format,
  );

  static VARP fromListND<T extends ffi.SizedNativeType>(
    Iterable<num> data,
    Iterable<int> shape, {
    DimensionFormat format = DimensionFormat.NHWC,
  }) => op.constant<T>(data, shape, format: format);

  VARP astype<T extends ffi.SizedNativeType>({HalideType? dtype}) => op.cast<T>(this, dtype: dtype);

  String getName() => C.mnn_expr_VARP_getName(ptr).cast<Utf8>().toDartString();
  void setName(String name) {
    final cname = name.toNativeUtf8().cast<ffi.Char>();
    C.mnn_expr_VARP_setName(ptr, cname);
    malloc.free(cname);
  }

  bool setDevicePtr(ffi.Pointer<ffi.Void> devicePtr, int memoryType) =>
      C.mnn_expr_VARP_setDevicePtr(ptr, devicePtr, memoryType);
  bool copyToDevicePtr(ffi.Pointer<ffi.Void> devicePtr, int memoryType) =>
      C.mnn_expr_VARP_copyToDevicePtr(ptr, devicePtr, memoryType);

  // this will return a new shared_ptr of internel shared_ptr<Expr>
  // the refcount will increase and safe to dispose
  (Expr expr, int index) get expr {
    final p = C.mnn_expr_VARP_getExpr(ptr);
    final rval = (Expr.fromPointer(p.ref.expr), p.ref.index);
    malloc.free(p);
    return rval;
  }

  bool resize(List<int> dims) {
    final cdims = dims.i32;
    final rval = C.mnn_expr_VARP_resize(ptr, cdims.ptr);
    cdims.dispose();
    return rval;
  }

  ffi.Pointer<T> readMap<T extends ffi.NativeType>() => C.mnn_expr_VARP_readMap(ptr).cast<T>();
  ffi.Pointer<T> writeMap<T extends ffi.NativeType>() => C.mnn_expr_VARP_writeMap(ptr).cast<T>();

  bool input(VARP src) => C.mnn_expr_VARP_input(ptr, src.ptr);

  static void replace(VARP dst, VARP src) => C.mnn_expr_VARP_static_replace(dst.ptr, src.ptr);

  // static std::vector<VARP> load(const char* fileName);
  static List<VARP> loadFromFile(String fileName) {
    final cname = fileName.toNativeUtf8().cast<ffi.Char>();
    final p = C.mnn_expr_VARP_static_load(cname);
    malloc.free(cname);
    final size = C.mnn_expr_VecVARP_size(p);
    final rval = List.generate(size, (index) => VARP.fromPointer(C.mnn_expr_VecVARP_at(p, index)));
    C.mnn_expr_VecVARP_free(p);
    return rval;
  }

  // static std::map<std::string, VARP> loadMap(const char* fileName);
  static Map<String, VARP> loadMapFromFile(String fileName) {
    final cname = fileName.toNativeUtf8().cast<ffi.Char>();
    final p = C.mnn_expr_VARP_static_loadMap(cname);
    malloc.free(cname);
    final varmap = VarMap.fromPointer(p);
    final rval = varmap.toMap();
    varmap.dispose();
    return rval;
  }

  // static std::vector<VARP> load(const uint8_t* buffer, size_t length);
  static List<VARP> loadFromBuffer(Uint8List buffer) {
    final pBuf = malloc<ffi.Uint8>(buffer.length)..asTypedList(buffer.length).setAll(0, buffer);
    final p = C.mnn_expr_VARP_static_loadBuffer(pBuf, buffer.length);
    malloc.free(pBuf);

    final size = C.mnn_expr_VecVARP_size(p);
    final rval = List.generate(size, (index) => VARP.fromPointer(C.mnn_expr_VecVARP_at(p, index)));
    C.mnn_expr_VecVARP_free(p);
    return rval;
  }

  // static std::map<std::string, VARP> loadMap(const uint8_t* buffer, size_t length);
  static Map<String, VARP> loadMapFromBuffer(Uint8List buffer) {
    final pBuf = malloc<ffi.Uint8>(buffer.length)..asTypedList(buffer.length).setAll(0, buffer);
    final p = C.mnn_expr_VARP_static_loadMapBuffer(pBuf, buffer.length);
    malloc.free(pBuf);

    final varmap = VarMap.fromPointer(p);
    final rval = varmap.toMap();
    varmap.dispose();
    return rval;
  }

  // static std::pair<std::map<std::string, VARP>, std::map<std::string, VARP>> getInputAndOutput(const std::map<std::string, VARP>& allVariable);

  // static std::vector<VARP> mapToSequence(const std::map<std::string, VARP>& source);

  // static std::vector<EXPRP> getExecuteOrder(const std::vector<VARP>& output);

  // static void save(const std::vector<VARP>& vars, const char* fileName);
  static void saveToFile(List<VARP> vars, String fileName) {
    final cname = fileName.toNativeUtf8().cast<ffi.Char>();
    final cVars = vars.toNativeVec();
    C.mnn_expr_VARP_static_save(cVars.ptr, cname);
    malloc.free(cname);
    cVars.dispose();
  }

  // static std::vector<int8_t> save(const std::vector<VARP>& vars);
  static VecI8 saveToBuffer(List<VARP> vars) {
    final cVars = vars.toNativeVec();
    final p = C.mnn_expr_VARP_static_saveBytes(cVars.ptr);
    cVars.dispose();
    return VecI8.fromPointer(p);
  }

  // static void save(const std::vector<VARP>& vars, NetT* dest);

  // Pack a few Variable to compute in one pipeline
  // static void prepareCompute(const std::vector<VARP>& vars, bool forceCPU = false);
  // static void compute(const std::vector<VARP>& vars, bool forceCPU = false);

  int linkNumber() => C.mnn_expr_VARP_linkNumber(ptr);

  // const std::vector<WeakEXPRP>& toExprs() const;

  void setExpr(Expr expr, int index) => C.mnn_expr_VARP_setExpr(ptr, expr.ptr, index);

  Tensor getTensor() => Tensor.fromPointer(C.mnn_expr_VARP_getTensor(ptr), attach: false);

  VariableInfo? get info {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    return p == ffi.nullptr ? null : VariableInfo.fromPointer(p);
  }

  VariableInfo? getInfo() => info;

  DimensionFormat? get order {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    final rval = p == ffi.nullptr ? null : DimensionFormat.fromValue(p.ref.order);
    C.mnn_expr_Variable_Info_free(p.cast());
    return rval;
  }

  DimensionFormat? get dataFormat => order;

  List<int>? get dim {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    final rval = p == ffi.nullptr ? null : p.ref.dim.asTypedList(p.ref.ndim).toList();
    C.mnn_expr_Variable_Info_free(p.cast());
    return rval;
  }

  List<int>? get shape => dim;
  int? get ndim {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    final rval = p == ffi.nullptr ? null : p.ref.ndim;
    C.mnn_expr_Variable_Info_free(p.cast());
    return rval;
  }

  HalideType? get dtype {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    final rval = p == ffi.nullptr ? null : HalideType.fromNative(p.ref.type);
    C.mnn_expr_Variable_Info_free(p.cast());
    return rval;
  }

  int? get size {
    final p = C.mnn_expr_VARP_getInfo(ptr);
    final rval = p == ffi.nullptr ? null : p.ref.size;
    C.mnn_expr_Variable_Info_free(p.cast());
    return rval;
  }

  List<num>? get data {
    final info = this.info;
    if (info == null || (info.isEmpty) || info.size <= 0) {
      return null;
    }
    final List<num> dataList = switch (dtype) {
      HalideType.f32 => readMap().cast<ffi.Float>().asTypedList(info.size),
      HalideType.f64 => readMap().cast<ffi.Double>().asTypedList(info.size),
      HalideType.i32 => readMap().cast<ffi.Int32>().asTypedList(info.size),
      HalideType.u8 => readMap().cast<ffi.Uint8>().asTypedList(info.size),
      HalideType.i8 => readMap().cast<ffi.Int8>().asTypedList(info.size),
      HalideType.i16 => readMap().cast<ffi.Int16>().asTypedList(info.size),
      HalideType.u16 => readMap().cast<ffi.Uint16>().asTypedList(info.size),
      HalideType.i64 => readMap().cast<ffi.Int64>().asTypedList(info.size),
      HalideType.u64 => readMap().cast<ffi.Uint64>().asTypedList(info.size),
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

  /// compare the two VARP whether they point to the same underlying expression
  @override
  bool operator ==(Object other) => other is VARP && C.mnn_expr_VARP_op_eqeq(ptr, other.ptr);
  @override
  int get hashCode => ptr.address.hashCode;

  /// compare the objects by their underlying expression address
  bool operator <(VARP other) => C.mnn_expr_VARP_op_less(ptr, other.ptr);

  /// compare the objects by their underlying expression address
  bool operator <=(VARP other) => C.mnn_expr_VARP_op_lessequal(ptr, other.ptr);

  ({VARP begin, VARP end, VARP strides, int beginMask, int endMask, int shrinkAxisMask}) _parseSliceParams(
    String index,
  ) {
    final parts = index.split(',').map((e) => e.trim()).toList();
    final begins = <int>[];
    final ends = <int>[];
    final strides = <int>[];
    var beginMask = 0;
    var endMask = 0;
    var shrinkAxisMask = 0;

    for (int i = 0; i < parts.length; i++) {
      final part = parts[i];
      if (part.contains(':')) {
        final slices = part.split(':');

        if (slices[0].isEmpty) {
          beginMask |= 1 << i;
          begins.add(0);
        } else {
          begins.add(int.parse(slices[0]));
        }

        if (slices.length > 1) {
          if (slices[1].isEmpty) {
            endMask |= 1 << i;
            ends.add(0);
          } else {
            ends.add(int.parse(slices[1]));
          }
        } else {
          // Should not happen for valid slice syntax with ':'
        }

        if (slices.length > 2) {
          if (slices[2].isEmpty) {
            strides.add(1);
          } else {
            strides.add(int.parse(slices[2]));
          }
        } else {
          strides.add(1);
        }
      } else {
        shrinkAxisMask |= 1 << i;
        final idx = int.parse(part);
        begins.add(idx);
        ends.add(idx + 1);
        strides.add(1);
      }
    }

    return (
      begin: VARP.fromList1D<ffi.Int32>(begins),
      end: VARP.fromList1D<ffi.Int32>(ends),
      strides: VARP.fromList1D<ffi.Int32>(strides),
      beginMask: beginMask,
      endMask: endMask,
      shrinkAxisMask: shrinkAxisMask,
    );
  }

  VARP operator [](dynamic index) {
    if (index is int) {
      return op.gather(this, VARP.scalar<ffi.Int32>(index));
    } else if (index is String) {
      final params = _parseSliceParams(index);
      return op.stridedSlice(
        this,
        params.begin,
        params.end,
        params.strides,
        params.beginMask,
        params.endMask,
        0,
        0,
        params.shrinkAxisMask,
      );
    }
    throw ArgumentError.value(index, 'index', 'Index must be int or String');
  }

  void operator []=(dynamic index, VARP value) {
    if (!(index is int || index is String)) {
      throw ArgumentError.value(index, 'index', 'Index must be int or String');
    }
    final params = _parseSliceParams(index is int ? index.toString() : index as String);

    final target = op.stridedSlice(
      this,
      params.begin,
      params.end,
      params.strides,
      params.beginMask,
      params.endMask,
      0,
      0,
      params.shrinkAxisMask,
    );
    final tShape = target.shape;
    final vShape = value.shape;
    target.release();

    bool shapeMatch = tShape != null && vShape != null && tShape.length == vShape.length;
    if (shapeMatch) {
      for (int i = 0; i < tShape.length; i++) {
        if (tShape[i] != vShape[i]) {
          shapeMatch = false;
          break;
        }
      }
    }
    MnnAssert(
      shapeMatch,
      "value shape $vShape is not equal to target slice shape $tShape",
    );

    final newPtr = C.mnn_expr_StridedSliceWrite(
      ptr,
      params.begin.ptr,
      params.end.ptr,
      params.strides.ptr,
      value.ptr,
      params.beginMask,
      params.endMask,
      0,
      0,
      params.shrinkAxisMask,
    );
    // replace this.ptr with newPtr
    reattach(newPtr);
  }

  dynamic toList() {
    final info = this.info;
    final data = this.data;
    if (info == null || info.isEmpty || data == null) {
      return data?[0];
    }
    if (info.ndim == 1) {
      return data;
    }
    return _reshape(data, info.dim);
  }

  dynamic _reshape(List<num> flat, List<int> shape) {
    if (shape.length == 1) {
      return flat;
    }
    final int size = shape[0];
    final int blockSize = flat.length ~/ size;
    final List<dynamic> res = [];
    final List<int> subShape = shape.sublist(1);
    for (int i = 0; i < size; i++) {
      res.add(_reshape(flat.sublist(i * blockSize, (i + 1) * blockSize), subShape));
    }
    return res;
  }

  num get value {
    final info = this.info;
    if (info == null || info.isEmpty) {
      throw MNNException("$this is empty");
    }
    if (info.size != 1) {
      throw StateError(
        'VARP.value can only be called on scalar or size-1 tensors. Current size: ${info.size}',
      );
    }
    final data = this.data;
    if (data == null || data.isEmpty) {
      throw MNNException("$this is empty");
    }
    return data[0];
  }

  num item() => value;

  bool fix(InputType type) => C.mnn_expr_VARP_fix(ptr, type.value);

  void setOrder(DimensionFormat format) => C.mnn_expr_VARP_setOrder(ptr, format.value);

  @override
  String toString() {
    final s = StringBuffer('VARP(address=0x${ptr.address.toRadixString(16)}');
    if (!(info?.isEmpty ?? true)) {
      s.write(', shape=$shape, dtype=$dtype, size=$size');
    } else {
      s.write(', shape=(), dtype=(), size=0');
    }
    s.write(')');
    return s.toString();
  }
}

class VecVARP extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_VecVARP_free);

  VecVARP.fromPointer(C.VecVARP_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory VecVARP.create({int size = 0, VARP? value}) {
    final p = C.mnn_expr_VecVARP_create(size, value?.ptr ?? ffi.nullptr);
    return VecVARP.fromPointer(p);
  }

  factory VecVARP.of(List<VARP> vars) {
    final p = C.mnn_expr_VecVARP_create(0, ffi.nullptr);
    for (int i = 0; i < vars.length; i++) {
      C.mnn_expr_VecVARP_push_back(p, vars[i].ptr);
    }
    return VecVARP.fromPointer(p);
  }

  int size() => C.mnn_expr_VecVARP_size(ptr);
  int get length => C.mnn_expr_VecVARP_size(ptr);

  List<VARP> toList() => List.generate(size(), (i) => VARP.fromPointer(C.mnn_expr_VecVARP_at(ptr, i)));
  List<VARP> asList() => List.generate(size(), (i) => VARP.fromPointer(C.mnn_expr_VecVARP_at_ref(ptr, i)));

  VARP at(int index) => VARP.fromPointer(C.mnn_expr_VecVARP_at(ptr, index));
  VARP atRef(int index) => VARP.fromPointer(C.mnn_expr_VecVARP_at_ref(ptr, index));
  void set(int index, VARP varp) => C.mnn_expr_VecVARP_set(ptr, index, varp.ptr);
  void push_back(VARP varp) => C.mnn_expr_VecVARP_push_back(ptr, varp.ptr);

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_VecVARP_free(ptr);
  }

  @override
  String toString() {
    return "VecVARP(address=0x${ptr.address.toRadixString(16)})";
  }
}

class VarMap extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(C.addresses.mnn_expr_VecVARP_free);

  VarMap.fromPointer(C.VARMAP_t ptr, {super.attach, super.externalSize}) : super(ptr.cast());

  factory VarMap.create() {
    final p = C.mnn_expr_VARMAP_create();
    return VarMap.fromPointer(p);
  }

  factory VarMap.of(Map<String, VARP> vars) {
    final p = C.mnn_expr_VARMAP_create();
    for (final MapEntry(key: key, value: value) in vars.entries) {
      final ckey = key.toNativeUtf8().cast<ffi.Char>();
      C.mnn_expr_VARMAP_set(p, ckey, value.ptr);
      malloc.free(ckey);
    }
    return VarMap.fromPointer(p);
  }

  int size() => C.mnn_expr_VARMAP_size(ptr);
  int get length => C.mnn_expr_VARMAP_size(ptr);

  Map<String, VARP> toMap() {
    final pKeys = C.mnn_expr_VARMAP_keys(ptr);
    if (pKeys == ffi.nullptr) {
      return {};
    }
    final map = <String, VARP>{};
    for (int i = 0; i < length; i++) {
      final cKey = pKeys[i];
      if (cKey == ffi.nullptr) {
        continue;
      }
      final key = cKey.cast<Utf8>().toDartString();
      final value = VARP.fromPointer(C.mnn_expr_VARMAP_get(ptr, cKey));
      map[key] = value;

      malloc.free(cKey);
      cKey.value = 0;
    }
    malloc.free(pKeys);
    return map;
  }

  VARP at(String key) {
    final ckey = key.toNativeUtf8().cast<ffi.Char>();
    final rval = VARP.fromPointer(C.mnn_expr_VARMAP_get(ptr, ckey));
    malloc.free(ckey);
    return rval;
  }

  VARP atRef(String key) {
    final ckey = key.toNativeUtf8().cast<ffi.Char>();
    final rval = VARP.fromPointer(C.mnn_expr_VARMAP_get_ref(ptr, ckey));
    malloc.free(ckey);
    return rval;
  }

  void set(String key, VARP varp) {
    final ckey = key.toNativeUtf8().cast<ffi.Char>();
    C.mnn_expr_VARMAP_set(ptr, ckey, varp.ptr);
    malloc.free(ckey);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  void release() {
    finalizer.detach(this);
    C.mnn_expr_VARMAP_free(ptr);
  }

  @override
  String toString() {
    return "VarMap(address=0x${ptr.address.toRadixString(16)})";
  }
}

extension ListVarpExtension on List<VARP> {
  VecVARP toNativeVec() => VecVARP.of(this);
}
