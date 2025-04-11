/// Copyright (c) 2025, rainyl. All rights reserved.
/// Use of this source code is governed by a
/// Apache 2.0 license that can be found in the LICENSE file.

import 'dart:ffi' as ffi;

import 'package:ffi/ffi.dart';

import 'base.dart';
import 'exception.dart';
import 'g/mnn.g.dart' as c;
import 'rect.dart';

class Matrix extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(c.addresses.mnn_cv_matrix_destroy);

  Matrix.fromPointer(ffi.Pointer<c.mnn_cv_matrix_t> ptr, {super.attach, super.externalSize})
      : super(ptr.cast());

  factory Matrix.create() {
    final ptr = c.mnn_cv_matrix_create();
    return Matrix.fromPointer(ptr.cast());
  }

  /// Sets Matrix to scale by (sx, sy). Returned matrix is:
  ///
  ///     | sx  0  0 |
  ///     |  0 sy  0 |
  ///     |  0  0  1 |
  ///
  /// @param sx  horizontal scale factor
  /// @param sy  vertical scale factor
  /// @return    Matrix with scale
  factory Matrix.makeScaleXY({required double sx, double sy = 1}) {
    final m = Matrix.create();
    m.setScale(sx, sy);
    return m;
  }

  /// Sets Matrix to scale by (scale, scale). Returned matrix is:
  ///
  ///     | scale   0   0 |
  ///     |   0   scale 0 |
  ///     |   0     0   1 |
  ///
  /// @param scale  horizontal and vertical scale factor
  /// @return       Matrix with scale
  factory Matrix.makeScale(double scale) => Matrix.makeScaleXY(sx: scale, sy: scale);

  /// Sets Matrix to translate by (dx, dy). Returned matrix is:
  ///
  ///     | 1 0 dx |
  ///     | 0 1 dy |
  ///     | 0 0  1 |
  ///
  /// @param dx  horizontal translation
  /// @param dy  vertical translation
  /// @return    Matrix with translation
  factory Matrix.makeTrans(double dx, double dy) {
    final m = Matrix.create();
    m.setTranslate(dx, dy);
    return m;
  }

  factory Matrix.makeAll(
    double scaleX,
    double skewX,
    double transX,
    double skewY,
    double scaleY,
    double transY,
    double persp0,
    double persp1,
    double persp2,
  ) {
    final m = Matrix.create();
    m.setAll(
      scaleX: scaleX,
      scaleY: scaleY,
      skewX: skewX,
      skewY: skewY,
      translateX: transX,
      translateY: transY,
      persp0: persp0,
      persp1: persp1,
      persp2: persp2,
    );
    return m;
  }

  /// Returns Matrix set to scale and translate src Rect to dst Rect. stf selects
  /// whether mapping completely fills dst or preserves the aspect ratio, and how to
  /// align src within dst. Returns the identity Matrix if src is empty. If dst is
  /// empty, returns Matrix set to:
  ///
  ///     | 0 0 0 |
  ///     | 0 0 0 |
  ///     | 0 0 1 |
  ///
  /// @param src  Rect to map from
  /// @param dst  Rect to map to
  /// @param stf  one of: kFill_ScaleToFit, kStart_ScaleToFit,
  ///             kCenter_ScaleToFit, kEnd_ScaleToFit
  /// @return     Matrix mapping src to dst
  factory Matrix.makeRectToRect(Rect src, Rect dst, ScaleToFit scaleToFit) {
    final result = Matrix.create();
    result.setRectToRect(src, dst, scaleToFit);
    return result;
  }

  factory Matrix.I() {
    final result = Matrix.create();
    result.setIdentity();
    return result;
  }

  factory Matrix.concat(Matrix a, Matrix b) {
    final result = Matrix.create();
    result.setConcat(a, b);
    return result;
  }

  /// Returns a bit field describing the transformations the matrix may
  /// perform. The bit field is computed conservatively, so it may include
  /// false positives.
  ///
  /// For example, when kPerspective_Mask is set, all other bits are set.
  ///
  /// [TypeMask.kIdentity_Mask], or combinations of:
  ///   [TypeMask.kTranslate_Mask], [TypeMask.kScale_Mask],
  ///   [TypeMask.kAffine_Mask], [TypeMask.kPerspective_Mask]
  TypeMask get type => TypeMask.fromValue(c.mnn_cv_matrix_get_type(ptr));

  /// Returns true if Matrix is identity.  Identity matrix is:
  ///
  /// | 1 0 0 |
  /// | 0 1 0 |
  /// | 0 0 1 |
  ///
  /// @return  true if Matrix has no effect
  bool get isIdentity => c.mnn_cv_matrix_is_identity(ptr);

  /// Returns true if Matrix at most scales and translates. Matrix may be identity,
  /// contain only scale elements, only translate elements, or both. Matrix form is:
  ///
  ///      | scale-x    0    translate-x |
  ///      |    0    scale-y translate-y |
  ///      |    0       0 1      |
  ///
  /// @return  true if Matrix is identity; or scales, translates, or both
  bool get isScaleTranslate => c.mnn_cv_matrix_is_scale_translate(ptr);

  /// Returns true if Matrix is identity, or translates. Matrix form is:
  ///
  ///    | 1 0 translate-x |
  ///    | 0 1 translate-y |
  ///    | 0 0      1      |
  ///
  /// @return  true if Matrix is identity, or translates
  bool get isTranslate => c.mnn_cv_matrix_is_translate(ptr);

  ///  Returns true Matrix maps Rect to another Rect. If true, Matrix is identity,
  ///  or scales, or rotates a multiple of 90 degrees, or mirrors on axes. In all
  ///  cases, Matrix may also have translation. Matrix form is either:
  ///
  ///     | scale-x    0    translate-x |
  ///     |    0    scale-y translate-y |
  ///     |    0       0 1      |
  ///
  ///      or
  ///
  ///     |    0     rotate-x translate-x |
  ///     | rotate-y    0     translate-y |
  ///     |    00  1      |
  ///
  ///  for non-zero values of scale-x, scale-y, rotate-x, and rotate-y.
  ///
  ///  Also called preservesAxisAlignment(); use the one that provides better inline
  ///  documentation.
  ///
  ///  @return  true if Matrix maps one Rect into another
  bool get rectStaysRect => c.mnn_cv_matrix_rect_stays_rect(ptr);

  double get(int index) => c.mnn_cv_matrix_get(ptr, index);

  /// Returns scale factor multiplied by x-axis input, contributing to x-axis output.
  /// With mapPoints(), scales Point along the x-axis.
  ///
  /// @return  horizontal scale factor
  double get scaleX => get(kMScaleX);

  /// Returns scale factor multiplied by y-axis input, contributing to y-axis output.
  /// With mapPoints(), scales Point along the y-axis.
  ///
  /// @return  vertical scale factor
  double get scaleY => get(kMScaleY);

  /// Returns scale factor multiplied by y-axis input, contributing to x-axis output.
  ///  With mapPoints(), skews Point along the x-axis.
  ///  Skewing both axes can rotate Point.
  ///
  /// @return  horizontal skew factor
  double get skewX => get(kMSkewX);

  /// Returns scale factor multiplied by x-axis input, contributing to y-axis output.
  /// With mapPoints(), skews Point along the y-axis.
  /// Skewing both axes can rotate Point.
  ///
  /// @return  vertical skew factor
  double get skewY => get(kMSkewY);

  /// Returns translation contributing to x-axis output.
  /// With mapPoints(), moves Point along the x-axis.
  ///
  /// @return  horizontal translation factor
  double get translateX => get(kMTransX);

  double get translateY => get(kMTransY);

  double get perspX => get(kMPersp0);

  double get perspY => get(kMPersp1);

  void set(int index, double value) => c.mnn_cv_matrix_set(ptr, index, value);

  set scaleX(double value) => set(kMScaleX, value);
  set scaleY(double value) => set(kMScaleY, value);
  set skewX(double value) => set(kMSkewX, value);
  set skewY(double value) => set(kMSkewY, value);
  set translateX(double value) => set(kMTransX, value);
  set translateY(double value) => set(kMTransY, value);
  set perspX(double value) => set(kMPersp0, value);
  set perspY(double value) => set(kMPersp1, value);

  /// Sets all values from parameters. Sets matrix to:
  ///
  ///     | scaleX  skewX transX |
  ///     |  skewY scaleY transY |
  ///     | persp0 persp1 persp2 |
  ///
  /// @param scaleX  horizontal scale factor to store
  /// @param skewX   horizontal skew factor to store
  /// @param transX  horizontal translation to store
  /// @param skewY   vertical skew factor to store
  /// @param scaleY  vertical scale factor to store
  /// @param transY  vertical translation to store
  /// @param persp0  input x-axis values perspective factor to store
  /// @param persp1  input y-axis values perspective factor to store
  /// @param persp2  perspective scale factor to store
  void setAll({
    double scaleX = 1,
    double scaleY = 1,
    double skewX = 0,
    double skewY = 0,
    double translateX = 0,
    double translateY = 0,
    double persp0 = 0,
    double persp1 = 0,
    double persp2 = 0,
  }) {
    c.mnn_cv_matrix_set_all(
      ptr,
      scaleX,
      skewX,
      translateX,
      skewY,
      scaleY,
      translateY,
      persp0,
      persp1,
      persp2,
    );
  }

  /// Copies nine scalar values contained by Matrix into buffer, in member value
  /// ascending order: kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY,
  /// kMPersp0, kMPersp1, kMPersp2.
  ///
  /// @param buffer  storage for nine scalar values
  List<double> get9() {
    final p = calloc<ffi.Float>(9);
    c.mnn_cv_matrix_get9(ptr, p);
    final result = List<double>.generate(9, (index) => p[index]);
    calloc.free(p);
    return result;
  }

  /// Sets Matrix to nine scalar values in buffer, in member value ascending order:
  /// kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1,
  /// kMPersp2.
  ///
  /// Sets matrix to:
  ///
  ///     | buffer[0] buffer[1] buffer[2] |
  ///     | buffer[3] buffer[4] buffer[5] |
  ///     | buffer[6] buffer[7] buffer[8] |
  ///
  /// In the future, set9 followed by get9 may not return the same values. Since Matrix
  /// maps non-homogeneous coordinates, scaling all nine values produces an equivalent
  /// transformation, possibly improving precision.
  ///
  /// @param buffer  nine scalar values
  void set9(List<double> values) {
    final p = calloc<ffi.Float>(9);
    p.asTypedList(9).setAll(0, values);
    c.mnn_cv_matrix_set9(ptr, p);
    calloc.free(p);
  }

  /// Sets Matrix to identity; which has no effect on mapped Point. Sets Matrix to:
  ///
  ///     | 1 0 0 |
  ///     | 0 1 0 |
  ///     | 0 0 1 |
  ///
  /// Also called setIdentity(); use the one that provides better inline
  /// documentation.
  void reset() => c.mnn_cv_matrix_reset(ptr);

  /// Sets Matrix to identity; which has no effect on mapped Point. Sets Matrix to:
  ///
  ///     | 1 0 0 |
  ///     | 0 1 0 |
  ///     | 0 0 1 |
  ///
  /// Also called reset(); use the one that provides better inline
  /// documentation.
  void setIdentity() => c.mnn_cv_matrix_set_identity(ptr);

  /// Sets Matrix to translate by (dx, dy).
  ///
  /// @param dx  horizontal translation
  /// @param dy  vertical translation
  void setTranslate(double dx, double dy) => c.mnn_cv_matrix_set_translate(ptr, dx, dy);

  /// Sets Matrix to scale by sx and sy, about a pivot point at (px, py).
  /// The pivot point is unchanged when mapped with Matrix.
  ///
  /// @param sx  horizontal scale factor
  /// @param sy  vertical scale factor
  /// @param px  pivot x
  /// @param py  pivot y
  void setScale(double sx, double sy, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_set_scale(ptr, sx, sy, px, py);

  /// Sets Matrix to rotate by degrees about a pivot point at (px, py).
  /// The pivot point is unchanged when mapped with Matrix.
  ///
  /// Positive degrees rotates clockwise.
  ///
  /// @param degrees  angle of axes relative to upright axes
  /// @param px       pivot x
  /// @param py       pivot y
  void setRotate(double degrees, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_set_rotate(ptr, degrees, px, py);

  /// Sets Matrix to rotate by sinValue and cosValue, about a pivot point at (px, py).
  /// The pivot point is unchanged when mapped with Matrix.
  ///
  /// Vector (sinValue, cosValue) describes the angle of rotation relative to (0, 1).
  /// Vector length specifies scale.
  ///
  /// @param sinValue  rotation vector x-axis component
  /// @param cosValue  rotation vector y-axis component
  /// @param pxpivot x-axis
  /// @param pypivot y-axis
  void setSinCos(double sin, double cos, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_set_sincos(ptr, sin, cos, px, py);

  /// Sets Matrix to skew by kx and ky, about a pivot point at (px, py).
  /// The pivot point is unchanged when mapped with Matrix.
  ///
  /// @param kx  horizontal skew factor
  /// @param ky  vertical skew factor
  /// @param px  pivot x
  /// @param py  pivot y
  void setSkew(double kx, double ky, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_set_skew(ptr, kx, ky, px, py);

  /// Sets Matrix to Matrix a multiplied by Matrix b. Either a or b may be this.
  ///
  /// Given:
  ///
  ///       | A B C |      | J K L |
  ///   a = | D E F |, b = | M N O |
  ///       | G H I |      | P Q R |
  ///
  /// sets Matrix to:
  ///
  ///         | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
  /// a * b = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
  ///         | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |
  ///
  /// @param a  Matrix on left side of multiply expression
  /// @param b  Matrix on right side of multiply expression
  void setConcat(Matrix a, Matrix b) => c.mnn_cv_matrix_set_concat(ptr, a.ptr, b.ptr);

  /// Sets Matrix to Matrix multiplied by Matrix constructed from translation (dx, dy).
  /// This can be thought of as moving the point to be mapped before applying Matrix.
  ///
  /// Given:
  ///
  ///              | A B C |               | 1 0 dx |
  ///     Matrix = | D E F |,  T(dx, dy) = | 0 1 dy |
  ///              | G H I |               | 0 0  1 |
  ///
  /// sets Matrix to:
  ///
  ///                          | A B C | | 1 0 dx |   | A B A*dx+B*dy+C |
  ///     Matrix * T(dx, dy) = | D E F | | 0 1 dy | = | D E D*dx+E*dy+F |
  ///                          | G H I | | 0 0  1 |   | G H G*dx+H*dy+I |
  ///
  /// @param dx  x-axis translation before applying Matrix
  /// @param dy  y-axis translation before applying Matrix
  void preTranslate(double dx, double dy) => c.mnn_cv_matrix_pre_translate(ptr, dx, dy);

  /// Sets Matrix to Matrix multiplied by Matrix constructed from scaling by (sx, sy)
  /// about pivot point (px, py).
  /// This can be thought of as scaling about a pivot point before applying Matrix.
  ///
  /// Given:
  ///
  ///              | A B C |                       | sx  0 dx |
  ///     Matrix = | D E F |,  S(sx, sy, px, py) = |  0 sy dy |
  ///              | G H I |                       |  0  0  1 |
  ///
  /// where
  ///
  ///     dx = px - sx * px
  ///     dy = py - sy * py
  ///
  /// sets Matrix to:
  ///
  ///                                  | A B C | | sx  0 dx |   | A*sx B*sy A*dx+B*dy+C |
  ///     Matrix * S(sx, sy, px, py) = | D E F | |  0 sy dy | = | D*sx E*sy D*dx+E*dy+F |
  ///                                  | G H I | |  0  0  1 |   | G*sx H*sy G*dx+H*dy+I |
  ///
  /// @param sx  horizontal scale factor
  /// @param sy  vertical scale factor
  /// @param px  pivot x
  /// @param py  pivot y
  void preScale(double sx, double sy, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_pre_scale(ptr, sx, sy, px, py);

  /// Sets Matrix to Matrix multiplied by Matrix constructed from rotating by degrees
  /// about pivot point (px, py).
  /// This can be thought of as rotating about a pivot point before applying Matrix.
  ///
  /// Positive degrees rotates clockwise.
  ///
  /// Given:
  ///
  ///              | A B C |                        | c -s dx |
  ///     Matrix = | D E F |,  R(degrees, px, py) = | s  c dy |
  ///              | G H I |                        | 0  0  1 |
  ///
  /// where
  ///
  ///     c  = cos(degrees)
  ///     s  = sin(degrees)
  ///     dx =  s * py + (1 - c) * px
  ///     dy = -s * px + (1 - c) * py
  ///
  /// sets Matrix to:
  ///
  ///                                   | A B C | | c -s dx |   | Ac+Bs -As+Bc A*dx+B*dy+C |
  ///     Matrix * R(degrees, px, py) = | D E F | | s  c dy | = | Dc+Es -Ds+Ec D*dx+E*dy+F |
  ///                                   | G H I | | 0  0  1 |   | Gc+Hs -Gs+Hc G*dx+H*dy+I |
  ///
  /// @param degrees  angle of axes relative to upright axes
  /// @param px       pivot x
  /// @param py       pivot y
  void preRotate(double degrees, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_pre_rotate(ptr, degrees, px, py);

  /// Sets Matrix to Matrix multiplied by Matrix constructed from skewing by (kx, ky)
  /// about pivot point (px, py).
  /// This can be thought of as skewing about a pivot point before applying Matrix.
  ///
  /// Given:
  ///
  ///              | A B C |                       |  1 kx dx |
  ///     Matrix = | D E F |,  K(kx, ky, px, py) = | ky  1 dy |
  ///              | G H I |                       |  0  0  1 |
  ///
  /// where
  ///
  ///     dx = -kx * py
  ///     dy = -ky * px
  ///
  /// sets Matrix to:
  ///
  ///                                  | A B C | |  1 kx dx |   | A+B*ky A*kx+B A*dx+B*dy+C |
  ///     Matrix * K(kx, ky, px, py) = | D E F | | ky  1 dy | = | D+E*ky D*kx+E D*dx+E*dy+F |
  ///                                  | G H I | |  0  0  1 |   | G+H*ky G*kx+H G*dx+H*dy+I |
  ///
  /// @param kx  horizontal skew factor
  /// @param ky  vertical skew factor
  /// @param px  pivot x
  /// @param py  pivot y
  void preSkew(double kx, double ky, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_pre_skew(ptr, kx, ky, px, py);

  /// Sets Matrix to Matrix multiplied by Matrix other.
  /// This can be thought of mapping by other before applying Matrix.
  ///
  /// Given:
  ///
  ///              | A B C |          | J K L |
  ///     Matrix = | D E F |, other = | M N O |
  ///              | G H I |          | P Q R |
  ///
  /// sets Matrix to:
  ///
  ///                      | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
  ///     Matrix * other = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
  ///                      | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |
  ///
  /// @param other  Matrix on right side of multiply expression
  void preConcat(Matrix other) => c.mnn_cv_matrix_pre_concat(ptr, other.ptr);

  /// Sets Matrix to Matrix constructed from translation (dx, dy) multiplied by Matrix.
  /// This can be thought of as moving the point to be mapped after applying Matrix.
  ///
  /// Given:
  ///
  ///              | J K L |               | 1 0 dx |
  ///     Matrix = | M N O |,  T(dx, dy) = | 0 1 dy |
  ///              | P Q R |               | 0 0  1 |
  ///
  /// sets Matrix to:
  ///
  ///                          | 1 0 dx | | J K L |   | J+dx*P K+dx*Q L+dx*R |
  ///     T(dx, dy) * Matrix = | 0 1 dy | | M N O | = | M+dy*P N+dy*Q O+dy*R |
  ///                          | 0 0  1 | | P Q R |   |      P      Q      R |
  ///
  /// @param dx  x-axis translation after applying Matrix
  /// @param dy  y-axis translation after applying Matrix
  void postTranslate(double dx, double dy) => c.mnn_cv_matrix_post_translate(ptr, dx, dy);

  /// Sets Matrix to Matrix constructed from scaling by (sx, sy) about pivot point
  /// (px, py), multiplied by Matrix.
  /// This can be thought of as scaling about a pivot point after applying Matrix.
  ///
  /// Given:
  ///
  ///              | J K L |                       | sx  0 dx |
  ///     Matrix = | M N O |,  S(sx, sy, px, py) = |  0 sy dy |
  ///              | P Q R |                       |  0  0  1 |
  ///
  /// where
  ///
  ///     dx = px - sx * px
  ///     dy = py - sy * py
  ///
  /// sets Matrix to:
  ///
  ///                                  | sx  0 dx | | J K L |   | sx*J+dx*P sx*K+dx*Q sx*L+dx+R |
  ///     S(sx, sy, px, py) * Matrix = |  0 sy dy | | M N O | = | sy*M+dy*P sy*N+dy*Q sy*O+dy*R |
  ///                                  |  0  0  1 | | P Q R |   |         P         Q         R |
  ///
  /// @param sx  horizontal scale factor
  /// @param sy  vertical scale factor
  /// @param px  pivot x
  /// @param py  pivot y
  void postScale(double sx, double sy, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_post_scale(ptr, sx, sy, px, py);

  /// Sets Matrix to Matrix constructed from scaling by (1/divx, 1/divy) about pivot point (px, py), multiplied by
  ///        Matrix.
  ///
  /// Returns false if either divx or divy is zero.
  ///
  /// Given:
  ///
  ///              | J K L |                   | sx  0  0 |
  ///     Matrix = | M N O |,  I(divx, divy) = |  0 sy  0 |
  ///              | P Q R |                   |  0  0  1 |
  ///
  /// where
  ///
  ///     sx = 1 / divx
  ///     sy = 1 / divy
  ///
  /// sets Matrix to:
  ///
  ///                              | sx  0  0 | | J K L |   | sx*J sx*K sx*L |
  ///     I(divx, divy) * Matrix = |  0 sy  0 | | M N O | = | sy*M sy*N sy*O |
  ///                              |  0  0  1 | | P Q R |   |    P    Q    R |
  ///
  /// @param divx  integer divisor for inverse scale in x
  /// @param divy  integer divisor for inverse scale in y
  /// @return      true on successful scale
  void postIDiv(int divx, int divy) => c.mnn_cv_matrix_post_idiv(ptr, divx, divy);

  /// Sets Matrix to Matrix constructed from rotating by degrees about pivot point
  /// (px, py), multiplied by Matrix.
  /// This can be thought of as rotating about a pivot point after applying Matrix.
  ///
  /// Positive degrees rotates clockwise.
  ///
  /// Given:
  ///
  ///              | J K L |                        | c -s dx |
  ///     Matrix = | M N O |,  R(degrees, px, py) = | s  c dy |
  ///              | P Q R |                        | 0  0  1 |
  ///
  /// where
  ///
  ///     c  = cos(degrees)
  ///     s  = sin(degrees)
  ///     dx =  s * py + (1 - c) * px
  ///     dy = -s * px + (1 - c) * py
  ///
  /// sets Matrix to:
  ///
  ///                                   |c -s dx| |J K L|   |cJ-sM+dx*P cK-sN+dx*Q cL-sO+dx+R|
  ///     R(degrees, px, py) * Matrix = |s  c dy| |M N O| = |sJ+cM+dy*P sK+cN+dy*Q sL+cO+dy*R|
  ///                                   |0  0  1| |P Q R|   |         P          Q          R|
  ///
  /// @param degrees  angle of axes relative to upright axes
  /// @param px       pivot x
  /// @param py       pivot y
  void postRotate(double degrees, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_post_rotate(ptr, degrees, px, py);

  /// Sets Matrix to Matrix constructed from skewing by (kx, ky) about pivot point
  /// (px, py), multiplied by Matrix.
  /// This can be thought of as skewing about a pivot point after applying Matrix.
  ///
  /// Given:
  ///
  ///              | J K L |                       |  1 kx dx |
  ///     Matrix = | M N O |,  K(kx, ky, px, py) = | ky  1 dy |
  ///              | P Q R |                       |  0  0  1 |
  ///
  /// where
  ///
  ///     dx = -kx * py
  ///     dy = -ky * px
  ///
  /// sets Matrix to:
  ///
  ///                                  | 1 kx dx| |J K L|   |J+kx*M+dx*P K+kx*N+dx*Q L+kx*O+dx+R|
  ///     K(kx, ky, px, py) * Matrix = |ky  1 dy| |M N O| = |ky*J+M+dy*P ky*K+N+dy*Q ky*L+O+dy*R|
  ///                                  | 0  0  1| |P Q R|   |          P           Q           R|
  ///
  /// @param kx  horizontal skew factor
  /// @param ky  vertical skew factor
  /// @param px  pivot x
  /// @param py  pivot y
  void postSkew(double kx, double ky, {double px = 0, double py = 0}) =>
      c.mnn_cv_matrix_post_skew(ptr, kx, ky, px, py);

  /// Sets Matrix to Matrix other multiplied by Matrix.
  /// This can be thought of mapping by other after applying Matrix.
  ///
  /// Given:
  ///
  ///              | J K L |           | A B C |
  ///     Matrix = | M N O |,  other = | D E F |
  ///              | P Q R |           | G H I |
  ///
  /// sets Matrix to:
  ///
  ///                      | A B C |   | J K L |   | AJ+BM+CP AK+BN+CQ AL+BO+CR |
  ///     other * Matrix = | D E F | * | M N O | = | DJ+EM+FP DK+EN+FQ DL+EO+FR |
  ///                      | G H I |   | P Q R |   | GJ+HM+IP GK+HN+IQ GL+HO+IR |
  ///
  /// @param other  Matrix on left side of multiply expression
  void postConcat(Matrix other) => c.mnn_cv_matrix_post_concat(ptr, other.ptr);

  /// Initializes Matrix with scale and translate elements.
  ///
  ///     | sx  0 tx |
  ///     |  0 sy ty |
  ///     |  0  0  1 |
  ///
  /// @param sx  horizontal scale factor to store
  /// @param sy  vertical scale factor to store
  /// @param tx  horizontal translation to store
  /// @param ty  vertical translation to store
  void setScaleTranslate(double sx, double sy, double dx, double dy) =>
      c.mnn_cv_matrix_set_scale_translate(ptr, sx, sy, dx, dy);

  /// Sets Matrix to scale and translate src Rect to dst Rect. stf selects whether
  /// mapping completely fills dst or preserves the aspect ratio, and how to align
  /// src within dst. Returns false if src is empty, and sets Matrix to identity.
  /// Returns true if dst is empty, and sets Matrix to:
  ///
  ///     | 0 0 0 |
  ///     | 0 0 0 |
  ///     | 0 0 1 |
  ///
  /// @param src  Rect to map from
  /// @param dst  Rect to map to
  /// @param stf  one of: kFill_ScaleToFit, kStart_ScaleToFit,
  ///             kCenter_ScaleToFit, kEnd_ScaleToFit
  /// @return     true if Matrix can represent Rect mapping
  bool setRectToRect(Rect src, Rect dst, ScaleToFit scaleToFit) =>
      c.mnn_cv_matrix_set_rect_to_rect(ptr, src.ref, dst.ref, scaleToFit.value);

  /// Sets Matrix to map src to dst. count must be zero or greater, and four or less.
  ///
  /// If count is zero, sets Matrix to identity and returns true.
  ///
  /// If count is one, sets Matrix to translate and returns true.
  ///
  /// If count is two or more, sets Matrix to map Point if possible; returns false
  ///
  /// if Matrix cannot be constructed. If count is four, Matrix may include
  /// perspective.
  ///
  /// @param src    Point to map from
  /// @param dst    Point to map to
  /// @param count  number of Point in src and dst
  /// @return       true if Matrix was constructed successfully
  bool setPolyToPoly(List<Point> src, List<Point> dst, {int? count}) {
    if (src.length != dst.length) {
      throw ArgumentError('src.length=${src.length} != dst.length=${dst.length}');
    }
    count ??= src.length;
    final srcPtr = calloc<c.mnn_cv_point_t>(count);
    final dstPtr = calloc<c.mnn_cv_point_t>(count);
    for (var i = 0; i < count; i++) {
      srcPtr[i] = src[i].ref;
      dstPtr[i] = dst[i].ref;
    }
    final result = c.mnn_cv_matrix_set_poly_to_poly(ptr, srcPtr, dstPtr, count);
    calloc.free(srcPtr);
    calloc.free(dstPtr);
    return result;
  }

  /// Maps src Point array of length count to dst Point array of equal or greater
  /// length. Point are mapped by multiplying each Point by Matrix. Given:
  ///
  ///              | A B C |        | x |
  ///     Matrix = | D E F |,  pt = | y |
  ///              | G H I |        | 1 |
  ///
  /// where
  ///
  ///     for (i = 0; i < count; ++i) {
  ///         x = src[i].fX
  ///         y = src[i].fY
  ///     }
  ///
  /// each dst Point is computed as:
  ///
  ///                   |A B C| |x|                               Ax+By+C   Dx+Ey+F
  ///     Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
  ///                   |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
  ///
  /// src and dst may point to the same storage.
  ///
  /// @param dst    storage for mapped Point
  /// @param src    Point to transform
  /// @param count  number of Point to transform
  List<Point> mapPoints(List<Point> src, {bool inplace = false}) {
    final count = src.length;
    final srcPtr = calloc<c.mnn_cv_point_t>(count);
    for (var i = 0; i < count; i++) {
      srcPtr[i] = src[i].ref;
    }
    if (inplace) {
      c.mnn_cv_matrix_map_points_inplace(ptr, srcPtr, count);
      for (var i = 0; i < count; i++) {
        src[i] = Point.fromNative(srcPtr[i]);
      }
      calloc.free(srcPtr);
      return src;
    }

    final dstPtr = calloc<c.mnn_cv_point_t>(count);
    c.mnn_cv_matrix_map_points(ptr, srcPtr, dstPtr, count);
    final result = List<Point>.generate(count, (index) => Point.fromNative(dstPtr[index]));
    calloc.free(srcPtr);
    calloc.free(dstPtr);
    return result;
  }

  /// Sets inverse to reciprocal matrix, returning true if Matrix can be inverted.
  /// Geometrically, if Matrix maps from source to destination, inverse Matrix
  /// maps from destination to source. If Matrix can not be inverted, inverse is
  /// unchanged.
  ///
  /// @param inverse  storage for inverted Matrix; may be nullptr
  /// @return         true if Matrix can be inverted
  (bool, Matrix) invert() {
    final result = Matrix.create();
    final success = c.mnn_cv_matrix_invert(ptr, result.ptr);
    return (success, result);
  }

  /// Returns Point (x, y) multiplied by Matrix. Given:
  ///
  ///              | A B C |        | x |
  ///     Matrix = | D E F |,  pt = | y |
  ///              | G H I |        | 1 |
  ///
  /// result is computed as:
  ///
  ///                   |A B C| |x|                               Ax+By+C   Dx+Ey+F
  ///     Matrix * pt = |D E F| |y| = |Ax+By+C Dx+Ey+F Gx+Hy+I| = ------- , -------
  ///                   |G H I| |1|                               Gx+Hy+I   Gx+Hy+I
  ///
  /// @param x  x-axis value of Point to map
  /// @param y  y-axis value of Point to map
  /// @return   mapped Point
  Point mapXY(double x, double y) {
    final px = calloc<ffi.Float>();
    final py = calloc<ffi.Float>();
    c.mnn_cv_matrix_map_xy(ptr, x, y, px, py);
    final result = Point.fromXY(px.value, py.value);
    calloc.free(px);
    calloc.free(py);
    return result;
  }

  /// Sets dst to bounds of src corners mapped by Matrix.
  /// Returns true if mapped corners are dst corners.
  ///
  /// Returned value is the same as calling rectStaysRect().
  ///
  /// @param dst  storage for bounds of mapped Point
  /// @param src  Rect to map
  /// @return     true if dst is equivalent to mapped src
  (bool, Rect) mapRect(Rect src) {
    final p = calloc<c.mnn_cv_rect_t>();
    final success = c.mnn_cv_matrix_map_rect(ptr, p, src.ptr.cast());
    return (success, Rect.fromPointer(p.cast()));
  }

  /// Sets dst to bounds of src corners mapped by Matrix. If matrix contains
  /// elements other than scale or translate: asserts if SK_DEBUG is defined;
  /// otherwise, results are undefined.
  ///
  /// @param dst  storage for bounds of mapped Point
  /// @param src  Rect to map
  Rect mapRectScaleTranslate(Rect src) {
    final p = calloc<c.mnn_cv_rect_t>();
    c.mnn_cv_matrix_map_rect_scale_translate(ptr, p, src.ptr.cast());
    return Rect.fromPointer(p.cast());
  }

  /// Returns true if Matrix equals m, using an efficient comparison.
  ///
  /// Returns false when the sign of zero values is the different; when one
  /// matrix has positive zero value and the other has negative zero value.
  ///
  /// Returns true even when both Matrix contain NaN.
  ///
  /// NaN never equals any value, including itself. To improve performance, NaN values
  /// are treated as bit patterns that are equal if their bit patterns are equal.
  ///
  /// @param m  Matrix to compare
  /// @return   true if m and Matrix are represented by identical bit patterns
  bool cheapEqualTo(Matrix other) => c.mnn_cv_matrix_cheap_equal_to(ptr, other.ptr);

  /// Sets internal cache to unknown state. Use to force update after repeated
  /// modifications to Matrix element reference returned by operator[](int index).
  void dirtyMatrixTypeCache() => c.mnn_cv_matrix_dirty_matrix_type_cache(ptr);

  // Matrix organizes its values in row order.
  // These members correspond to each value in Matrix.

  /// horizontal scale factor
  static const int kMScaleX = 0;

  /// horizontal skew factor
  static const int kMSkewX = 1;

  /// horizontal translation
  static const int kMTransX = 2;

  /// vertical skew factor
  static const int kMSkewY = 3;

  /// vertical scale factor
  static const int kMScaleY = 4;

  /// vertical translation
  static const int kMTransY = 5;

  /// input x perspective factor
  static const int kMPersp0 = 6;

  /// input y perspective factor
  static const int kMPersp1 = 7;

  /// perspective bias
  /// Affine arrays are in column major order to match the matrix used by
  /// PDF and XPS.
  static const int kMPersp2 = 8;

  /// horizontal scale factor
  static const int kAScaleX = 0;

  /// vertical skew factor
  static const int kASkewY = 1;

  /// horizontal skew factor
  static const int kASkewX = 2;

  /// vertical scale factor
  static const int kAScaleY = 3;

  /// horizontal translation
  static const int kATransX = 4;

  /// vertical translation
  static const int kATransY = 5;

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => get9();

  @override
  String toString() {
    // kMScaleX, kMSkewX, kMTransX, kMSkewY, kMScaleY, kMTransY, kMPersp0, kMPersp1, kMPersp2
    final vals = get9();
    return '[${vals[0].toStringAsFixed(2)}, ${vals[1].toStringAsFixed(2)}, ${vals[2].toStringAsFixed(2)},\n'
        ' ${vals[3].toStringAsFixed(2)}, ${vals[4].toStringAsFixed(2)}, ${vals[5].toStringAsFixed(2)},\n'
        ' ${vals[6].toStringAsFixed(2)}, ${vals[7].toStringAsFixed(2)}, ${vals[8].toStringAsFixed(2)}]';
  }

  @override
  void release() {
    c.mnn_cv_matrix_destroy(ptr);
  }
}

/// enum Matrix::TypeMask
/// Enum of bit fields for mask returned by getType().
/// Used to identify the complexity of Matrix, to optimize performance.
enum TypeMask {
  /// identity Matrix; all bits clear
  kIdentity_Mask(0),

  /// translation Matrix
  kTranslate_Mask(0x01),

  /// scale Matrix
  kScale_Mask(0x02),

  /// skew or rotate Matrix
  kAffine_Mask(0x04),

  /// perspective Matrix
  kPerspective_Mask(0x08);

  final int value;
  const TypeMask(this.value);
  static TypeMask fromValue(int value) => switch (value) {
        0 => kIdentity_Mask,
        1 => kTranslate_Mask,
        2 => kScale_Mask,
        4 => kAffine_Mask,
        8 => kPerspective_Mask,
        _ => throw MNNException('Unknown TypeMask: $value'),
      };
}

/// enum Matrix::ScaleToFit
/// ScaleToFit describes how Matrix is constructed to map one Rect to another.
///
/// ScaleToFit may allow Matrix to have unequal horizontal and vertical scaling,
/// or may restrict Matrix to square scaling. If restricted, ScaleToFit specifies
/// how Matrix maps to the side or center of the destination Rect.
enum ScaleToFit {
  /// scales in x and y to fill destination Rect
  kFill_ScaleToFit(0),

  /// scales and aligns to left and top
  kStart_ScaleToFit(1),

  /// scales and aligns to center
  kCenter_ScaleToFit(2),

  /// scales and aligns to right and bottom
  kEnd_ScaleToFit(3);

  final int value;
  const ScaleToFit(this.value);
  static ScaleToFit fromValue(int value) => switch (value) {
        0 => kFill_ScaleToFit,
        1 => kStart_ScaleToFit,
        2 => kCenter_ScaleToFit,
        3 => kEnd_ScaleToFit,
        _ => throw MNNException('Unknown ScaleToFit: $value'),
      };
}
