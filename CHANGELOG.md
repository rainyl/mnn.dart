## 0.1.1

- feat(cv): add YUV NV21 image conversion and drawing functions
- Implement mnn_cv_buildImgVarpYuvNV21 to build VARP from YUV NV21
- Add drawing functions including arrowedLine, circle, ellipse, line, rectangle
- Add contour and polygon drawing functions (drawContours, fillPoly)
- Implement image transformation functions (flip, rotate)
- Add nms operation for non-maximum suppression
- Add runtime manager info getters for memory and FLOPs

## 0.1.0

- Note: breaking change
- bump MNN to 3.3.0
- bump dart sdk to 3.10
- new: add support of `MNN::Expr` module
- new: add numpy-like interfaces, e.g., `np.array()`, `np.argmax()`
- breaking change: move `mnn.f32`, `mnn.i32` etc. to `mnn.float32`, `mnn.int32` etc.
- new: add more functions to `VARP`
- new: support numpy-style slicing for `VARP.operator []`, e.g., `x[0]` `x["0, 1, :"]`
- new: support numpy-style slicing assignment for `VARP.operator []=`, e.g., `x[0]=...` `x["0, 1, :"]=...`
- breaking change: rename `VARP.list`, `VARP.list2D`, `VARP.list3D`, `VARP.list4D`, `VARP.listND` to `VARP.fromList1D`, `VARP.fromList2D`, `VARP.fromList3D`, `VARP.fromList4D`, `VARP.fromListND`
- breaking change: rename `Image.load`, `Image.fromMemory` -> `Image.file`, `Image.fromBytes`
- new: add `package:mnn/expr.dart`
- new: add `MNN::CV` module

## 0.0.3

- new: support custom definations
- new: add image process
- new: async run session
- sync: MNN 3.2.1

## 0.0.2

- fix: building on android, macos, windows, ios
- new: add flutter example

## 0.0.1

- Initial version.
