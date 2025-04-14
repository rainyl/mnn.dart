cmake_minimum_required(VERSION 3.11)
project(mnn_c_api VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MNNC_BUILD_TEST "Build test" OFF)

set(MNN_VERSION 3.1.2)

cmake_policy(SET CMP0077 NEW)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif ()

set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "Build tools/cpp or not" FORCE)
# set(MNN_SUPPORT_BF16 ON CACHE BOOL "Enable MNN's bf16 op" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "Build MNN Backends and expression separately. Only works with MNN_BUILD_SHARED_LIBS=ON" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "MNN build shared or static lib" FORCE)
# set(MNN_BUILD_OPENCV ON CACHE BOOL "Build OpenCV api in MNN." FORCE)
# set(MNN_USE_SYSTEM_LIB OFF CACHE BOOL "For opencl and vulkan, use system lib or use dlopen" FORCE)
# set(MNN_BUILD_HARD OFF CACHE BOOL "Build -mfloat-abi=hard or not" FORCE)
set(MNN_WIN_RUNTIME_MT OFF CACHE BOOL "MNN use /MT on Windows dll" FORCE)
# set(MNN_FORBID_MULTI_THREAD OFF CACHE BOOL "Disable Multi Thread" FORCE)
# set(MNN_OPENMP OFF CACHE BOOL "Use OpenMP's thread pool implementation. Does not work on iOS or Mac OS" FORCE)
# set(MNN_USE_THREAD_POOL ON CACHE BOOL "Use MNN's own thread pool implementation" FORCE)
set(MNN_BUILD_TRAIN OFF CACHE BOOL "Build MNN's training framework" FORCE)
set(MNN_BUILD_DEMO OFF CACHE BOOL "Build demo/exec or not" FORCE)
set(MNN_BUILD_QUANTOOLS OFF CACHE BOOL "Build Quantized Tools or not" FORCE)
set(MNN_EVALUATION OFF CACHE BOOL "Build Evaluation Tools or not" FORCE)
set(MNN_BUILD_CONVERTER OFF CACHE BOOL "Build Converter" FORCE)
set(MNN_SUPPORT_DEPRECATED_OP OFF CACHE BOOL "Enable MNN's tflite quantized op" FORCE)
# set(MNN_DEBUG_MEMORY OFF CACHE BOOL "MNN Debug Memory Access" FORCE)
# set(MNN_DEBUG_TENSOR_SIZE OFF CACHE BOOL "Enable Tensor Size" FORCE)
# set(MNN_GPU_TRACE OFF CACHE BOOL "Enable MNN Gpu Debug" FORCE)
# set(MNN_SUPPORT_RENDER OFF CACHE BOOL "Enable MNN Render Ops" FORCE)
# set(MNN_SUPPORT_TRANSFORMER_FUSE OFF CACHE BOOL "Enable MNN transformer Fuse Ops" FORCE)
set(NATIVE_LIBRARY_OUTPUT OFF CACHE BOOL "Native Library Path" FORCE)
set(NATIVE_INCLUDE_OUTPUT OFF CACHE BOOL "Native Include Path" FORCE)
set(MNN_AAPL_FMWK OFF CACHE BOOL "Build MNN.framework instead of traditional .a/.dylib" FORCE)
# set(MNN_WITH_PLUGIN OFF CACHE BOOL "Build with plugin op support." FORCE)
# set(MNN_BUILD_MINI OFF CACHE BOOL "Build MNN-MINI that just supports fixed shape models." FORCE)
# set(MNN_USE_SSE ON CACHE BOOL "Use SSE optimization for x86 if possiable" FORCE)
set(MNN_BUILD_CODEGEN OFF CACHE BOOL "Build with codegen" FORCE)
set(MNN_ENABLE_COVERAGE OFF CACHE BOOL "Build with coverage enable" FORCE)
# set(MNN_BUILD_PROTOBUFFER ON CACHE BOOL "Build with protobuffer in MNN" FORCE)
# set(MNN_BUILD_LLM OFF CACHE BOOL "Build llm library based MNN." FORCE)
# set(MNN_BUILD_DIFFUSION OFF CACHE BOOL "Build diffusion demo based MNN." FORCE)
# set(MNN_INTERNAL OFF CACHE BOOL "Build with MNN internal features, such as model authentication, metrics logging" FORCE)
set(MNN_JNI OFF CACHE BOOL "Build MNN Jni for java to use" FORCE)
# set(MNN_LOW_MEMORY OFF CACHE BOOL "Build MNN support low memory for weight quant model." FORCE)
# set(MNN_CPU_WEIGHT_DEQUANT_GEMM OFF CACHE BOOL "Build MNN CPU weight dequant related gemm kernels." FORCE)
# set(MNN_BUILD_AUDIO OFF CACHE BOOL "Build audio api in MNN." FORCE)

if(WIN32)
  set(MNN_WIN_RUNTIME_MT OFF CACHE BOOL "" FORCE)
endif(WIN32)


include(FetchContent)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.28.0")
  FetchContent_Declare(
    MNN
    # GIT_REPOSITORY "https://github.com/alibaba/MNN.git"
    # GIT_TAG ${MNN_VERSION}
    URL https://github.com/alibaba/MNN/archive/refs/tags/${MNN_VERSION}.tar.gz
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(MNN)
else()
  FetchContent_Declare(
    MNN
    # GIT_REPOSITORY "https://github.com/alibaba/MNN.git"
    # GIT_TAG ${MNN_VERSION}
    URL https://github.com/alibaba/MNN/archive/refs/tags/${MNN_VERSION}.tar.gz
  )
  FetchContent_GetProperties(MNN)
  if(NOT MNN_POPULATED)
    FetchContent_Populate(MNN)
    add_subdirectory(${MNN_SOURCE_DIR} ${MNN_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

set(MNNC_LINK_LIBS MNN)
if (ANDROID)
    set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS} mediandk -landroid)
elseif (APPLE AND IOS)
    # https://github.com/opencv/opencv/blob/450e741f8d53ff12b4e194c7762adaefb952555a/platforms/ios/build_framework.py#L386
    set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS}
            "-framework UIKit"
            "-framework AVFoundation"
            "-framework CoreVideo"
            "-framework CoreGraphics"
            "-framework CoreMedia"
            "-framework CoreImage"
            "-framework QuartzCore"
            "-framework Accelerate"
            "-framework Metal"
            "-framework Foundation"
    )
elseif (APPLE AND NOT IOS)
    set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS}
            "-framework Cocoa"
            "-framework AVFoundation"
            "-framework OpenGL"
            "-framework CoreVideo"
            "-framework CoreGraphics"
            "-framework CoreMedia"
            "-framework CoreImage"
            "-framework QuartzCore"
            "-framework Accelerate"
            "-framework Metal"
            "-framework Foundation"
    )
endif ()

set(SOURCES
    autotime.cpp
    interpreter.cpp
    tensor.cpp
    image_process.cpp
    mnn_stb_image.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MNN
)

add_library(mnn_c_api SHARED ${SOURCES})

target_link_libraries(mnn_c_api PUBLIC ${MNNC_LINK_LIBS})

if(MNNC_BUILD_TEST)
  add_executable(test_a test/test_interpreter.cpp)
  target_link_libraries(test_a mnn_c_api)
endif(MNNC_BUILD_TEST)

install(TARGETS mnn_c_api DESTINATION lib)
