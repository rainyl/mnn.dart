cmake_minimum_required(VERSION 3.11)
project(mnn_c_api VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MNNC_BUILD_TEST "Build test" OFF)

set(MNN_VERSION 3.1.1)

cmake_policy(SET CMP0077 NEW)
set(MNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(MNN_SUPPORT_BF16 ON CACHE BOOL "" FORCE)
set(MNN_SEP_BUILD OFF CACHE BOOL "" FORCE)
set(MNN_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

if(WIN32)
  set(MNN_WIN_RUNTIME_MT OFF CACHE BOOL "" FORCE)
endif(WIN32)


include(FetchContent)
FetchContent_Declare(
  MNN
  GIT_REPOSITORY "https://github.com/alibaba/MNN.git"
  GIT_TAG ${MNN_VERSION}
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(MNN)

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
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MNN
)

add_library(mnn_c_api SHARED ${SOURCES})

target_link_libraries(mnn_c_api PUBLIC ${MNNC_LINK_LIBS})

add_library(mnn_stb_image SHARED mnn_stb_image.cpp stb_image.h stb_image_resize2.h stb_image_write.h)

if(MNNC_BUILD_TEST)
  add_executable(test_a test/test_interpreter.cpp)
  target_link_libraries(test_a mnn_c_api)
endif(MNNC_BUILD_TEST)

install(TARGETS mnn_c_api mnn_stb_image DESTINATION lib)
