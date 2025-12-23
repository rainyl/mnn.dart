cmake_minimum_required(VERSION 3.11)
project(mnn_c_api VERSION 1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MNNC_BUILD_TEST "Build test" OFF)

set(MNN_VERSION 3.3.0)

cmake_policy(SET CMP0077 NEW)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif ()

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
    if(MNN_OPENCL)
      set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS} MNN_CL)
    endif(MNN_OPENCL)
    if (MNN_OPENGL)
      set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS} MNN_GL)
    endif(MNN_OPENGL)
    if (MNN_VULKAN)
      set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS} MNN_Vulkan)
    endif(MNN_VULKAN)

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
