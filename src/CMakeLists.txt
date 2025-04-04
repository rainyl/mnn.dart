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
if(APPLE)
  set(MNNC_LINK_LIBS ${MNNC_LINK_LIBS} "-framework Foundation" "-framework Metal" "-framework Cocoa")
endif(APPLE)

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

if(MNNC_BUILD_TEST)
  add_executable(test_a test/test_interpreter.cpp)
  target_link_libraries(test_a mnn_c_api)
endif(MNNC_BUILD_TEST)

install(TARGETS mnn_c_api DESTINATION lib)
