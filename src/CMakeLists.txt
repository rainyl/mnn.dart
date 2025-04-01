cmake_minimum_required(VERSION 3.0)
project(mnn_c_api)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加源文件
set(SOURCES
    autotime.cpp
    interpreter.cpp
    tensor.cpp
)

# 添加头文件路径
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/MNN
)

# 创建共享库
add_library(mnn_c_api SHARED ${SOURCES})

# 链接MNN库
set(MNN_LIB /Users/rainy/DEV/mnn.dart/MNN.framework/MNN)
find_library(COCOA_LIB Cocoa)
find_library(FOUNDATION_LIB Foundation)
find_library(METAL_LIB Metal)
target_link_libraries(mnn_c_api PRIVATE ${MNN_LIB} ${COCOA_LIB} ${FOUNDATION_LIB} ${METAL_LIB})
