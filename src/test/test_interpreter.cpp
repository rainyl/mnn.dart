//
// get_session_input_all_example.cpp
// Example of using mnn_interpreter_get_session_input_all
//

#include "../interpreter.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    // 创建interpreter实例
    mnn_interpreter_t interpreter = mnn_interpreter_create_from_file(argv[1], nullptr);
    if (!interpreter) {
        printf("Failed to create interpreter\n");
        return 1;
    }

    // 创建session配置
    mnn_schedule_config_t config;
    config.type = MNN_FORWARD_CPU;
    config.num_thread = 4;
    config.backend_config = nullptr;

    // 创建session
    mnn_session_t session = mnn_interpreter_create_session(interpreter, &config, nullptr);
    if (!session) {
        printf("Failed to create session\n");
        mnn_interpreter_destroy(interpreter);
        return 1;
    }

    // 获取所有输入张量
    mnn_tensor_t* tensors = nullptr;
    const char** names = nullptr;
    size_t count = 0;

    mnn_error_code_t ret = mnn_interpreter_get_session_input_all(
        interpreter,
        session,
        &tensors,
        &names,
        &count
    );

    if (ret != NO_ERROR) {
        printf("Failed to get session inputs: %d\n", ret);
        mnn_interpreter_release_session(interpreter, session, nullptr);
        mnn_interpreter_destroy(interpreter);
        return 1;
    }

    // 打印所有输入张量信息
    printf("Found %zu input tensors:\n", count);
    for (size_t i = 0; i < count; i++) {
        printf("  %s:\n", names[i]);
        tensors[i]->printShape();
    }

    // 清理资源
    free(tensors);  // 释放张量数组
    free((void*)names);  // 释放名称数组

    // 释放session和interpreter
    mnn_interpreter_release_session(interpreter, session, nullptr);
    mnn_interpreter_destroy(interpreter);

    return 0;
}