//
// get_session_input_all_example.cpp
// Example of using mnn_interpreter_get_session_input_all
//

#include "../image_process.h"
#include "../interpreter.h"
#include "MNN/ImageProcess.hpp"
#include "MNN/Matrix.h"
#include "MNN/expr/Expr.hpp"
#include "expr.h"
#include <iostream>
#include <memory>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <model_path>\n", argv[0]);
    return 1;
  }

  // auto varp = MNN::Express::Variable::load(argv[1]);
  auto varp = mnn_expr_VARP_static_load(argv[1]);
  mnn_expr_VARP_static_save(varp, "model.bin");
  auto _info = mnn_expr_VARP_getInfo(&varp->at(0));

  if (_info == nullptr) {
    std::cout << "getInfo.info is nullptr" << std::endl;
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

  auto buf = new float[9];
  auto input = interpreter->getSessionInput(session, nullptr);
  MNN::CV::ImageProcess::Config conf;
  std::shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(conf, input));
  MNN::CV::Matrix trans;
  trans.postScale(2.0, 1.5);
  trans.invert(&trans);
  trans.get9(buf);
  std::cout << std::endl;
  std::cout << "matrix1: ";
  for (int i = 0; i < 9; i++) { std::cout << ", " << buf[i]; }
  std::cout << std::endl;

  pretreat->setMatrix(trans);
  pretreat->matrix().get9(buf);
  std::cout << std::endl;
  std::cout << "matrix2: ";
  for (int i = 0; i < 9; i++) { std::cout << ", " << buf[i]; }
  std::cout << std::endl;

  // 获取所有输入张量
  mnn_tensor_t *tensors = nullptr;
  const char **names = nullptr;
  size_t count = 0;

  mnn_error_code_t ret =
      mnn_interpreter_get_session_input_all(interpreter, session, &tensors, &names, &count);

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
  free(tensors);       // 释放张量数组
  free((void *)names); // 释放名称数组

  // 释放session和interpreter
  mnn_interpreter_release_session(interpreter, session, nullptr);
  mnn_interpreter_destroy(interpreter);

  return 0;
}
