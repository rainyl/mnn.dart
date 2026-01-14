/*
 * autotime.cpp
 * MNN C API for AutoTime
 *
 * This file provides C-style API for MNN's AutoTime functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#include "mnn_c/autotime.h"
#include "MNN/AutoTime.hpp"

extern "C" {

mnn_timer_t mnn_timer_create() { return new MNN::Timer(); }

void mnn_timer_destroy(mnn_timer_t timer) {
  if (timer == nullptr) return;
  delete static_cast<MNN::Timer *>(timer);
  timer = nullptr;
}

void mnn_timer_reset(mnn_timer_t timer) { static_cast<MNN::Timer *>(timer)->reset(); }

uint64_t mnn_timer_duration_us(mnn_timer_t timer) {
  return static_cast<MNN::Timer *>(timer)->durationInUs();
}

uint64_t mnn_timer_current(mnn_timer_t timer) {
  return static_cast<MNN::Timer *>(timer)->current();
}

mnn_auto_time_t mnn_auto_time_create(int line, const char *func) {
  return new MNN::AutoTime(line, func);
}

void mnn_auto_time_destroy(mnn_auto_time_t auto_time) {
  if (auto_time == nullptr) return;
  delete static_cast<MNN::AutoTime *>(auto_time);
  auto_time = nullptr;
}

} // extern "C"
