#include "autotime.h"
#include "MNN/AutoTime.hpp"

extern "C" {

mnn_timer_t mnn_timer_create() { return new MNN::Timer(); }

void mnn_timer_destroy(mnn_timer_t timer) { delete static_cast<MNN::Timer *>(timer); }

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
  delete static_cast<MNN::AutoTime *>(auto_time);
}

} // extern "C"
