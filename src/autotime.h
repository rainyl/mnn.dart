//
// autotime.h
// MNN C API for AutoTime
//
// This file provides C-style API for MNN's AutoTime functionality
// All functions use snake_case naming convention
//

#ifndef MNN_AUTOTIME_H
#define MNN_AUTOTIME_H

#include <stdint.h>

#ifdef __cplusplus
#include "MNN/AutoTime.hpp"
extern "C" {
#endif

/**
 * @brief Opaque pointer type representing a timer object
 */
#ifdef __cplusplus
typedef MNN::Timer *mnn_timer_t;
typedef MNN::AutoTime *mnn_auto_time_t;
#else
typedef void *mnn_timer_t;
typedef void *mnn_auto_time_t;
#endif

/**
 * @brief Creates a new timer instance
 * @return Pointer to the newly created timer
 */
mnn_timer_t mnn_timer_create();

/**
 * @brief Destroys a timer instance
 * @param timer Timer instance to destroy
 */
void mnn_timer_destroy(mnn_timer_t timer);

/**
 * @brief Resets the timer to current time
 * @param timer Timer instance to reset
 */
void mnn_timer_reset(mnn_timer_t timer);

/**
 * @brief Gets the duration in microseconds since last reset
 * @param timer Timer instance to query
 * @return Duration in microseconds
 */
uint64_t mnn_timer_duration_us(mnn_timer_t timer);

/**
 * @brief Gets the current time value from timer
 * @param timer Timer instance to query
 * @return Current time value
 */
uint64_t mnn_timer_current(mnn_timer_t timer);

/**
 * @brief Creates a new auto timer instance
 * @param line Source code line number (for debugging)
 * @param func Function name (for debugging)
 * @return Pointer to the newly created auto timer
 */
mnn_auto_time_t mnn_auto_time_create(int line, const char *func);

/**
 * @brief Destroys an auto timer instance
 * @param auto_time Auto timer instance to destroy
 */
void mnn_auto_time_destroy(mnn_auto_time_t auto_time);

#ifdef __cplusplus
}
#endif

#endif // MNN_AUTOTIME_H
