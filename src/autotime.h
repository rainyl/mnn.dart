/*
 * autotime.h
 * MNN C API for AutoTime
 *
 * This file provides C-style API for MNN's AutoTime functionality
 *
 * Author: Rainyl
 * License: Apache License 2.0
 */

#ifndef MNN_AUTOTIME_H
#define MNN_AUTOTIME_H

#include "mnn_type.h"
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
MNN_C_API mnn_timer_t mnn_timer_create();

/**
 * @brief Destroys a timer instance
 * @param timer Timer instance to destroy
 */
MNN_C_API void mnn_timer_destroy(mnn_timer_t timer);

/**
 * @brief Resets the timer to current time
 * @param timer Timer instance to reset
 */
MNN_C_API void mnn_timer_reset(mnn_timer_t timer);

/**
 * @brief Gets the duration in microseconds since last reset
 * @param timer Timer instance to query
 * @return Duration in microseconds
 */
MNN_C_API uint64_t mnn_timer_duration_us(mnn_timer_t timer);

/**
 * @brief Gets the current time value from timer
 * @param timer Timer instance to query
 * @return Current time value
 */
MNN_C_API uint64_t mnn_timer_current(mnn_timer_t timer);

/**
 * @brief Creates a new auto timer instance
 * @param line Source code line number (for debugging)
 * @param func Function name (for debugging)
 * @return Pointer to the newly created auto timer
 */
MNN_C_API mnn_auto_time_t mnn_auto_time_create(int line, const char *func);

/**
 * @brief Destroys an auto timer instance
 * @param auto_time Auto timer instance to destroy
 */
MNN_C_API void mnn_auto_time_destroy(mnn_auto_time_t auto_time);

#ifdef __cplusplus
}
#endif

#endif // MNN_AUTOTIME_H
