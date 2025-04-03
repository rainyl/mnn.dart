import 'dart:ffi' as ffi;

import 'base.dart';
import 'g/mnn.g.dart' as c;

class Timer extends NativeObject {
  static final _finalizer = ffi.NativeFinalizer(c.addresses.mnn_timer_destroy);

  Timer.fromPointer(super.ptr, {super.attach, super.externalSize});

  /// Creates a new timer instance
  factory Timer.create() {
    return Timer.fromPointer(c.mnn_timer_create());
  }

  /// Resets the timer to current time
  void reset() => c.mnn_timer_reset(ptr);

  /// Gets the duration in microseconds since last reset
  int durationInUs() => c.mnn_timer_duration_us(ptr);

  /// Gets the current timestamp value from the timer
  int current() => c.mnn_timer_current(ptr);

  @override
  void release() {
    c.mnn_timer_destroy(ptr);
  }

  @override
  ffi.NativeFinalizer get finalizer => _finalizer;

  @override
  List<Object?> get props => [ptr.address];

  @override
  String toString() {
    return 'Timer(address=0x${ptr.address})';
  }
}
