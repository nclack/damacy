// Single-worker scheduler. Owns a platform_thread + mutex + cond and
// drives a caller-supplied step on a fixed cadence under the lock.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct scheduler;

  // Worker tick callback. Invoked under scheduler_lock. Return non-zero
  // to have the scheduler broadcast its cond.
  typedef int (*scheduler_step_fn)(void* arg);

  // Spawn the worker. idle_ns must be > 0. Returns NULL on failure.
  struct scheduler* scheduler_create(scheduler_step_fn step,
                                     void* arg,
                                     int64_t idle_ns);

  // Signal shutdown, join, free. NULL-safe.
  void scheduler_destroy(struct scheduler* s);

  void scheduler_lock(struct scheduler* s);
  void scheduler_unlock(struct scheduler* s);

  // Caller must hold the lock. Released for the duration; reacquired
  // before return. Spurious wakeups possible.
  void scheduler_wait(struct scheduler* s);

  // Caller must hold the lock.
  void scheduler_broadcast(struct scheduler* s);

#ifdef __cplusplus
}
#endif
