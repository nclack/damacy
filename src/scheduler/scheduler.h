// Single-worker scheduler. Owns a platform_thread + mutex + cond and
// drives a caller-supplied step on a fixed cadence under the lock.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct scheduler;
  struct numa_resolved;

  // Worker tick callback. Invoked under scheduler_lock. Return non-zero
  // to have the scheduler broadcast its cond.
  typedef int (*scheduler_step_fn)(void* arg);

  // Spawn the worker. idle_ns must be > 0. `affinity` is the resolved
  // NUMA placement plan from numa_init; pass NULL (or a struct with
  // node<0) to skip affinity. Returns NULL on failure.
  struct scheduler* scheduler_create(scheduler_step_fn step,
                                     void* arg,
                                     int64_t idle_ns,
                                     const struct numa_resolved* affinity);

  // Signal shutdown, join, free. NULL-safe.
  void scheduler_destroy(struct scheduler* s);

  void scheduler_lock(struct scheduler* s);
  void scheduler_unlock(struct scheduler* s);

  // Caller must hold the lock. Released for the duration; reacquired
  // before return. Spurious wakeups possible.
  void scheduler_wait(struct scheduler* s);

  // Timed variant. Logs a warning with `site` and the elapsed time on
  // timeout, then re-waits. Use a macro at call sites to capture
  // __FILE__:__LINE__ as `site`.
  void scheduler_wait_diag(struct scheduler* s,
                           const char* site,
                           int timeout_ms);
#define SCHEDULER_WAIT_DIAG(s, ms)                                             \
  scheduler_wait_diag((s), __FILE__ ":" SCHED_STR(__LINE__), (ms))
#define SCHED_STR(x) SCHED_STR2(x)
#define SCHED_STR2(x) #x

  // Caller must hold the lock.
  void scheduler_broadcast(struct scheduler* s);

#ifdef __cplusplus
}
#endif
