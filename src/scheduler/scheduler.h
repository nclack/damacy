// Single-worker scheduler. Owns a platform_thread + mutex + cond and
// drives a caller-supplied step on a fixed cadence under the lock.
#pragma once

#include <stdatomic.h>
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

  // Timed variant. Caller must hold the lock (same as scheduler_wait).
  // On timeout, the supplied counter is incremented and logged on the
  // 1st, then every Nth occurrence (see DIAG_LOG_EVERY in scheduler.c)
  // — so a real hang is visible without flooding. Counter is reset to
  // zero on a successful (non-timeout) wake. Use the SCHEDULER_WAIT_DIAG
  // macro: it allocates a per-call-site static counter, which is safe
  // to share across threads only because the lock serializes access.
  void scheduler_wait_diag(struct scheduler* s,
                           const char* site,
                           int timeout_ms,
                           _Atomic int* count);
#define DAMACY_SCHED_STRINGIFY_INNER(x) #x
#define DAMACY_SCHED_STRINGIFY(x) DAMACY_SCHED_STRINGIFY_INNER(x)
// Counter is _Atomic so an accidental call from outside scheduler_lock
// is still well-defined (just non-throttled), instead of undefined.
#define SCHEDULER_WAIT_DIAG(s, ms)                                             \
  do {                                                                         \
    static _Atomic int _damacy_diag_count = 0;                                 \
    scheduler_wait_diag((s),                                                   \
                        __FILE__ ":" DAMACY_SCHED_STRINGIFY(__LINE__),         \
                        (ms),                                                  \
                        &_damacy_diag_count);                                  \
  } while (0)

  // Caller must hold the lock.
  void scheduler_broadcast(struct scheduler* s);

#ifdef __cplusplus
}
#endif
