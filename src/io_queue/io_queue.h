// SPMC job queue with monotonic-sequence completion events.
//
// One producer feeds N worker-thread consumers. record() captures the
// current seq; io_event_wait() blocks until every job posted before
// that record() has finished, regardless of completion order.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct io_queue;
  struct numa_resolved;

  struct io_event
  {
    uint64_t seq;
  };

  // nthreads in [1, DAMACY_MAX_IO_THREADS]. `affinity` is the resolved
  // NUMA placement plan from numa_init; pass NULL (or a struct with
  // node<0) to skip affinity. Returns NULL on failure.
  struct io_queue* io_queue_create(int nthreads,
                                   const struct numa_resolved* affinity);

  // Drains queued jobs, then joins workers. Caller must ensure no
  // io_queue_post() is concurrent with or follows this call; racing posts
  // may leak ctx_free or use-after-free. Hangs if any in-flight fn()
  // blocks; the caller is responsible for unblocking them.
  void io_queue_destroy(struct io_queue* q);

  // Returns 0 on success. On failure ctx is not consumed; caller frees.
  // ctx_free, if non-NULL, runs after fn completes.
  int io_queue_post(struct io_queue* q,
                    void (*fn)(void*),
                    void* ctx,
                    void (*ctx_free)(void*));

  struct io_event io_queue_record(struct io_queue* q);

  // Blocks until ev.seq is retired, or the queue shuts down — callers
  // that need to disambiguate should check io_queue_is_shutdown() after
  // the wait.
  void io_event_wait(const struct io_queue* q, struct io_event ev);

  // Non-blocking io_event_wait; non-zero iff retired.
  int io_event_query(const struct io_queue* q, struct io_event ev);

  int io_queue_is_shutdown(const struct io_queue* q);

#ifdef __cplusplus
}
#endif
