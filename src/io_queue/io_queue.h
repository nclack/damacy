// Multi-worker job queue with monotonic-sequence completion events.
//
// post() enqueues fn(ctx). Workers may execute jobs in parallel, finishing
// in any order. record() captures the current sequence high-water mark.
// io_event_wait() blocks until *every* job posted before that record()
// call has finished, regardless of completion order. retired_seq advances
// to the lowest seq still in flight or queued, derived from each worker's
// currently-processing seq plus the queue's tail entry — no per-seq
// bookkeeping, so out-of-order completion patterns can't desynchronize it.
//
// nthreads is encapsulated in the queue, capped at DAMACY_MAX_IO_THREADS.
// nthreads=0 runs jobs synchronously on the posting thread.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct io_queue;

  struct io_event
  {
    uint64_t seq;
  };

  // Create a queue with `nthreads` worker threads. nthreads must be >= 0.
  // Returns NULL on failure.
  struct io_queue* io_queue_create(int nthreads);
  void io_queue_destroy(struct io_queue* q);

  // Post a job. Returns 0 on success, non-zero on failure.
  // On failure, the job is NOT posted; caller must free ctx.
  // ctx_free (if non-NULL) is called with ctx after fn completes.
  int io_queue_post(struct io_queue* q,
                    void (*fn)(void*),
                    void* ctx,
                    void (*ctx_free)(void*));

  // Capture the current sequence number.
  struct io_event io_queue_record(struct io_queue* q);

  // Block until all jobs up to and including ev.seq have completed.
  void io_event_wait(const struct io_queue* q, struct io_event ev);

  // Non-blocking equivalent of io_event_wait. Returns non-zero if every
  // job up to and including ev.seq has finished, zero otherwise.
  int io_event_query(const struct io_queue* q, struct io_event ev);

  // Returns non-zero once io_queue_destroy has been called.
  int io_queue_is_shutdown(const struct io_queue* q);

#ifdef __cplusplus
}
#endif
