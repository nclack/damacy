// Small fixed-size thread pool for parallel-for over CPU work.
// Workers spin briefly when idle, then sleep on a condvar.
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

  struct threadpool;

  // Create a pool with `nthreads` workers (excluding the calling thread).
  // The caller participates as tid 0; workers occupy tids 1..nthreads.
  // Total parallelism = nthreads + 1. nthreads >= 0; 0 runs serially on the
  // caller. Returns NULL on failure.
  struct threadpool* threadpool_new(int nthreads);
  void threadpool_free(struct threadpool* p);

  // Total parallelism = workers + 1. NULL pool counts as 1.
  int threadpool_size(const struct threadpool* p);

  // Static parallel for: divide [0, n) into threadpool_size(p) slices.
  typedef void (*threadpool_range_fn)(size_t beg,
                                      size_t end,
                                      int tid,
                                      void* ctx);

  void threadpool_for_n(struct threadpool* p,
                        size_t n,
                        threadpool_range_fn fn,
                        void* ctx);

  // Dynamic parallel for: each worker fetch_add's a shared counter.
  typedef void (*threadpool_index_fn)(size_t i, int tid, void* ctx);

  void threadpool_for_n_dynamic(struct threadpool* p,
                                size_t n,
                                threadpool_index_fn fn,
                                void* ctx);

  // Broadcast: call fn(tid, nthreads, ctx) once per participant.
  typedef void (*threadpool_broadcast_fn)(int tid, int nthreads, void* ctx);

  void threadpool_for_threads(struct threadpool* p,
                              threadpool_broadcast_fn fn,
                              void* ctx);

#ifdef __cplusplus
}
#endif
