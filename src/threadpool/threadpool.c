#include "threadpool/threadpool.h"

#include "platform/platform.h"
#include "util/prelude.h"

#include <stdalign.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>

#define THREADPOOL_SPIN_ITERS 10000u
#define THREADPOOL_CACHELINE 64u

typedef _Atomic(uint64_t) atomic_u64;
typedef _Atomic(int) atomic_int_t;
typedef _Atomic(size_t) atomic_size_t_t;

typedef void (*pool_task_fn)(int tid, void* ctx);

struct worker
{
  struct platform_thread* thread;
  struct threadpool* pool;
  int tid;
};

struct threadpool
{
  alignas(THREADPOOL_CACHELINE) atomic_u64 epoch;
  alignas(THREADPOOL_CACHELINE) atomic_int_t threads_to_sync;
  alignas(THREADPOOL_CACHELINE) atomic_size_t_t dynamic_progress;
  alignas(THREADPOOL_CACHELINE) atomic_int_t stop;

  struct platform_mutex* mu;
  struct platform_cond* cv;
  struct worker* workers;
  int nworkers;

  pool_task_fn task_fn;
  void* task_ctx;
};

static void
worker_main(void* arg)
{
  struct worker* w = (struct worker*)arg;
  struct threadpool* p = w->pool;
  uint64_t last_epoch = 0;

  for (;;) {
    uint64_t epoch = atomic_load_explicit(&p->epoch, memory_order_acquire);
    if (epoch == last_epoch &&
        !atomic_load_explicit(&p->stop, memory_order_acquire)) {
      for (unsigned i = 0; i < THREADPOOL_SPIN_ITERS; ++i) {
        platform_cpu_pause();
        epoch = atomic_load_explicit(&p->epoch, memory_order_acquire);
        if (epoch != last_epoch ||
            atomic_load_explicit(&p->stop, memory_order_acquire))
          break;
      }
    }

    if (epoch == last_epoch &&
        !atomic_load_explicit(&p->stop, memory_order_acquire)) {
      platform_mutex_lock(p->mu);
      while ((epoch = atomic_load_explicit(&p->epoch, memory_order_acquire)) ==
               last_epoch &&
             !atomic_load_explicit(&p->stop, memory_order_acquire))
        platform_cond_wait(p->cv, p->mu);
      platform_mutex_unlock(p->mu);
    }

    if (atomic_load_explicit(&p->stop, memory_order_acquire))
      break;

    last_epoch = epoch;
    p->task_fn(w->tid, p->task_ctx);
    atomic_fetch_sub_explicit(&p->threads_to_sync, 1, memory_order_release);
  }
}

static void
dispatch(struct threadpool* p, pool_task_fn fn, void* ctx)
{
  if (!p || p->nworkers == 0) {
    fn(0, ctx);
    return;
  }

  p->task_fn = fn;
  p->task_ctx = ctx;
  atomic_store_explicit(&p->threads_to_sync, p->nworkers, memory_order_relaxed);

  platform_mutex_lock(p->mu);
  atomic_fetch_add_explicit(&p->epoch, 1, memory_order_release);
  platform_cond_broadcast(p->cv);
  platform_mutex_unlock(p->mu);

  fn(0, ctx);

  while (atomic_load_explicit(&p->threads_to_sync, memory_order_acquire) > 0)
    platform_cpu_pause();
}

// Helper: free a partially-constructed threadpool from threadpool_new's
// Fail label. Each leaf-free is NULL-safe; this just guards the
// container deref.
static void
threadpool_free_partial(struct threadpool* p)
{
  if (!p)
    return;
  free(p->workers);
  platform_cond_free(p->cv);
  platform_mutex_free(p->mu);
  free(p);
}

struct threadpool*
threadpool_new(int nthreads)
{
  struct threadpool* p = NULL;

  CHECK_SILENT(Fail, nthreads >= 0);
  p = (struct threadpool*)calloc(1, sizeof(*p));
  CHECK_SILENT(Fail, p);

  p->nworkers = nthreads;
  atomic_store_explicit(&p->epoch, 0, memory_order_relaxed);
  atomic_store_explicit(&p->threads_to_sync, 0, memory_order_relaxed);
  atomic_store_explicit(&p->dynamic_progress, 0, memory_order_relaxed);
  atomic_store_explicit(&p->stop, 0, memory_order_relaxed);

  if (nthreads == 0)
    return p;

  p->mu = platform_mutex_new();
  p->cv = platform_cond_new();
  CHECK_SILENT(Fail, p->mu);
  CHECK_SILENT(Fail, p->cv);

  p->workers = (struct worker*)calloc((size_t)nthreads, sizeof(struct worker));
  CHECK_SILENT(Fail, p->workers);

  for (int i = 0; i < nthreads; ++i) {
    p->workers[i].pool = p;
    p->workers[i].tid = i + 1;
    p->workers[i].thread = platform_thread_start(worker_main, &p->workers[i]);
    if (!p->workers[i].thread) {
      atomic_store_explicit(&p->stop, 1, memory_order_release);
      platform_mutex_lock(p->mu);
      platform_cond_broadcast(p->cv);
      platform_mutex_unlock(p->mu);
      for (int j = 0; j < i; ++j)
        platform_thread_join(p->workers[j].thread);
      goto Fail;
    }
  }

  return p;

Fail:
  threadpool_free_partial(p);
  return NULL;
}

void
threadpool_free(struct threadpool* p)
{
  if (!p)
    return;

  if (p->nworkers > 0) {
    atomic_store_explicit(&p->stop, 1, memory_order_release);
    platform_mutex_lock(p->mu);
    platform_cond_broadcast(p->cv);
    platform_mutex_unlock(p->mu);
    for (int i = 0; i < p->nworkers; ++i)
      platform_thread_join(p->workers[i].thread);
    free(p->workers);
    platform_cond_free(p->cv);
    platform_mutex_free(p->mu);
  }
  free(p);
}

int
threadpool_size(const struct threadpool* p)
{
  return p ? p->nworkers + 1 : 1;
}

struct for_n_ctx
{
  size_t n;
  int nslices;
  threadpool_range_fn fn;
  void* user;
};

static void
for_n_runner(int tid, void* vctx)
{
  struct for_n_ctx* c = (struct for_n_ctx*)vctx;
  size_t per = c->n / (size_t)c->nslices;
  size_t rem = c->n % (size_t)c->nslices;
  size_t beg = (size_t)tid * per + ((size_t)tid < rem ? (size_t)tid : rem);
  size_t extra = (size_t)tid < rem ? 1u : 0u;
  size_t end = beg + per + extra;
  if (beg < end)
    c->fn(beg, end, tid, c->user);
}

void
threadpool_for_n(struct threadpool* p,
                 size_t n,
                 threadpool_range_fn fn,
                 void* ctx)
{
  if (n == 0)
    return;
  struct for_n_ctx c = {
    .n = n, .nslices = threadpool_size(p), .fn = fn, .user = ctx
  };
  dispatch(p, for_n_runner, &c);
}

struct for_n_dyn_ctx
{
  size_t n;
  threadpool_index_fn fn;
  void* user;
  atomic_size_t_t* progress;
};

static void
for_n_dyn_runner(int tid, void* vctx)
{
  struct for_n_dyn_ctx* c = (struct for_n_dyn_ctx*)vctx;
  for (;;) {
    size_t i = atomic_fetch_add_explicit(c->progress, 1u, memory_order_relaxed);
    if (i >= c->n)
      break;
    c->fn(i, tid, c->user);
  }
}

void
threadpool_for_n_dynamic(struct threadpool* p,
                         size_t n,
                         threadpool_index_fn fn,
                         void* ctx)
{
  if (n == 0)
    return;
  if (!p) {
    for (size_t i = 0; i < n; ++i)
      fn(i, 0, ctx);
    return;
  }
  atomic_store_explicit(&p->dynamic_progress, 0, memory_order_relaxed);
  struct for_n_dyn_ctx c = {
    .n = n, .fn = fn, .user = ctx, .progress = &p->dynamic_progress
  };
  dispatch(p, for_n_dyn_runner, &c);
}

struct for_threads_ctx
{
  threadpool_broadcast_fn fn;
  int nthreads;
  void* user;
};

static void
for_threads_runner(int tid, void* vctx)
{
  struct for_threads_ctx* c = (struct for_threads_ctx*)vctx;
  c->fn(tid, c->nthreads, c->user);
}

void
threadpool_for_threads(struct threadpool* p,
                       threadpool_broadcast_fn fn,
                       void* ctx)
{
  struct for_threads_ctx c = { .fn = fn,
                               .nthreads = threadpool_size(p),
                               .user = ctx };
  dispatch(p, for_threads_runner, &c);
}
