// Direct unit tests for io_queue. The store_fs path exercises the happy
// case but never fills the ring (in-flight ≪ DAMACY_IO_QUEUE_INITIAL_CAP)
// and never uses nthreads=0, so ring_grow and the synchronous fast path
// were uncovered. These tests target those gaps.
#include "damacy_limits.h"
#include "expect.h"
#include "io_queue/io_queue.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>

// Per-job ctx: heap-allocated, freed by inc_free. Carries two shared
// counters so we can verify both fn and ctx_free were called N times.
struct ctx
{
  atomic_int* runs;
  atomic_int* frees;
};

static struct ctx*
ctx_new(atomic_int* runs, atomic_int* frees)
{
  struct ctx* c = (struct ctx*)malloc(sizeof(*c));
  c->runs = runs;
  c->frees = frees;
  return c;
}

static void
inc_run(void* p)
{
  struct ctx* c = (struct ctx*)p;
  atomic_fetch_add(c->runs, 1);
}

static void
inc_free(void* p)
{
  struct ctx* c = (struct ctx*)p;
  atomic_fetch_add(c->frees, 1);
  free(c);
}

// gate_job parks a worker on a CV until the test releases it. Used to
// hold the (single) worker so subsequent posts pile up in the ring.
struct gate
{
  pthread_mutex_t m;
  pthread_cond_t cv;
  int released;
  atomic_int entered;
};

static void
gate_init(struct gate* g)
{
  pthread_mutex_init(&g->m, NULL);
  pthread_cond_init(&g->cv, NULL);
  g->released = 0;
  atomic_store(&g->entered, 0);
}

static void
gate_destroy(struct gate* g)
{
  pthread_mutex_destroy(&g->m);
  pthread_cond_destroy(&g->cv);
}

static void
gate_job(void* p)
{
  struct gate* g = (struct gate*)p;
  pthread_mutex_lock(&g->m);
  atomic_fetch_add(&g->entered, 1);
  while (!g->released)
    pthread_cond_wait(&g->cv, &g->m);
  pthread_mutex_unlock(&g->m);
}

static void
gate_release(struct gate* g)
{
  pthread_mutex_lock(&g->m);
  g->released = 1;
  pthread_cond_broadcast(&g->cv);
  pthread_mutex_unlock(&g->m);
}

// ---- tests ------------------------------------------------------------------

static int
test_invalid_args(void)
{
  // NULL queue is rejected before any deref.
  EXPECT(io_queue_post(NULL, inc_run, NULL, NULL) != 0);

  struct io_queue* q = io_queue_create(1);
  EXPECT(q);
  // NULL fn is rejected; nothing is posted, nothing leaked.
  EXPECT(io_queue_post(q, NULL, NULL, NULL) != 0);
  io_queue_destroy(q);
  return 0;
}

static int
test_synchronous(void)
{
  // nthreads=0: jobs run on the posting thread, retired_seq advances
  // before post() returns.
  struct io_queue* q = io_queue_create(0);
  EXPECT(q);

  atomic_int runs = 0, frees = 0;
  const int N = 5;
  for (int i = 0; i < N; ++i)
    EXPECT(io_queue_post(q, inc_run, ctx_new(&runs, &frees), inc_free) == 0);

  EXPECT(atomic_load(&runs) == N);
  EXPECT(atomic_load(&frees) == N);

  // wait/query must work in the sync case too.
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(io_event_query(q, ev));

  io_queue_destroy(q);
  return 0;
}

static int
test_basic_async(void)
{
  struct io_queue* q = io_queue_create(2);
  EXPECT(q);

  atomic_int runs = 0, frees = 0;
  const int N = 32;
  for (int i = 0; i < N; ++i)
    EXPECT(io_queue_post(q, inc_run, ctx_new(&runs, &frees), inc_free) == 0);

  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(io_event_query(q, ev));
  EXPECT(atomic_load(&runs) == N);
  EXPECT(atomic_load(&frees) == N);

  io_queue_destroy(q);
  return 0;
}

static int
test_record_before_post(void)
{
  // record() with no posts returns seq=0; wait() must return immediately
  // and query() must report retired.
  struct io_queue* q = io_queue_create(1);
  EXPECT(q);
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(io_event_query(q, ev));
  io_queue_destroy(q);
  return 0;
}

static int
test_ring_grow(void)
{
  // Force ring_grow by occupying the sole worker with a blocking job
  // and posting > 2 × DAMACY_IO_QUEUE_INITIAL_CAP trivial jobs so the
  // ring grows at least twice (initial 512 → ~630 → ~772).
  struct io_queue* q = io_queue_create(1);
  EXPECT(q);

  struct gate g;
  gate_init(&g);
  EXPECT(io_queue_post(q, gate_job, &g, NULL) == 0);

  // Spin until the gate job has actually started — only then is the
  // worker parked and guaranteed not to drain the ring under us.
  while (atomic_load(&g.entered) == 0)
    sched_yield();

  atomic_int runs = 0, frees = 0;
  const int N = (int)DAMACY_IO_QUEUE_INITIAL_CAP + 256; // forces one grow
  for (int i = 0; i < N; ++i)
    EXPECT(io_queue_post(q, inc_run, ctx_new(&runs, &frees), inc_free) == 0);

  // Release the worker; trivial jobs drain in posted order.
  gate_release(&g);

  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(atomic_load(&runs) == N);
  EXPECT(atomic_load(&frees) == N);

  io_queue_destroy(q);
  gate_destroy(&g);
  return 0;
}

int
main(void)
{
  RUN(test_invalid_args);
  RUN(test_synchronous);
  RUN(test_basic_async);
  RUN(test_record_before_post);
  RUN(test_ring_grow);
  log_info("all tests passed");
  return 0;
}
