// Direct unit tests for io_queue: ring_grow and out-of-order completion
// aren't exercised by the store_fs path.
#include "damacy_limits.h"
#include "expect.h"
#include "io_queue/io_queue.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdlib.h>

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

// Parks a worker until gate_release.
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

static int
test_invalid_args(void)
{
  EXPECT(io_queue_post(NULL, inc_run, NULL, NULL) != 0);
  EXPECT(io_queue_create(0) == NULL);
  EXPECT(io_queue_create(-1) == NULL);
  EXPECT(io_queue_create((int)DAMACY_MAX_IO_THREADS + 1) == NULL);

  struct io_queue* q = io_queue_create(1);
  EXPECT(q);
  EXPECT(io_queue_post(q, NULL, NULL, NULL) != 0);
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
  // Empty queue: record returns seq=0, immediately retired.
  struct io_queue* q = io_queue_create(1);
  EXPECT(q);
  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(io_event_query(q, ev));
  io_queue_destroy(q);
  return 0;
}

static int
test_out_of_order_completion(void)
{
  // Regression: retired_seq must track the lowest in-flight seq even
  // when higher seqs finish first. Posts W gate jobs, releases all but
  // gate[0], waits for a barrier post — its retirement proves the
  // higher seqs completed without advancing past gate[0].
  enum
  {
    W = 4
  };
  struct io_queue* q = io_queue_create(W);
  EXPECT(q);

  struct gate gates[W];
  struct io_event evs[W];
  for (int i = 0; i < W; ++i)
    gate_init(&gates[i]);

  for (int i = 0; i < W; ++i) {
    EXPECT(io_queue_post(q, gate_job, &gates[i], NULL) == 0);
    evs[i] = io_queue_record(q);
  }

  for (int i = 0; i < W; ++i)
    while (atomic_load(&gates[i].entered) == 0)
      sched_yield();

  for (int i = W - 1; i >= 1; --i)
    gate_release(&gates[i]);

  // Post W-1 trivial jobs and spin until they all complete. This forces
  // advance_retired to run for each released gate's seq before we
  // observe retired_seq. (io_event_wait can't be used here — retired_seq
  // is stuck at 0 because gate[0] still holds seq=1.)
  atomic_int runs = 0, frees = 0;
  for (int i = 0; i < W - 1; ++i)
    EXPECT(io_queue_post(q, inc_run, ctx_new(&runs, &frees), inc_free) == 0);
  while (atomic_load(&runs) < W - 1)
    sched_yield();
  EXPECT(!io_event_query(q, evs[0]));

  gate_release(&gates[0]);
  io_event_wait(q, evs[W - 1]);
  EXPECT(io_event_query(q, evs[W - 1]));

  io_queue_destroy(q);
  for (int i = 0; i < W; ++i)
    gate_destroy(&gates[i]);
  return 0;
}

static int
test_ring_grow(void)
{
  // Park the only worker, post > initial cap to force a grow.
  struct io_queue* q = io_queue_create(1);
  EXPECT(q);

  struct gate g;
  gate_init(&g);
  EXPECT(io_queue_post(q, gate_job, &g, NULL) == 0);
  while (atomic_load(&g.entered) == 0)
    sched_yield();

  atomic_int runs = 0, frees = 0;
  const int N = (int)DAMACY_IO_QUEUE_INITIAL_CAP + 256;
  for (int i = 0; i < N; ++i)
    EXPECT(io_queue_post(q, inc_run, ctx_new(&runs, &frees), inc_free) == 0);

  gate_release(&g);

  struct io_event ev = io_queue_record(q);
  io_event_wait(q, ev);
  EXPECT(atomic_load(&runs) == N); // gate doesn't touch runs/frees
  EXPECT(atomic_load(&frees) == N);

  io_queue_destroy(q);
  gate_destroy(&g);
  return 0;
}

int
main(void)
{
  RUN(test_invalid_args);
  RUN(test_basic_async);
  RUN(test_record_before_post);
  RUN(test_out_of_order_completion);
  RUN(test_ring_grow);
  log_info("all tests passed");
  return 0;
}
