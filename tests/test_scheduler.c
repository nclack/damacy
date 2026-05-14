// Unit tests for src/scheduler/. Verifies the lifecycle + the shared
// lock/wait/broadcast handshake without any damacy / CUDA dependency.
//
//   test_create_destroy           — basic spawn + join
//   test_step_runs_periodically   — step is called repeatedly
//   test_signal_wakes_waiter      — step returning non-zero wakes a waiter
//   test_external_broadcast       — non-worker broadcaster wakes a waiter
//   test_invalid_args             — null step / non-positive idle rejected

#include "expect.h"
#include "platform/platform.h"
#include "scheduler/scheduler.h"

#include <stdlib.h>

struct ctx
{
  int n_steps; // under scheduler lock
  int target;  // step returns 1 once n_steps == target
};

static int
step_count(void* p)
{
  struct ctx* c = (struct ctx*)p;
  c->n_steps += 1;
  return c->n_steps == c->target ? 1 : 0;
}

static int
test_create_destroy(void)
{
  struct ctx c = { 0, 0 };
  struct scheduler* s = scheduler_create(step_count, &c, 1000000);
  EXPECT(s != NULL);
  scheduler_destroy(s);
  scheduler_destroy(NULL); // NULL-safe
  return 0;
}

static int
test_step_runs_periodically(void)
{
  struct ctx c = { 0, -1 }; // never signal; just count
  struct scheduler* s = scheduler_create(step_count, &c, 500000); // 500 µs
  EXPECT(s != NULL);
  platform_sleep_ns(20000000); // 20 ms → expect >> 10 ticks
  scheduler_lock(s);
  int n = c.n_steps;
  scheduler_unlock(s);
  scheduler_destroy(s);
  EXPECT(n >= 10);
  return 0;
}

static int
test_signal_wakes_waiter(void)
{
  struct ctx c = { 0, 5 };
  struct scheduler* s = scheduler_create(step_count, &c, 500000);
  EXPECT(s != NULL);
  scheduler_lock(s);
  while (c.n_steps < c.target)
    scheduler_wait(s);
  EXPECT(c.n_steps >= c.target);
  scheduler_unlock(s);
  scheduler_destroy(s);
  return 0;
}

// Step that does nothing (we only want to verify external broadcast wakes us).
static int
step_noop(void* p)
{
  (void)p;
  return 0;
}

struct broadcaster_args
{
  struct scheduler* s;
  int* flag; // set under lock then broadcast
};

static void
broadcaster_fn(void* p)
{
  struct broadcaster_args* a = (struct broadcaster_args*)p;
  // Give the main thread time to enter scheduler_wait.
  platform_sleep_ns(10000000); // 10 ms
  scheduler_lock(a->s);
  *a->flag = 1;
  scheduler_broadcast(a->s);
  scheduler_unlock(a->s);
}

static int
test_external_broadcast(void)
{
  struct scheduler* s =
    scheduler_create(step_noop, NULL, 1000000000); // 1 s tick
  EXPECT(s != NULL);
  int flag = 0;
  struct broadcaster_args a = { s, &flag };
  struct platform_thread* t = platform_thread_start(broadcaster_fn, &a);
  EXPECT(t != NULL);

  scheduler_lock(s);
  while (!flag)
    scheduler_wait(s);
  scheduler_unlock(s);
  platform_thread_join(t);
  scheduler_destroy(s);
  EXPECT(flag == 1);
  return 0;
}

static int
test_invalid_args(void)
{
  EXPECT(scheduler_create(NULL, NULL, 1000) == NULL);
  EXPECT(scheduler_create(step_noop, NULL, 0) == NULL);
  EXPECT(scheduler_create(step_noop, NULL, -1) == NULL);
  return 0;
}

int
main(void)
{
  RUN(test_create_destroy);
  RUN(test_step_runs_periodically);
  RUN(test_signal_wakes_waiter);
  RUN(test_external_broadcast);
  RUN(test_invalid_args);
  return 0;
}
