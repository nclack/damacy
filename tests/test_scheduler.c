// Unit tests for src/scheduler/. Verifies the lifecycle + the shared
// lock/wait/broadcast handshake without any damacy / CUDA dependency.
//
//   test_create_destroy           — basic spawn + join
//   test_step_repeats_without_signal — a zero-returning step keeps running
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
  struct scheduler* s = scheduler_create(step_count, &c, 1000000, NULL, NULL);
  EXPECT(s != NULL);
  scheduler_destroy(s);
  scheduler_destroy(NULL); // NULL-safe
  return 0;
}

struct progress
{
  struct platform_mutex* m;
  struct platform_cond* cv;
  int n_steps; // under m
};

static int
step_without_signal(void* p)
{
  struct progress* c = (struct progress*)p;
  platform_mutex_lock(c->m);
  c->n_steps += 1;
  platform_cond_broadcast(c->cv);
  platform_mutex_unlock(c->m);
  return 0;
}

static int
test_step_repeats_without_signal(void)
{
  struct progress c = { platform_mutex_new(), platform_cond_new(), 0 };
  EXPECT(c.m != NULL && c.cv != NULL);
  struct scheduler* s =
    scheduler_create(step_without_signal, &c, 500000, NULL, NULL);
  EXPECT(s != NULL);

  // Observe progress independently of the scheduler's ready notification.
  // There is no scheduling-speed requirement; CTest bounds a deadlocked test.
  platform_mutex_lock(c.m);
  while (c.n_steps < 10)
    platform_cond_wait(c.cv, c.m);
  int n = c.n_steps;
  platform_mutex_unlock(c.m);

  scheduler_destroy(s);
  platform_cond_free(c.cv);
  platform_mutex_free(c.m);
  EXPECT(n >= 10);
  return 0;
}

static int
test_signal_wakes_waiter(void)
{
  struct ctx c = { 0, 5 };
  struct scheduler* s = scheduler_create(step_count, &c, 500000, NULL, NULL);
  EXPECT(s != NULL);
  scheduler_lock(s);
  // The worker can reach the target only after scheduler_wait releases the
  // lock.
  c.n_steps = 0;
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
  scheduler_lock(a->s);
  *a->flag = 1;
  scheduler_broadcast(a->s);
  scheduler_unlock(a->s);
}

static int
test_external_broadcast(void)
{
  struct scheduler* s = scheduler_create(step_noop, NULL, 500000, NULL, NULL);
  EXPECT(s != NULL);
  int flag = 0;
  struct broadcaster_args a = { s, &flag };
  // The broadcaster cannot set the predicate until scheduler_wait releases
  // this lock, so the test always exercises the wait/broadcast handshake.
  scheduler_lock(s);
  struct platform_thread* t = platform_thread_start(broadcaster_fn, &a);
  EXPECT(t != NULL);

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
  EXPECT(scheduler_create(NULL, NULL, 1000, NULL, NULL) == NULL);
  EXPECT(scheduler_create(step_noop, NULL, 0, NULL, NULL) == NULL);
  EXPECT(scheduler_create(step_noop, NULL, -1, NULL, NULL) == NULL);
  return 0;
}

int
main(void)
{
  RUN(test_create_destroy);
  RUN(test_step_repeats_without_signal);
  RUN(test_signal_wakes_waiter);
  RUN(test_external_broadcast);
  RUN(test_invalid_args);
  return 0;
}
