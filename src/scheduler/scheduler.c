#include "scheduler/scheduler.h"

#include "log/log.h"
#include "numa/numa.h"
#include "platform/platform.h"

#include <stdlib.h>

struct scheduler
{
  struct platform_thread* thread;
  struct platform_mutex* m;
  struct platform_cond* cv;
  scheduler_step_fn step;
  void* arg;
  int64_t idle_ns;
  int shutdown; // protected by m
  // numa_apply_thread_affinity is a no-op when affinity.node < 0.
  struct numa_resolved affinity;
};

static void
worker_main(void* p)
{
  struct scheduler* s = (struct scheduler*)p;
  numa_apply_thread_affinity(&s->affinity, "scheduler_worker");
  for (;;) {
    platform_mutex_lock(s->m);
    if (s->shutdown) {
      platform_mutex_unlock(s->m);
      return;
    }
    int ready = s->step(s->arg);
    if (ready)
      platform_cond_broadcast(s->cv);
    platform_mutex_unlock(s->m);
    platform_sleep_ns(s->idle_ns);
  }
}

struct scheduler*
scheduler_create(scheduler_step_fn step,
                 void* arg,
                 int64_t idle_ns,
                 const struct numa_resolved* affinity)
{
  if (!step || idle_ns <= 0) {
    log_error("scheduler: invalid arguments (step=%d idle_ns=%lld)",
              step ? 1 : 0,
              (long long)idle_ns);
    return NULL;
  }
  struct scheduler* s = (struct scheduler*)calloc(1, sizeof(*s));
  if (!s) {
    log_error("scheduler: alloc failed");
    return NULL;
  }
  s->step = step;
  s->arg = arg;
  s->idle_ns = idle_ns;
  // Copy unconditionally; numa_apply_thread_affinity no-ops when
  // node < 0. calloc would leave node = 0 (a valid node id), so be
  // explicit when the caller passes NULL.
  if (affinity)
    s->affinity = *affinity;
  else
    s->affinity.node = -1;
  s->m = platform_mutex_new();
  s->cv = platform_cond_new();
  if (!s->m || !s->cv) {
    log_error("scheduler: mutex/cond alloc failed");
    scheduler_destroy(s);
    return NULL;
  }
  s->thread = platform_thread_start(worker_main, s);
  if (!s->thread) {
    log_error("scheduler: thread start failed");
    scheduler_destroy(s);
    return NULL;
  }
  return s;
}

void
scheduler_destroy(struct scheduler* s)
{
  if (!s)
    return;
  if (s->thread) {
    platform_mutex_lock(s->m);
    s->shutdown = 1;
    platform_mutex_unlock(s->m);
    platform_thread_join(s->thread);
  }
  platform_cond_free(s->cv);
  platform_mutex_free(s->m);
  free(s);
}

void
scheduler_lock(struct scheduler* s)
{
  platform_mutex_lock(s->m);
}

void
scheduler_unlock(struct scheduler* s)
{
  platform_mutex_unlock(s->m);
}

void
scheduler_wait(struct scheduler* s)
{
  platform_cond_wait(s->cv, s->m);
}

void
scheduler_wait_diag(struct scheduler* s, const char* site, int timeout_ms)
{
  if (platform_cond_timedwait_ms(s->cv, s->m, timeout_ms))
    log_warn("scheduler_wait_diag: %d ms timeout at %s", timeout_ms, site);
}

void
scheduler_broadcast(struct scheduler* s)
{
  platform_cond_broadcast(s->cv);
}
