#include "platform/platform.h"

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

size_t
platform_page_size(void)
{
  return (size_t)sysconf(_SC_PAGESIZE);
}

size_t
platform_page_alignment(void)
{
  size_t ps = platform_page_size();
  return ps > 0 ? ps : 4096;
}

size_t
platform_available_memory(void)
{
  long pages = sysconf(_SC_AVPHYS_PAGES);
  long page_sz = sysconf(_SC_PAGESIZE);
  if (pages > 0 && page_sz > 0)
    return (size_t)pages * (size_t)page_sz;
  return 0;
}

void*
platform_aligned_alloc(size_t alignment, size_t size)
{
  return aligned_alloc(alignment, size);
}

void
platform_aligned_free(void* ptr)
{
  free(ptr);
}

void
platform_sleep_ns(int64_t ns)
{
  struct timespec ts = {
    .tv_sec = ns / 1000000000LL,
    .tv_nsec = ns % 1000000000LL,
  };
  nanosleep(&ts, NULL);
}

static int64_t
monotonic_ns(void)
{
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  return (int64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

float
platform_toc(struct platform_clock* clock)
{
  int64_t now = monotonic_ns();
  float elapsed = (float)((now - clock->last_ns) / 1e9);
  clock->last_ns = now;
  return elapsed;
}

void
platform_localtime(const time_t* t, struct tm* out)
{
  localtime_r(t, out);
}

void
platform_call_once(platform_once* flag, void (*fn)(void))
{
  pthread_once(flag, fn);
}

struct platform_thread
{
  pthread_t handle;
  void (*fn)(void*);
  void* arg;
};

static void*
thread_trampoline(void* p)
{
  struct platform_thread* t = (struct platform_thread*)p;
  t->fn(t->arg);
  return NULL;
}

struct platform_thread*
platform_thread_start(void (*fn)(void*), void* arg)
{
  struct platform_thread* t =
    (struct platform_thread*)calloc(1, sizeof(struct platform_thread));
  if (!t)
    return NULL;
  t->fn = fn;
  t->arg = arg;
  if (pthread_create(&t->handle, NULL, thread_trampoline, t) != 0) {
    free(t);
    return NULL;
  }
  return t;
}

int
platform_thread_join(struct platform_thread* t)
{
  if (!t)
    return -1;
  int rc = pthread_join(t->handle, NULL);
  free(t);
  return rc;
}

struct platform_mutex
{
  pthread_mutex_t m;
};

struct platform_mutex*
platform_mutex_new(void)
{
  struct platform_mutex* m =
    (struct platform_mutex*)calloc(1, sizeof(struct platform_mutex));
  if (!m)
    return NULL;
  if (pthread_mutex_init(&m->m, NULL) != 0) {
    free(m);
    return NULL;
  }
  return m;
}

void
platform_mutex_free(struct platform_mutex* m)
{
  if (!m)
    return;
  pthread_mutex_destroy(&m->m);
  free(m);
}

void
platform_mutex_lock(struct platform_mutex* m)
{
  pthread_mutex_lock(&m->m);
}

void
platform_mutex_unlock(struct platform_mutex* m)
{
  pthread_mutex_unlock(&m->m);
}

struct platform_cond
{
  pthread_cond_t c;
};

struct platform_cond*
platform_cond_new(void)
{
  struct platform_cond* c =
    (struct platform_cond*)calloc(1, sizeof(struct platform_cond));
  if (!c)
    return NULL;
  if (pthread_cond_init(&c->c, NULL) != 0) {
    free(c);
    return NULL;
  }
  return c;
}

void
platform_cond_free(struct platform_cond* c)
{
  if (!c)
    return;
  pthread_cond_destroy(&c->c);
  free(c);
}

void
platform_cond_wait(struct platform_cond* c, struct platform_mutex* m)
{
  pthread_cond_wait(&c->c, &m->m);
}

int
platform_cond_timedwait_ms(struct platform_cond* c,
                           struct platform_mutex* m,
                           int timeout_ms)
{
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  ts.tv_sec += timeout_ms / 1000;
  ts.tv_nsec += (long)(timeout_ms % 1000) * 1000000L;
  if (ts.tv_nsec >= 1000000000L) {
    ts.tv_sec += 1;
    ts.tv_nsec -= 1000000000L;
  }
  return pthread_cond_timedwait(&c->c, &m->m, &ts) == ETIMEDOUT ? 1 : 0;
}

void
platform_cond_broadcast(struct platform_cond* c)
{
  pthread_cond_broadcast(&c->c);
}

void
platform_cpu_pause(void)
{
#if defined(__i386__) || defined(__x86_64__)
  __asm__ __volatile__("pause" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
  __asm__ __volatile__("yield" ::: "memory");
#else
  sched_yield();
#endif
}

int
platform_default_thread_count(void)
{
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (int)n : 1;
}

void*
platform_dlopen(const char* path)
{
  return dlopen(path, RTLD_LAZY | RTLD_LOCAL);
}

void*
platform_dlsym(void* handle, const char* name)
{
  return dlsym(handle, name);
}

void
platform_dlclose(void* handle)
{
  if (!handle)
    return;
  dlclose(handle);
}

const char*
platform_dlerror(void)
{
  return dlerror();
}

const char*
platform_getenv(const char* name)
{
  return getenv(name);
}
