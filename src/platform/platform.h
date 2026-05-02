#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  void platform_sleep_ns(int64_t ns);
  size_t platform_page_size(void);
  size_t platform_page_alignment(void);
  void* platform_aligned_alloc(size_t alignment, size_t size);
  void platform_aligned_free(void* ptr);
  size_t platform_available_memory(void);

  struct platform_clock
  {
    int64_t last_ns;
  };

  float platform_toc(struct platform_clock* clock);

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
  typedef INIT_ONCE platform_once;
#define PLATFORM_ONCE_INIT INIT_ONCE_STATIC_INIT
#else
#include <pthread.h>
typedef pthread_once_t platform_once;
#define PLATFORM_ONCE_INIT PTHREAD_ONCE_INIT
#endif

  void platform_call_once(platform_once* flag, void (*fn)(void));

  struct platform_thread;
  struct platform_mutex;
  struct platform_cond;

  struct platform_thread* platform_thread_start(void (*fn)(void*), void* arg);
  int platform_thread_join(struct platform_thread* t);

  struct platform_mutex* platform_mutex_new(void);
  void platform_mutex_free(struct platform_mutex* m);
  void platform_mutex_lock(struct platform_mutex* m);
  void platform_mutex_unlock(struct platform_mutex* m);

  struct platform_cond* platform_cond_new(void);
  void platform_cond_free(struct platform_cond* c);
  void platform_cond_wait(struct platform_cond* c, struct platform_mutex* m);
  void platform_cond_broadcast(struct platform_cond* c);

  void platform_cpu_pause(void);
  int platform_default_thread_count(void);

#ifdef __cplusplus
}
#endif
