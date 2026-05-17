#pragma once

#include <stddef.h>
#include <stdint.h>
#include <time.h>

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

  void platform_localtime(const time_t* t, struct tm* out);

#include <pthread.h>
  typedef pthread_once_t platform_once;
#define PLATFORM_ONCE_INIT PTHREAD_ONCE_INIT

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
  // Returns 1 on timeout, 0 on signal/spurious-wake.
  int platform_cond_timedwait_ms(struct platform_cond* c,
                                 struct platform_mutex* m,
                                 int timeout_ms);
  void platform_cond_broadcast(struct platform_cond* c);

  void platform_cpu_pause(void);
  int platform_default_thread_count(void);

  // Path is platform-native; no translation between "libfoo.so.N" / "foo.dll".
  void* platform_dlopen(const char* path);
  void* platform_dlsym(void* handle, const char* name);
  void platform_dlclose(void* handle);
  const char* platform_dlerror(void);

#ifdef __cplusplus
}
#endif
