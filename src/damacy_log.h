#pragma once

// Public logging control for damacy consumers.
//
// Use these to forward a caller-supplied log level into damacy, intercept
// messages with a callback for routing into a host logging framework
// (e.g. Python's `logging`), or silence the default stderr sink.
//
// Callbacks fire on whichever thread produced the log line. Logger state
// is process-global; there is no per-instance scoping.

#include <time.h>

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum damacy_log_level
  {
    DAMACY_LOG_TRACE = 0,
    DAMACY_LOG_DEBUG,
    DAMACY_LOG_INFO,
    DAMACY_LOG_WARN,
    DAMACY_LOG_ERROR,
    DAMACY_LOG_FATAL,
  } damacy_log_level;

  // Pointers are valid only for the duration of the callback; copy
  // anything that needs to outlive it.
  typedef struct damacy_log_event
  {
    const char* msg;
    const char* file;
    int line;
    damacy_log_level level;
    struct timespec time;
  } damacy_log_event;

  typedef void (*damacy_log_fn)(const damacy_log_event* ev, void* udata);

  // Gate the default stderr sink. Registered callbacks are unaffected;
  // they each have their own threshold set at registration.
  void damacy_log_set_level(damacy_log_level level);

  // Suppress the default stderr sink. Registered callbacks keep firing.
  void damacy_log_set_quiet(int quiet);

  // Register fn to receive events at or above threshold, independent of
  // the global level set by damacy_log_set_level. Returns 0 on success,
  // non-zero if the callback table is full.
  //
  // The callback must not invoke damacy log macros (log_info, etc.) —
  // doing so will recurse without bound.
  int damacy_log_add_callback(damacy_log_fn fn,
                              void* udata,
                              damacy_log_level threshold);

#ifdef __cplusplus
}
#endif
