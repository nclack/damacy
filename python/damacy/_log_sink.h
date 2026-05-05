#pragma once

// Internal header for the Python log sink. Owned by _native.c; the
// implementation lives in _log_sink.c.

#ifdef __cplusplus
extern "C"
{
#endif

  // Wire a damacy log callback into a fixed-size MPSC ring and start a
  // GIL-acquiring daemon thread that drains the ring into Python's
  // `logging.getLogger("damacy")`. Idempotent — repeated calls return 0.
  // Returns 0 on success, -1 on failure (with a Python error set).
  int log_sink_install(void);

  // Stop the drain thread, join it, and clear the ring. Safe to call
  // even if log_sink_install was never called or already torn down.
  void log_sink_shutdown(void);

#ifdef __cplusplus
}
#endif
