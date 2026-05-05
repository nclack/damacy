// Python log sink for damacy. Producer threads (any thread inside the C
// library) never touch the GIL — they push fixed-size records onto an
// MPSC ring guarded by a pthread mutex. A drain thread owned by this
// module wakes on a condvar, batches records out of the ring, then
// acquires the GIL to dispatch each one to logging.getLogger("damacy").
//
// On full: drop newest, increment a counter; once per drained batch we
// emit a single synthetic record reporting the drop count since the
// previous report.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_log_sink.h"
#include "damacy_log.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define RING_CAP 1024
#define MSG_BYTES 384
#define FILE_BYTES 128

struct slot
{
  damacy_log_level level;
  int line;
  struct timespec time;
  char file[FILE_BYTES];
  char msg[MSG_BYTES];
};

static struct
{
  struct slot slots[RING_CAP];
  uint64_t head;
  uint64_t tail;
  uint64_t dropped;
  pthread_mutex_t mu;
  pthread_cond_t cv;
  pthread_t drain_thread;
  int running;
  int installed;
  PyObject* logger;
} R;

// damacy levels are 0..5; logging.* uses 5/10/20/30/40/50. TRACE has no
// stdlib counterpart so we map it to 5 (below DEBUG).
static int
to_py_level(damacy_log_level lvl)
{
  switch (lvl) {
    case DAMACY_LOG_TRACE:
      return 5;
    case DAMACY_LOG_DEBUG:
      return 10;
    case DAMACY_LOG_INFO:
      return 20;
    case DAMACY_LOG_WARN:
      return 30;
    case DAMACY_LOG_ERROR:
      return 40;
    case DAMACY_LOG_FATAL:
      return 50;
  }
  return 20;
}

// Producer side. Called from arbitrary C threads with no GIL guarantee.
// Must not block on Python state.
static void
sink_cb(const damacy_log_event* ev, void* udata)
{
  (void)udata;
  pthread_mutex_lock(&R.mu);
  if (R.head - R.tail >= RING_CAP) {
    R.dropped++;
    pthread_mutex_unlock(&R.mu);
    return;
  }
  struct slot* s = &R.slots[R.head % RING_CAP];
  s->level = ev->level;
  s->line = ev->line;
  s->time = ev->time;
  if (ev->file) {
    size_t n = strnlen(ev->file, FILE_BYTES - 1);
    memcpy(s->file, ev->file, n);
    s->file[n] = '\0';
  } else {
    s->file[0] = '\0';
  }
  if (ev->msg) {
    size_t n = strnlen(ev->msg, MSG_BYTES - 1);
    memcpy(s->msg, ev->msg, n);
    s->msg[n] = '\0';
  } else {
    s->msg[0] = '\0';
  }
  R.head++;
  pthread_cond_signal(&R.cv);
  pthread_mutex_unlock(&R.mu);
}

// Drain one batch under the GIL. Caller holds the GIL.
static void
dispatch_batch(struct slot* batch, size_t n, uint64_t dropped)
{
  PyObject* log_method =
    R.logger ? PyObject_GetAttrString(R.logger, "log") : NULL;
  if (!log_method) {
    if (PyErr_Occurred())
      PyErr_Clear();
    return;
  }
  for (size_t i = 0; i < n; ++i) {
    const struct slot* s = &batch[i];
    PyObject* res =
      PyObject_CallFunction(log_method, "is", to_py_level(s->level), s->msg);
    if (res) {
      Py_DECREF(res);
    } else if (PyErr_Occurred()) {
      // Don't let a logging-side exception kill the drain thread; just
      // print it via the default unraisable hook and continue.
      PyErr_WriteUnraisable(log_method);
    }
  }
  if (dropped > 0) {
    char buf[64];
    snprintf(buf,
             sizeof buf,
             "damacy: %llu log records dropped",
             (unsigned long long)dropped);
    PyObject* res = PyObject_CallFunction(log_method, "is", 30 /*WARN*/, buf);
    if (res) {
      Py_DECREF(res);
    } else if (PyErr_Occurred()) {
      PyErr_WriteUnraisable(log_method);
    }
  }
  Py_DECREF(log_method);
}

// Drain thread main. Sleeps on the condvar; on wake, copies pending
// records out under the mutex (no GIL), then dispatches under the GIL.
static void*
drain_main(void* arg)
{
  (void)arg;
  // Local batch sized to drain in one pass without holding the mutex
  // for too long. Bound by ring capacity.
  static __thread struct slot batch[RING_CAP];

  for (;;) {
    pthread_mutex_lock(&R.mu);
    while (R.running && R.head == R.tail)
      pthread_cond_wait(&R.cv, &R.mu);

    size_t n = (size_t)(R.head - R.tail);
    if (n > RING_CAP)
      n = RING_CAP;
    for (size_t i = 0; i < n; ++i)
      batch[i] = R.slots[(R.tail + i) % RING_CAP];
    R.tail += n;
    uint64_t dropped = R.dropped;
    R.dropped = 0;
    int running = R.running;
    pthread_mutex_unlock(&R.mu);

    if (n > 0 || dropped > 0) {
      PyGILState_STATE g = PyGILState_Ensure();
      dispatch_batch(batch, n, dropped);
      PyGILState_Release(g);
    }

    if (!running) {
      // One last drain happened above; if more arrived after we read
      // R.head, they'll be lost — that's an explicit shutdown loss
      // we accept rather than risk hanging on join.
      break;
    }
  }
  return NULL;
}

int
log_sink_install(void)
{
  if (R.installed)
    return 0;

  // Resolve the Python logger now while we hold the GIL.
  PyObject* logging = PyImport_ImportModule("logging");
  if (!logging)
    return -1;
  PyObject* get_logger = PyObject_GetAttrString(logging, "getLogger");
  Py_DECREF(logging);
  if (!get_logger)
    return -1;
  PyObject* logger = PyObject_CallFunction(get_logger, "s", "damacy");
  Py_DECREF(get_logger);
  if (!logger)
    return -1;
  R.logger = logger;

  if (pthread_mutex_init(&R.mu, NULL) != 0)
    goto Fail;
  if (pthread_cond_init(&R.cv, NULL) != 0) {
    pthread_mutex_destroy(&R.mu);
    goto Fail;
  }
  R.head = R.tail = R.dropped = 0;
  R.running = 1;

  if (pthread_create(&R.drain_thread, NULL, drain_main, NULL) != 0) {
    pthread_cond_destroy(&R.cv);
    pthread_mutex_destroy(&R.mu);
    goto Fail;
  }

  // Register the producer-side callback. Threshold is TRACE so we route
  // everything to Python — Python's logging filters by its own level.
  if (damacy_log_add_callback(sink_cb, NULL, DAMACY_LOG_TRACE) != 0) {
    pthread_mutex_lock(&R.mu);
    R.running = 0;
    pthread_cond_broadcast(&R.cv);
    pthread_mutex_unlock(&R.mu);
    pthread_join(R.drain_thread, NULL);
    pthread_cond_destroy(&R.cv);
    pthread_mutex_destroy(&R.mu);
    PyErr_SetString(PyExc_RuntimeError,
                    "damacy_log_add_callback: callback table full");
    goto Fail;
  }

  R.installed = 1;
  return 0;

Fail:
  Py_CLEAR(R.logger);
  return -1;
}

void
log_sink_shutdown(void)
{
  if (!R.installed)
    return;

  // damacy_log has no remove_callback today; the callback continues to
  // exist as a function pointer, but R.installed=0 prevents re-entry on
  // shutdown paths and the thread is gone so nothing reads R.* anyway.
  pthread_mutex_lock(&R.mu);
  R.running = 0;
  pthread_cond_broadcast(&R.cv);
  pthread_mutex_unlock(&R.mu);

  // Release the GIL while joining; the drain thread needs it to deliver
  // any final records and then return.
  Py_BEGIN_ALLOW_THREADS pthread_join(R.drain_thread, NULL);
  Py_END_ALLOW_THREADS

    pthread_cond_destroy(&R.cv);
  pthread_mutex_destroy(&R.mu);
  Py_CLEAR(R.logger);
  R.installed = 0;
}
