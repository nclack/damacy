// damacy._native — Python C extension wrapping the damacy public C API.
//
// Module shape (built up in stages):
//   __version__              — string constant
//   set_log_level(level)     — forward to damacy_log_set_level
//   set_log_quiet(quiet)     — forward to damacy_log_set_quiet
//
// At import: register the log-ring callback (see _log_sink.c) and start
// the drain thread. At interpreter finalize: stop+join the drain thread
// and tear down the ring (we keep the callback registered — the C
// library is process-global and may be loaded again on re-import).

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cuda.h>

#include "damacy.h"
#include "damacy_limits.h"
#include "damacy_log.h"
#include "log/log.h"

#include "_api.h"
#include "_log_sink.h"

#include <pthread.h>

static PyObject*
py_set_log_level(PyObject* self, PyObject* arg)
{
  (void)self;
  long lvl = PyLong_AsLong(arg);
  if (lvl == -1 && PyErr_Occurred())
    return NULL;
  if (lvl < DAMACY_LOG_TRACE || lvl > DAMACY_LOG_FATAL) {
    PyErr_SetString(PyExc_ValueError,
                    "log level must be in [TRACE..FATAL] (0..5)");
    return NULL;
  }
  damacy_log_set_level((damacy_log_level)lvl);
  Py_RETURN_NONE;
}

static PyObject*
py_set_log_quiet(PyObject* self, PyObject* arg)
{
  (void)self;
  int quiet = PyObject_IsTrue(arg);
  if (quiet < 0)
    return NULL;
  damacy_log_set_quiet(quiet);
  Py_RETURN_NONE;
}

// _log_emit(level, msg) — debug helper. Emits a log line through the C
// dispatcher from the calling thread (which still holds the GIL). Useful
// for smoke-testing the routing chain.
static PyObject*
py_log_emit(PyObject* self, PyObject* args)
{
  (void)self;
  int level;
  const char* msg;
  if (!PyArg_ParseTuple(args, "is", &level, &msg))
    return NULL;
  log_log(level, "<py>", 0, "%s", msg);
  Py_RETURN_NONE;
}

struct emit_args
{
  int level;
  char msg[256];
};

static void*
emit_thread_main(void* arg)
{
  struct emit_args* a = arg;
  log_log(a->level, "<py-thread>", 0, "%s", a->msg);
  return NULL;
}

// _log_emit_from_thread(level, msg) — spawn a detached C thread (no GIL
// held by the time it runs log_log) that emits one log line and exits.
// Joins before returning so a test can wait for delivery via the drain.
static PyObject*
py_log_emit_from_thread(PyObject* self, PyObject* args)
{
  (void)self;
  int level;
  const char* msg;
  if (!PyArg_ParseTuple(args, "is", &level, &msg))
    return NULL;

  struct emit_args* a = PyMem_Malloc(sizeof *a);
  if (!a) {
    PyErr_NoMemory();
    return NULL;
  }
  a->level = level;
  size_t n = strnlen(msg, sizeof a->msg - 1);
  memcpy(a->msg, msg, n);
  a->msg[n] = '\0';

  pthread_t tid;
  int ok;
  Py_BEGIN_ALLOW_THREADS ok = pthread_create(&tid, NULL, emit_thread_main, a);
  if (ok == 0)
    pthread_join(tid, NULL);
  Py_END_ALLOW_THREADS

    PyMem_Free(a);
  if (ok != 0) {
    PyErr_SetString(PyExc_RuntimeError, "pthread_create failed");
    return NULL;
  }
  Py_RETURN_NONE;
}

// cuda_init_primary(device=0) — make device's primary CUcontext current
// on the calling thread. damacy_create requires a current CUcontext;
// PyTorch sets one up implicitly, but bare-Python callers (and pytest)
// have to do it themselves. Mirrors tests/cuda_init.h on the C side.
static PyObject*
py_cuda_init_primary(PyObject* self, PyObject* args, PyObject* kw)
{
  (void)self;
  static char* kws[] = { "device", NULL };
  int device = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|i", kws, &device))
    return NULL;

  CUresult r;
  if ((r = cuInit(0)) != CUDA_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "cuInit failed (%d)", (int)r);
    return NULL;
  }
  CUdevice dev = 0;
  if ((r = cuDeviceGet(&dev, device)) != CUDA_SUCCESS) {
    PyErr_Format(
      PyExc_RuntimeError, "cuDeviceGet(%d) failed (%d)", device, (int)r);
    return NULL;
  }
  CUcontext ctx = NULL;
  if ((r = cuDevicePrimaryCtxRetain(&ctx, dev)) != CUDA_SUCCESS) {
    PyErr_Format(
      PyExc_RuntimeError, "cuDevicePrimaryCtxRetain failed (%d)", (int)r);
    return NULL;
  }
  if ((r = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "cuCtxSetCurrent failed (%d)", (int)r);
    return NULL;
  }
  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
  { "set_log_level",
    py_set_log_level,
    METH_O,
    "Set the threshold for the C-side stderr sink (0=TRACE..5=FATAL)." },
  { "set_log_quiet",
    py_set_log_quiet,
    METH_O,
    "Toggle the C-side stderr sink. Python sink (logging.getLogger(\"damacy\"))"
    " keeps firing regardless." },
  { "_log_emit",
    py_log_emit,
    METH_VARARGS,
    "Internal: emit a log line through the C dispatcher from the calling "
    "thread." },
  { "_log_emit_from_thread",
    py_log_emit_from_thread,
    METH_VARARGS,
    "Internal: emit a log line from a freshly-spawned C thread "
    "(reproduces the GIL-deadlock scenario the ring-buffered sink avoids)." },
  { "cuda_init_primary",
    (PyCFunction)(void (*)(void))py_cuda_init_primary,
    METH_VARARGS | METH_KEYWORDS,
    "cuda_init_primary(device=0): make device's primary CUcontext current "
    "on the calling thread. Required before damacy._native.Pipeline(...) when "
    "the caller (e.g. pytest) hasn't already set up a context." },
  { NULL, NULL, 0, NULL },
};

static int
module_exec(PyObject* m)
{
  if (PyModule_AddStringConstant(m, "__version__", "0.0.1") < 0)
    return -1;

  // Log-level constants mirror damacy_log.h.
  static const struct
  {
    const char* name;
    int value;
  } levels[] = {
    { "LOG_TRACE", DAMACY_LOG_TRACE }, { "LOG_DEBUG", DAMACY_LOG_DEBUG },
    { "LOG_INFO", DAMACY_LOG_INFO },   { "LOG_WARN", DAMACY_LOG_WARN },
    { "LOG_ERROR", DAMACY_LOG_ERROR }, { "LOG_FATAL", DAMACY_LOG_FATAL },
  };
  for (size_t i = 0; i < sizeof levels / sizeof levels[0]; ++i) {
    if (PyModule_AddIntConstant(m, levels[i].name, levels[i].value) < 0)
      return -1;
  }

  // Compile-time ceilings the test suite (and callers) compare against
  // when sizing damacy_config.max_chunk_uncompressed_bytes.
  if (PyModule_AddIntConstant(m,
                              "MAX_CHUNK_UNCOMPRESSED_BYTES",
                              (long long)DAMACY_MAX_CHUNK_UNCOMPRESSED_BYTES) <
      0)
    return -1;

  // Quiet the default C stderr sink — Python users want logging.* to be
  // authoritative. Callers can flip this back with set_log_quiet(False).
  damacy_log_set_quiet(1);

  if (log_sink_install() != 0) {
    PyErr_SetString(PyExc_RuntimeError, "failed to install damacy log sink");
    return -1;
  }
  if (api_register_types(m) != 0)
    return -1;
  return 0;
}

static void
module_free(void* m)
{
  (void)m;
  log_sink_shutdown();
}

// PyModuleDef_Slot stores function pointers in a void* field; the cast
// is idiomatic CPython but trips -Wpedantic.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif
static PyModuleDef_Slot slots[] = {
  { Py_mod_exec, (void*)module_exec },
  { 0, NULL },
};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  .m_name = "damacy._native",
  .m_doc = "damacy native extension",
  .m_size = 0,
  .m_methods = methods,
  .m_slots = slots,
  .m_free = module_free,
};

PyMODINIT_FUNC
PyInit__native(void)
{
  return PyModuleDef_Init(&moduledef);
}
