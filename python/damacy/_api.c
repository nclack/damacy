// Pipeline / Batch python bindings around the public C API.
//
// Pipeline(...)   → damacy_create   p.push(s)  → damacy_push
// p.pop()         → damacy_pop
// p.stats()       → damacy_stats_get → dict
// batch.release() → damacy_release (tp_dealloc auto-releases)
// batch.info      → dict snapshot of damacy_batch_info
// batch.__dlpack__ → DLPack v0 (default) or v1 capsule, dispatched by
//                     the consumer's `max_version` kwarg.
//
// Long-running C calls (push/pop/destroy/release) drop the GIL.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "_api.h"
#include "damacy.h"

#ifdef DAMACY_HAS_CUDA
#include <cuda.h>
#endif
#include <math.h>
#include <stdint.h>
#include <string.h>

// ---------- DLPack — minimal definitions, no external dep. ----------------
// https://dmlc.github.io/dlpack/latest/c_api.html
//
// v0 layout (DLManagedTensor, capsule "dltensor") is what PyTorch 2.8
// consumes today. v1.0 layout (DLManagedTensorVersioned, capsule
// "dltensor_versioned") is what CuPy / array-API future consumers ask
// for via max_version=(1,0).

typedef enum
{
  kDLCPU = 1,
  kDLCUDA = 2,
} DLDeviceType;

typedef enum
{
  kDLInt = 0u,
  kDLUInt = 1u,
  kDLFloat = 2u,
  kDLBfloat = 4u,
} DLDataTypeCode;

typedef struct
{
  int32_t device_type;
  int32_t device_id;
} DLDevice;

typedef struct
{
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} DLDataType;

typedef struct
{
  void* data;
  DLDevice device;
  int32_t ndim;
  DLDataType dtype;
  int64_t* shape;
  int64_t* strides;
  uint64_t byte_offset;
} DLTensor;

typedef struct
{
  uint32_t major;
  uint32_t minor;
} DLPackVersion;

// v0
struct DLManagedTensor;
typedef void (*DLManagedTensorDeleterV0)(struct DLManagedTensor* self);
typedef struct DLManagedTensor
{
  DLTensor dl_tensor;
  void* manager_ctx;
  DLManagedTensorDeleterV0 deleter;
} DLManagedTensor;

// v1.0
struct DLManagedTensorVersioned;
typedef void (*DLManagedTensorDeleter)(struct DLManagedTensorVersioned* self);

typedef struct DLManagedTensorVersioned
{
  DLPackVersion version;
  void* manager_ctx;
  DLManagedTensorDeleter deleter;
  uint64_t flags;
  DLTensor dl_tensor;
} DLManagedTensorVersioned;

// ---------- helpers ----------

// Macros below contain `return` — kept UPPERCASE so the control-flow is
// obvious at the call site.

// Raise RuntimeError and return NULL if a Pipeline/Batch handle is dead.
#define RETURN_IF_DESTROYED(obj, msg)                                          \
  do {                                                                         \
    if (!(obj)->handle) {                                                      \
      PyErr_SetString(PyExc_RuntimeError, (msg));                              \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

// Drop the GIL around a single statement. Py_BEGIN_/END_ALLOW_THREADS
// expand to brace tokens, so `expr` lands inside their implicit block.
#define WITH_GIL_RELEASED(expr)                                                \
  do {                                                                         \
    Py_BEGIN_ALLOW_THREADS expr;                                               \
    Py_END_ALLOW_THREADS                                                       \
  } while (0)

// PyType_Ready + Py_INCREF + AddObject with rollback. Returns -1 on
// failure, falling out of the enclosing PyInit_*-style function.
#define ADD_TYPE(m, name, type)                                                \
  do {                                                                         \
    if (PyType_Ready(&(type)) < 0)                                             \
      return -1;                                                               \
    Py_INCREF(&(type));                                                        \
    if (PyModule_AddObject((m), (name), (PyObject*)&(type)) < 0) {             \
      Py_DECREF(&(type));                                                      \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static int
parse_dtype(PyObject* obj, enum damacy_dtype* out)
{
  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    if (v < 0 || v > DAMACY_BF16) {
      PyErr_SetString(PyExc_ValueError, "dtype out of range");
      return -1;
    }
    *out = (enum damacy_dtype)v;
    return 0;
  }
  const char* s = PyUnicode_AsUTF8(obj);
  if (!s)
    return -1;
  if (!strcmp(s, "f32") || !strcmp(s, "float32")) {
    *out = DAMACY_F32;
    return 0;
  }
  if (!strcmp(s, "bf16") || !strcmp(s, "bfloat16")) {
    *out = DAMACY_BF16;
    return 0;
  }
  PyErr_Format(PyExc_ValueError, "unknown dtype: %s", s);
  return -1;
}

static const char*
dtype_name(enum damacy_dtype d)
{
  switch (d) {
    case DAMACY_F32:
      return "f32";
    case DAMACY_BF16:
      return "bf16";
  }
  return "?";
}

// Module-owned exception type. Subclasses RuntimeError so legacy callers
// catching RuntimeError still work; the Python wrapper layer remaps to
// per-status subclasses keyed by .status.
PyObject* DamacyError = NULL;

// Raise a status-tagged DamacyError. Sets .status (int) and .what (str)
// attributes on the instance. Returns NULL so callers can `return
// raise_status(...)`.
static PyObject*
raise_status(enum damacy_status s, const char* what)
{
  PyObject* msg =
    PyUnicode_FromFormat("damacy: %s failed (%s)", what, damacy_status_str(s));
  if (!msg)
    return NULL;
  PyObject* exc = PyObject_CallFunction(DamacyError, "O", msg);
  Py_DECREF(msg);
  if (!exc)
    return NULL;
  PyObject* status = PyLong_FromLong((long)s);
  PyObject* what_obj = PyUnicode_FromString(what);
  if (!status || !what_obj) {
    Py_XDECREF(status);
    Py_XDECREF(what_obj);
    Py_DECREF(exc);
    return NULL;
  }
  if (PyObject_SetAttrString(exc, "status", status) < 0 ||
      PyObject_SetAttrString(exc, "what", what_obj) < 0) {
    Py_DECREF(status);
    Py_DECREF(what_obj);
    Py_DECREF(exc);
    return NULL;
  }
  Py_DECREF(status);
  Py_DECREF(what_obj);
  PyErr_SetObject(DamacyError, exc);
  Py_DECREF(exc);
  return NULL;
}

// ---------- forward decls (Batch type below) ----------

typedef struct
{
  PyObject_HEAD struct damacy* handle;
  PyObject* dependencies;
} PipelineObj;

typedef struct
{
  PyObject_HEAD PipelineObj*
    parent;                    // owns the lifetime of the underlying damacy*
  struct damacy_batch* handle; // NULL once released
} BatchObj;

extern PyTypeObject PipelineType;
extern PyTypeObject BatchType;

// ---------- Batch ----------

// Idempotent: clears self->handle then releases under no-GIL.
static void
batch_do_release(BatchObj* self)
{
  if (self->handle && self->parent && self->parent->handle) {
    struct damacy_batch* b = self->handle;
    self->handle = NULL;
    WITH_GIL_RELEASED(damacy_release(self->parent->handle, b));
  }
}

// Deferred-release variant: damacy waits on `event` on its internal
// stream_post before reusing the slot's buffer. Returns 0 on success or
// -1 with a Python exception set on failure. self->handle is cleared
// either way, so re-entry is a no-op (matches batch_do_release).
static int
batch_do_release_event(BatchObj* self, void* event)
{
  if (!self->handle || !self->parent || !self->parent->handle)
    return 0;
  struct damacy* d = self->parent->handle;
  struct damacy_batch* b = self->handle;
  self->handle = NULL;
  enum damacy_status s;
  WITH_GIL_RELEASED(s = damacy_release_event(d, b, event));
  if (s != DAMACY_OK) {
    raise_status(s, "release");
    return -1;
  }
  return 0;
}

static PyObject*
Batch_release(BatchObj* self, PyObject* args, PyObject* kw)
{
  static char* kws[] = { "event", NULL };
  PyObject* event_obj = Py_None;
  if (!PyArg_ParseTupleAndKeywords(args, kw, "|O", kws, &event_obj))
    return NULL;
  if (event_obj == Py_None) {
    batch_do_release(self);
    Py_RETURN_NONE;
  }
  // Accept an integer CUevent handle (cast from cudaEvent_t /
  // torch.cuda.Event.cuda_event). The Python wrapper resolves higher-level
  // objects to this int before calling.
  unsigned long long ev = PyLong_AsUnsignedLongLong(event_obj);
  if (ev == (unsigned long long)-1 && PyErr_Occurred()) {
    PyErr_SetString(PyExc_TypeError,
                    "Batch.release: event must be an integer CUevent handle");
    return NULL;
  }
  if (batch_do_release_event(self, (void*)(uintptr_t)ev) != 0)
    return NULL;
  Py_RETURN_NONE;
}

static void
Batch_dealloc(BatchObj* self)
{
  batch_do_release(self);
  Py_XDECREF(self->parent);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject*
Batch_info(BatchObj* self, void* Py_UNUSED(closure))
{
  RETURN_IF_DESTROYED(self, "Batch has been released");
  struct damacy_batch_info info;
  damacy_batch_info(self->handle, &info);

  PyObject* shape = PyTuple_New(info.rank);
  if (!shape)
    return NULL;
  for (int i = 0; i < info.rank; ++i) {
    PyObject* v = PyLong_FromLongLong((long long)info.shape[i]);
    if (!v) {
      Py_DECREF(shape);
      return NULL;
    }
    PyTuple_SET_ITEM(shape, i, v);
  }

  return Py_BuildValue("{s:K,s:K,s:i,s:i,s:N,s:s,s:K,s:K}",
                       "data",
                       (unsigned long long)(uintptr_t)info.data,
                       "device_ptr",
                       (unsigned long long)(uintptr_t)info.data,
                       "device_type",
                       (int)info.device_type,
                       "device_id",
                       info.device_id,
                       "shape",
                       shape,
                       "dtype",
                       dtype_name(info.dtype),
                       "ready_stream",
                       (unsigned long long)(uintptr_t)info.ready_stream,
                       "batch_id",
                       (unsigned long long)info.batch_id);
}

// ---------- DLPack export ----------

// dtype_to_dl maps damacy_dtype → (DLDataType, bytes-per-element).
static int
dtype_to_dl(enum damacy_dtype dt, DLDataType* out_dt, uint8_t* out_bpe)
{
  switch (dt) {
    case DAMACY_F32:
      *out_dt = (DLDataType){ .code = kDLFloat, .bits = 32, .lanes = 1 };
      *out_bpe = 4;
      return 0;
    case DAMACY_BF16:
      *out_dt = (DLDataType){ .code = kDLBfloat, .bits = 16, .lanes = 1 };
      *out_bpe = 2;
      return 0;
  }
  return -1;
}

// Storage layout for one exported tensor. Holds both v0 and v1 managed
// headers — only one is wired up per export, picked by max_version. The
// capsule's manager_ctx points back to the payload so the deleter can
// drop the batch ref and free both at once.
struct dlpack_payload
{
  DLManagedTensor mt_v0;
  DLManagedTensorVersioned mt_v1;
  int64_t shape[DAMACY_MAX_RANK + 1]; // referenced by dl_tensor.shape
  struct damacy_batch* handle;
};

static void
dlpack_payload_free(struct dlpack_payload* p)
{
  if (!p)
    return;
  damacy_batch_release(p->handle);
  free(p);
}

static void
dlpack_deleter_v0(DLManagedTensor* self)
{
  if (!self)
    return;
  dlpack_payload_free((struct dlpack_payload*)self->manager_ctx);
}

static void
dlpack_deleter_v1(DLManagedTensorVersioned* self)
{
  if (!self)
    return;
  dlpack_payload_free((struct dlpack_payload*)self->manager_ctx);
}

static void
dlpack_capsule_destructor(PyObject* capsule)
{
  // Only fires when the capsule was never consumed; otherwise the
  // consumer renames to "used_dltensor" / "used_dltensor_versioned"
  // and runs the deleter itself.
  const char* name = PyCapsule_GetName(capsule);
  if (!name)
    return;
  if (strcmp(name, "dltensor") == 0) {
    DLManagedTensor* mt = PyCapsule_GetPointer(capsule, name);
    if (mt && mt->deleter)
      mt->deleter(mt);
  } else if (strcmp(name, "dltensor_versioned") == 0) {
    DLManagedTensorVersioned* mt = PyCapsule_GetPointer(capsule, name);
    if (mt && mt->deleter)
      mt->deleter(mt);
  }
}

#ifdef DAMACY_HAS_CUDA
static int
sync_streams_for_consumer(void* producer_stream_v, PyObject* stream_obj)
{
  // stream_obj per DLPack semantics:
  //   None  → producer must synchronize (block here)
  //   -1    → consumer asks for no sync (we still record nothing)
  //   1     → legacy default stream
  //   2     → per-thread default stream
  //   other → integer cuStream handle
  CUstream consumer = NULL;
  int sync_now = 0;
  int skip_sync = 0;
  if (!stream_obj || stream_obj == Py_None) {
    sync_now = 1;
  } else {
    long long s = PyLong_AsLongLong(stream_obj);
    if (s == -1 && PyErr_Occurred())
      return -1;
    if (s == -1) {
      skip_sync = 1;
    } else if (s == 0) {
      // 0 isn't strictly defined in DLPack but PyTorch sometimes passes
      // it; treat as legacy default.
      consumer = (CUstream)0;
    } else if (s == 1) {
      // DLPack sentinel 1 is the runtime API's cudaStreamLegacy. The
      // driver API's equivalent is the default stream NULL — passing
      // 0x1 to cuStreamWaitEvent fails with INVALID_VALUE.
      consumer = (CUstream)NULL;
    } else if (s == 2) {
      // DLPack sentinel 2 is the runtime API's cudaStreamPerThread.
      // There's no driver-API handle for it, so fall back to a host
      // sync on the producer (safe, heavier than an event wait).
      sync_now = 1;
    } else {
      consumer = (CUstream)(uintptr_t)s;
    }
  }

  if (skip_sync)
    return 0;

  CUstream producer = (CUstream)producer_stream_v;
  if (sync_now) {
    if (cuStreamSynchronize(producer) != CUDA_SUCCESS) {
      PyErr_SetString(PyExc_RuntimeError, "cuStreamSynchronize failed");
      return -1;
    }
    return 0;
  }
  if (consumer == producer)
    return 0; // same stream, ordered already.

  CUevent ev;
  if (cuEventCreate(&ev, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, "cuEventCreate failed");
    return -1;
  }
  if (cuEventRecord(ev, producer) != CUDA_SUCCESS) {
    cuEventDestroy(ev);
    PyErr_SetString(PyExc_RuntimeError, "cuEventRecord failed");
    return -1;
  }
  if (cuStreamWaitEvent(consumer, ev, 0) != CUDA_SUCCESS) {
    cuEventDestroy(ev);
    PyErr_SetString(PyExc_RuntimeError, "cuStreamWaitEvent failed");
    return -1;
  }
  cuEventDestroy(ev);
  return 0;
}

#endif

// Parse max_version per the array-API DLPack protocol. Returns 0 on
// success, -1 on error (PyErr set). *out_major / *out_minor land at the
// requested version, or (0,0) when the consumer didn't specify one.
//
// Accepts None, a 2-tuple (major, minor), or any 2-element sequence;
// PyTorch passes None today, CuPy and NumPy pass (1, 0).
static int
parse_max_version(PyObject* obj, uint32_t* out_major, uint32_t* out_minor)
{
  *out_major = 0;
  *out_minor = 0;
  if (!obj || obj == Py_None)
    return 0;
  PyObject* seq = PySequence_Fast(obj,
                                  "max_version must be a (major, minor) "
                                  "tuple or None");
  if (!seq)
    return -1;
  Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
  if (n != 2) {
    Py_DECREF(seq);
    PyErr_SetString(PyExc_TypeError,
                    "max_version must be a 2-element sequence (major, minor)");
    return -1;
  }
  // Check each conversion individually — a live PyErr from the first
  // call must not leak into the second.
  long mj = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, 0));
  if (mj == -1 && PyErr_Occurred()) {
    Py_DECREF(seq);
    return -1;
  }
  long mn = PyLong_AsLong(PySequence_Fast_GET_ITEM(seq, 1));
  if (mn == -1 && PyErr_Occurred()) {
    Py_DECREF(seq);
    return -1;
  }
  Py_DECREF(seq);
  if (mj < 0 || mn < 0) {
    PyErr_SetString(PyExc_ValueError, "max_version components must be >= 0");
    return -1;
  }
  *out_major = (uint32_t)mj;
  *out_minor = (uint32_t)mn;
  return 0;
}

static PyObject*
Batch_dlpack(BatchObj* self, PyObject* args, PyObject* kw)
{
  static char* kws[] = { "stream", "max_version", "dl_device", "copy", NULL };
  PyObject* stream_obj = Py_None;
  PyObject* max_version = Py_None;
  PyObject* dl_device = Py_None;
  PyObject* copy = Py_None;
  if (!PyArg_ParseTupleAndKeywords(
        args, kw, "|OOOO", kws, &stream_obj, &max_version, &dl_device, &copy))
    return NULL;
  if (copy != Py_None && PyObject_IsTrue(copy)) {
    PyErr_SetString(PyExc_BufferError,
                    "damacy DLPack: copy=True not supported");
    return NULL;
  }
  RETURN_IF_DESTROYED(self, "Batch has been released");

  // Decide the wire format. Per the array-API DLPack protocol:
  //   - max_version is None      → emit v0 capsule (legacy, what
  //                                 PyTorch 2.8 consumes today).
  //   - max_version.major >= 1   → emit v1.0 capsule.
  //   - max_version.major == 0   → emit v0 capsule.
  uint32_t want_major = 0, want_minor = 0;
  if (parse_max_version(max_version, &want_major, &want_minor) != 0)
    return NULL;
  const int emit_versioned = (max_version != Py_None) && (want_major >= 1);

  struct damacy_batch_info info;
  damacy_batch_info(self->handle, &info);

  DLDataType dlt;
  uint8_t bpe;
  if (dtype_to_dl(info.dtype, &dlt, &bpe) != 0) {
    PyErr_SetString(PyExc_RuntimeError, "unsupported dtype for DLPack");
    return NULL;
  }
  (void)bpe;

  if (dl_device != Py_None) {
    int requested_type, requested_id;
    if (!PyArg_ParseTuple(dl_device, "ii", &requested_type, &requested_id))
      return NULL;
    if (requested_type != (int)info.device_type ||
        requested_id != info.device_id) {
      PyErr_SetString(PyExc_BufferError,
                      "requested device differs from the batch device");
      return NULL;
    }
  }
  if (info.device_type == DAMACY_DEVICE_CPU) {
    if (stream_obj != Py_None) {
      PyErr_SetString(PyExc_ValueError, "CPU tensors require stream=None");
      return NULL;
    }
  }
#ifdef DAMACY_HAS_CUDA
  else if (info.ready_stream &&
           sync_streams_for_consumer(info.ready_stream, stream_obj) != 0)
    return NULL;
#endif

  struct dlpack_payload* p = calloc(1, sizeof *p);
  if (!p)
    return PyErr_NoMemory();

  for (int i = 0; i < info.rank; ++i)
    p->shape[i] = info.shape[i];

  p->handle = self->handle;
  damacy_batch_retain(p->handle);

  DLTensor dl = {
    .data = info.data,
    .device = (DLDevice){ .device_type = info.device_type,
                          .device_id = info.device_id },
    .ndim = (int32_t)info.rank,
    .dtype = dlt,
    .shape = p->shape,
    .strides = NULL, // contiguous
    .byte_offset = 0,
  };

  PyObject* cap = NULL;
  if (emit_versioned) {
    p->mt_v1.version = (DLPackVersion){ 1, 0 };
    p->mt_v1.manager_ctx = p;
    p->mt_v1.deleter = dlpack_deleter_v1;
    p->mt_v1.flags = 0;
    p->mt_v1.dl_tensor = dl;
    cap =
      PyCapsule_New(&p->mt_v1, "dltensor_versioned", dlpack_capsule_destructor);
  } else {
    p->mt_v0.dl_tensor = dl;
    p->mt_v0.manager_ctx = p;
    p->mt_v0.deleter = dlpack_deleter_v0;
    cap = PyCapsule_New(&p->mt_v0, "dltensor", dlpack_capsule_destructor);
  }
  if (!cap) {
    dlpack_payload_free(p);
    return NULL;
  }
  return cap;
}

static PyObject*
Batch_dlpack_device(BatchObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Batch has been released");
  struct damacy_batch_info info;
  damacy_batch_info(self->handle, &info);
  return Py_BuildValue("(ii)", (int)info.device_type, info.device_id);
}

static PyMethodDef Batch_methods[] = {
  { "release",
    (PyCFunction)(void (*)(void))Batch_release,
    METH_VARARGS | METH_KEYWORDS,
    "release(event=None): release this batch reference. DLPack consumers "
    "keep the buffer alive. A CUDA event delays reuse until that event "
    "completes. Idempotent." },
  { "__dlpack__",
    (PyCFunction)(void (*)(void))Batch_dlpack,
    METH_VARARGS | METH_KEYWORDS,
    "DLPack capsule export. Default (max_version=None) emits a v0 "
    "\"dltensor\" capsule for PyTorch compatibility. max_version=(1,0) "
    "or higher emits a v1.0 \"dltensor_versioned\" capsule. Honors "
    "stream=... per the DLPack protocol." },
  { "__dlpack_device__",
    (PyCFunction)Batch_dlpack_device,
    METH_NOARGS,
    "DLPack device tuple: (1, 0) for CPU or (2, ordinal) for CUDA." },
  { NULL, NULL, 0, NULL },
};

static PyGetSetDef Batch_getset[] = {
  { "info",
    (getter)Batch_info,
    NULL,
    "Snapshot of damacy_batch_info as a dict.",
    NULL },
  { NULL, NULL, NULL, NULL, NULL },
};

PyTypeObject BatchType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "damacy._native.Batch",
  .tp_basicsize = sizeof(BatchObj),
  .tp_flags = Py_TPFLAGS_DEFAULT,
  .tp_doc =
    "On-device batch handle from damacy.pop(). Call release() when done.",
  .tp_dealloc = (destructor)Batch_dealloc,
  .tp_methods = Batch_methods,
  .tp_getset = Batch_getset,
};

static BatchObj*
batch_new(PipelineObj* parent, struct damacy_batch* handle)
{
  BatchObj* b = PyObject_New(BatchObj, &BatchType);
  if (!b)
    return NULL;
  Py_INCREF(parent);
  b->parent = parent;
  b->handle = handle;
  return b;
}

// ---------- Pipeline ----------

static int
Pipeline_init(PipelineObj* self, PyObject* args, PyObject* kw)
{
  if (self->handle) {
    PyErr_SetString(PyExc_RuntimeError, "Pipeline is already initialized");
    return -1;
  }
  // kws[] / format string / variable list mirror struct damacy_config —
  // keep all three in sync when adding a field.
  static char* kws[] = { "samples_per_batch",
                         "lookahead_samples",
                         "dtype",
                         "max_chunk_uncompressed_bytes",
                         "max_gpu_memory_bytes",
                         "n_io_threads",
                         "metadata_io_concurrency",
                         "n_array_meta_cache",
                         "n_shard_index_cache",
                         "n_chunk_layout_cache",
                         "max_shards_per_sample",
                         "sample_shape",
                         "host_buffer_waves",
                         "max_chunks_per_wave",
                         "max_substreams_per_chunk",
                         "max_read_op_bytes",
                         "device",
                         "enable_gds",
                         "numa_strategy",
                         "numa_node",
                         "bypass_decode",
                         "metadata_latency_baseline_ns",
                         "metadata_latency_lognormal_mu_ln_ns",
                         "metadata_latency_lognormal_sigma_ln_ns",
                         "metadata_latency_cap_ns",
                         "metadata_latency_seed",
                         "max_index_bytes",
                         NULL };
  struct damacy_tuning td = damacy_tuning_defaults();
  unsigned int samples_per_batch = 0;
  unsigned int lookahead = 2;
  PyObject* dtype_obj = NULL;
  unsigned int max_chunk_uncompressed = td.max_chunk_uncompressed_bytes;
  unsigned long long max_gpu_bytes = 0;
  unsigned long long max_index_bytes = td.max_index_bytes;
  unsigned int n_io = td.n_io_threads;
  unsigned int metadata_io_concurrency = td.metadata_io_concurrency;
  unsigned int n_array_meta = DAMACY_DEFAULT_ARRAY_META_CACHE;
  unsigned int n_shard_index = DAMACY_DEFAULT_SHARD_INDEX_CACHE;
  unsigned int n_chunk_layout = DAMACY_DEFAULT_CHUNK_LAYOUT_CACHE;
  unsigned int max_shards_per_sample = DAMACY_DEFAULT_MAX_SHARDS_PER_SAMPLE;
  PyObject* sample_shape_obj = NULL;
  unsigned int host_buffer_waves = DAMACY_DEFAULT_HOST_BUFFER_WAVES;
  unsigned int max_chunks_per_wave = DAMACY_DEFAULT_MAX_CHUNKS_PER_WAVE;
  unsigned int max_substreams_per_chunk =
    DAMACY_DEFAULT_MAX_SUBSTREAMS_PER_CHUNK;
  unsigned long long max_read_op_bytes = td.max_read_op_bytes;
  int device = -1;
  int enable_gds = DAMACY_GDS_AUTO;
  int numa_strategy = DAMACY_NUMA_AUTO;
  int numa_node = -1;
  int bypass_decode = 0;
  unsigned long long metadata_latency_baseline_ns = 0;
  double metadata_latency_lognormal_mu_ln_ns = 0.0;
  double metadata_latency_lognormal_sigma_ln_ns = 0.0;
  unsigned long long metadata_latency_cap_ns = 0;
  unsigned long long metadata_latency_seed = 0;
  if (!PyArg_ParseTupleAndKeywords(args,
                                   kw,
                                   "IIOIKIIIIIIO|IIIKiiiipKddKKK",
                                   kws,
                                   &samples_per_batch,
                                   &lookahead,
                                   &dtype_obj,
                                   &max_chunk_uncompressed,
                                   &max_gpu_bytes,
                                   &n_io,
                                   &metadata_io_concurrency,
                                   &n_array_meta,
                                   &n_shard_index,
                                   &n_chunk_layout,
                                   &max_shards_per_sample,
                                   &sample_shape_obj,
                                   &host_buffer_waves,
                                   &max_chunks_per_wave,
                                   &max_substreams_per_chunk,
                                   &max_read_op_bytes,
                                   &device,
                                   &enable_gds,
                                   &numa_strategy,
                                   &numa_node,
                                   &bypass_decode,
                                   &metadata_latency_baseline_ns,
                                   &metadata_latency_lognormal_mu_ln_ns,
                                   &metadata_latency_lognormal_sigma_ln_ns,
                                   &metadata_latency_cap_ns,
                                   &metadata_latency_seed,
                                   &max_index_bytes))
    return -1;

  if (enable_gds != DAMACY_GDS_AUTO && enable_gds != DAMACY_GDS_ON &&
      enable_gds != DAMACY_GDS_OFF) {
    PyErr_Format(
      PyExc_ValueError, "enable_gds out of range (got %d)", enable_gds);
    return -1;
  }
  if (numa_strategy != DAMACY_NUMA_AUTO &&
      numa_strategy != DAMACY_NUMA_DISABLED &&
      numa_strategy != DAMACY_NUMA_PIN_TO) {
    PyErr_Format(
      PyExc_ValueError, "numa_strategy out of range (got %d)", numa_strategy);
    return -1;
  }
  if (!isfinite(metadata_latency_lognormal_mu_ln_ns) ||
      !isfinite(metadata_latency_lognormal_sigma_ln_ns) ||
      metadata_latency_lognormal_sigma_ln_ns < 0.0) {
    PyErr_SetString(PyExc_ValueError,
                    "metadata latency lognormal mu/sigma must be finite and "
                    "sigma must be >= 0");
    return -1;
  }

  enum damacy_dtype dt;
  if (parse_dtype(dtype_obj, &dt) != 0)
    return -1;

  struct damacy_config cfg = {
    .samples_per_batch = samples_per_batch,
    .lookahead_samples = lookahead,
    .dtype = dt,
    .device = device,
    .tuning = {
      .max_gpu_memory_bytes = (uint64_t)max_gpu_bytes,
      .max_index_bytes = (uint64_t)max_index_bytes,
      .max_chunk_uncompressed_bytes = max_chunk_uncompressed,
      .max_read_op_bytes = (uint64_t)max_read_op_bytes,
      .host_buffer_waves = (uint8_t)host_buffer_waves,
      .max_chunks_per_wave = (uint32_t)max_chunks_per_wave,
      .max_substreams_per_chunk = (uint32_t)max_substreams_per_chunk,
      .n_io_threads = n_io,
      .metadata_io_concurrency = metadata_io_concurrency,
      .n_array_meta_cache = n_array_meta,
      .n_shard_index_cache = n_shard_index,
      .n_chunk_layout_cache = n_chunk_layout,
      .max_shards_per_sample = max_shards_per_sample,
      .numa_strategy = (enum damacy_numa_strategy)numa_strategy,
      .numa_node = numa_node,
      .enable_gds = (enum damacy_gds_mode)enable_gds,
    },
    .debug = { .bypass_decode = (uint8_t)(bypass_decode ? 1 : 0),
               .metadata_latency = {
                 .baseline_ns = (uint64_t)metadata_latency_baseline_ns,
                 .lognormal_mu_ln_ns =
                   metadata_latency_lognormal_mu_ln_ns,
                 .lognormal_sigma_ln_ns =
                   metadata_latency_lognormal_sigma_ln_ns,
                 .cap_ns = (uint64_t)metadata_latency_cap_ns,
                 .seed = (uint64_t)metadata_latency_seed,
               } },
  };

  // sample_shape: a sequence of ints; copies into cfg.sample_shape[].
  {
    PyObject* seq = PySequence_Fast(sample_shape_obj,
                                    "sample_shape must be a sequence of ints");
    if (!seq)
      return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n <= 0 || n > DAMACY_MAX_RANK) {
      Py_DECREF(seq);
      PyErr_Format(PyExc_ValueError,
                   "sample_shape rank must be in [1, %d] (got %zd)",
                   DAMACY_MAX_RANK,
                   (Py_ssize_t)n);
      return -1;
    }
    for (Py_ssize_t i = 0; i < n; ++i) {
      PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
      long long v = PyLong_AsLongLong(item);
      if (v == -1 && PyErr_Occurred()) {
        Py_DECREF(seq);
        return -1;
      }
      if (v <= 0) {
        Py_DECREF(seq);
        PyErr_Format(
          PyExc_ValueError, "sample_shape[%zd] must be > 0 (got %lld)", i, v);
        return -1;
      }
      cfg.sample_shape[i] = (int64_t)v;
    }
    cfg.sample_rank = (uint8_t)n;
    Py_DECREF(seq);
  }

  struct damacy* d = NULL;
  enum damacy_status s;
  WITH_GIL_RELEASED(s = damacy_create(&cfg, &d));
  if (s != DAMACY_OK) {
    raise_status(s, "create");
    return -1;
  }
  self->handle = d;
  return 0;
}

static void
Pipeline_dealloc(PipelineObj* self)
{
  if (self->handle) {
    struct damacy* d = self->handle;
    self->handle = NULL;
    WITH_GIL_RELEASED(damacy_destroy(d));
  }
  Py_XDECREF(self->dependencies);
  Py_TYPE(self)->tp_free((PyObject*)self);
}

// Parse one Python sample dict into a damacy_sample. The uri string is
// borrowed from the dict's PyUnicode object — caller must keep the dict
// reachable until damacy_push returns (it copies the uri internally).
static int
parse_sample(PyObject* obj, struct damacy_sample* out)
{
  if (!PyDict_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "sample must be a dict");
    return -1;
  }
  PyObject* uri = PyDict_GetItemString(obj, "uri");
  PyObject* axes = PyDict_GetItemString(obj, "axes");
  if (!uri || !axes) {
    PyErr_SetString(PyExc_KeyError, "sample requires 'uri' and 'axes'");
    return -1;
  }
  if (PyDict_Size(obj) != 2) {
    PyErr_SetString(PyExc_ValueError, "sample accepts only 'uri' and 'axes'");
    return -1;
  }
  const char* uri_s = PyUnicode_AsUTF8(uri);
  if (!uri_s)
    return -1;
  if (!PyList_Check(axes) && !PyTuple_Check(axes)) {
    PyErr_SetString(PyExc_TypeError, "axes must be a list or tuple");
    return -1;
  }
  Py_ssize_t n = PySequence_Fast_GET_SIZE(axes);
  if (n < 1 || n > DAMACY_MAX_RANK) {
    PyErr_Format(PyExc_ValueError, "sample rank out of range: %zd", n);
    return -1;
  }

  out->uri = uri_s;
  out->rank = (uint8_t)n;
  for (Py_ssize_t d = 0; d < n; ++d) {
    PyObject* item = PySequence_Fast_GET_ITEM(axes, d);
    if ((!PyList_Check(item) && !PyTuple_Check(item)) ||
        PySequence_Fast_GET_SIZE(item) != 2) {
      PyErr_Format(
        PyExc_ValueError, "axes[%zd] must be a (kind, values) pair", d);
      return -1;
    }
    PyObject* kind = PySequence_Fast_GET_ITEM(item, 0);
    PyObject* values = PySequence_Fast_GET_ITEM(item, 1);
    if (!PyUnicode_Check(kind)) {
      PyErr_Format(PyExc_TypeError, "axes[%zd] kind must be a string", d);
      return -1;
    }
    struct damacy_axis_selection* axis = &out->axes[d];
    if (PyUnicode_CompareWithASCIIString(kind, "interval") == 0) {
      long long beg, end;
      if (!PyArg_ParseTuple(values, "LL", &beg, &end)) {
        PyErr_Format(
          PyExc_ValueError, "axes[%zd] interval must be a (beg, end) pair", d);
        return -1;
      }
      *axis = (struct damacy_axis_selection){
        .kind = DAMACY_AXIS_INTERVAL, .interval = { (int64_t)beg, (int64_t)end }
      };
    } else if (PyUnicode_CompareWithASCIIString(kind, "indices") == 0) {
      PyObject* sequence =
        PySequence_Fast(values, "index array must be iterable");
      if (!sequence)
        return -1;
      Py_ssize_t count = PySequence_Fast_GET_SIZE(sequence);
      if (!count || (uint64_t)count > UINT32_MAX ||
          (size_t)count > SIZE_MAX / sizeof(int64_t)) {
        Py_DECREF(sequence);
        PyErr_SetString(PyExc_ValueError, "index array size is out of range");
        return -1;
      }
      int64_t* copied = PyMem_Malloc((size_t)count * sizeof(*copied));
      if (!copied) {
        Py_DECREF(sequence);
        PyErr_NoMemory();
        return -1;
      }
      *axis = (struct damacy_axis_selection){
        .kind = DAMACY_AXIS_INDICES, .indices = { copied, (uint32_t)count }
      };
      for (Py_ssize_t i = 0; i < count; ++i) {
        copied[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(sequence, i));
        if (PyErr_Occurred()) {
          Py_DECREF(sequence);
          return -1;
        }
      }
      Py_DECREF(sequence);
    } else {
      PyErr_Format(PyExc_ValueError, "axes[%zd] has unknown kind %R", d, kind);
      return -1;
    }
  }
  return 0;
}

static void
free_samples(struct damacy_sample* samples, Py_ssize_t count)
{
  for (Py_ssize_t i = 0; i < count; ++i)
    for (uint8_t d = 0; d < samples[i].rank; ++d)
      if (samples[i].axes[d].kind == DAMACY_AXIS_INDICES)
        PyMem_Free((void*)samples[i].axes[d].indices.values);
  PyMem_Free(samples);
}

static PyObject*
Pipeline_push(PipelineObj* self, PyObject* arg)
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  if (!PyList_Check(arg) && !PyTuple_Check(arg)) {
    PyErr_SetString(PyExc_TypeError,
                    "push() expects a list or tuple of samples");
    return NULL;
  }
  Py_ssize_t n = PySequence_Fast_GET_SIZE(arg);
  if (n == 0)
    return Py_BuildValue("{s:i,s:i}", "consumed", 0, "status", (int)DAMACY_OK);

  struct damacy_sample* buf = PyMem_Calloc((size_t)n, sizeof *buf);
  if (!buf)
    return PyErr_NoMemory();

  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(arg, i);
    if (parse_sample(item, &buf[i]) != 0) {
      free_samples(buf, n);
      return NULL;
    }
  }

  struct damacy_sample_slice slice = { .beg = buf, .end = buf + n };
  struct damacy_push_result r;
  WITH_GIL_RELEASED(r = damacy_push(self->handle, slice));

  Py_ssize_t consumed = (Py_ssize_t)(r.unconsumed.beg - buf);
  free_samples(buf, n);

  // OK / AGAIN are not errors at this layer — return the consumed count
  // and the integer status so the caller can detect back-pressure. Any
  // other status (NOTFOUND/DTYPE/RANK/SHUTDOWN/...) raises.
  if (r.status != DAMACY_OK && r.status != DAMACY_AGAIN)
    return raise_status(r.status, "push");

  return Py_BuildValue(
    "{s:n,s:i}", "consumed", consumed, "status", (int)r.status);
}

static PyObject*
Pipeline_pop(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  struct damacy_batch* b = NULL;
  enum damacy_status s;
  WITH_GIL_RELEASED(s = damacy_pop(self->handle, &b));
  if (s != DAMACY_OK)
    return raise_status(s, "pop");
  BatchObj* batch = batch_new(self, b);
  if (!batch) {
    WITH_GIL_RELEASED(damacy_batch_release(b));
  }
  return (PyObject*)batch;
}

static PyObject*
metric_to_dict(const struct damacy_metric* m)
{
  return Py_BuildValue("{s:s,s:f,s:f,s:d,s:d,s:K}",
                       "name",
                       m->name ? m->name : "",
                       "ms",
                       (double)m->ms,
                       "best_ms",
                       (double)m->best_ms,
                       "input_bytes",
                       m->input_bytes,
                       "output_bytes",
                       m->output_bytes,
                       "count",
                       (unsigned long long)m->count);
}

// Set key→value, stealing the value reference. Returns 0 on success,
// -1 on error (decrements value on failure path).
static int
dict_set_steal(PyObject* d, const char* key, PyObject* value)
{
  if (!value)
    return -1;
  int rc = PyDict_SetItemString(d, key, value);
  Py_DECREF(value);
  return rc;
}

static PyObject*
Pipeline_stats(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  struct damacy_stats st;
  damacy_stats_get(self->handle, &st);

  PyObject* d = PyDict_New();
  if (!d)
    return NULL;

  const struct
  {
    const char* name;
    const struct damacy_metric* m;
  } metrics[] = {
    { "plan", &st.plan },
    { "io", &st.io },
    { "input_transfer", &st.input_transfer },
    { "decode", &st.decode },
    { "post_decode", &st.post_decode },
    { "decode_gap", &st.decode_gap },
    { "assemble", &st.assemble },
    { "bind_wait", &st.bind_wait },
    { "pop_wait", &st.pop_wait },
  };
  for (size_t i = 0; i < sizeof metrics / sizeof metrics[0]; ++i)
    if (dict_set_steal(d, metrics[i].name, metric_to_dict(metrics[i].m)) < 0)
      goto Fail;

  const struct
  {
    const char* name;
    unsigned long long val;
  } counters[] = {
    { "array_meta_hits", st.array_meta.hits },
    { "array_meta_misses", st.array_meta.misses },
    { "shard_index_hits", st.shard_index.hits },
    { "shard_index_misses", st.shard_index.misses },
    { "chunk_layout_hits", st.chunk_layout.hits },
    { "chunk_layout_misses", st.chunk_layout.misses },
    { "metadata_latency_ops", st.metadata_latency.ops },
    { "metadata_latency_stat_ops", st.metadata_latency.stat_ops },
    { "metadata_latency_submit_ops", st.metadata_latency.submit_ops },
    { "metadata_latency_active", st.metadata_latency.active },
    { "metadata_latency_max_active", st.metadata_latency.max_active },
    { "metadata_latency_total_sleep_ns", st.metadata_latency.total_sleep_ns },
    { "metadata_latency_max_sleep_ns", st.metadata_latency.max_sleep_ns },
    { "metadata_backend_read_jobs", st.metadata_backend.read_jobs },
    { "metadata_backend_read_active", st.metadata_backend.read_active },
    { "metadata_backend_read_max_active", st.metadata_backend.read_max_active },
    { "batches_emitted", st.batches_emitted },
    { "waves_emitted", st.waves_emitted },
    { "worker_steps", st.worker_steps },
    { "chunks_planned", st.chunks_planned },
    { "chunks_to_load", st.chunks_to_load },
    { "chunks_dispatched", st.chunks_dispatched },
    { "reads_issued", st.reads_issued },
    { "gpu_bytes_committed", st.gpu_bytes_committed },
    { "host_bytes_committed", st.host_bytes_committed },
  };
  for (size_t i = 0; i < sizeof counters / sizeof counters[0]; ++i)
    if (dict_set_steal(d,
                       counters[i].name,
                       PyLong_FromUnsignedLongLong(counters[i].val)) < 0)
      goto Fail;

  return d;
Fail:
  Py_DECREF(d);
  return NULL;
}

static PyObject*
Pipeline_stats_reset(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  damacy_stats_reset(self->handle);
  Py_RETURN_NONE;
}

static PyObject*
Pipeline_get_device(PipelineObj* self, void* Py_UNUSED(closure))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  return PyLong_FromLong(damacy_get_device(self->handle));
}

static PyGetSetDef Pipeline_getset[] = {
  { "device",
    (getter)Pipeline_get_device,
    NULL,
    "CUDA device index this pipeline is bound to.",
    NULL },
  { NULL, NULL, NULL, NULL, NULL },
};

static PyObject*
Pipeline_shutdown(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  WITH_GIL_RELEASED(damacy_shutdown(self->handle));
  Py_RETURN_NONE;
}

static PyMethodDef Pipeline_methods[] = {
  { "shutdown",
    (PyCFunction)Pipeline_shutdown,
    METH_NOARGS,
    "Stop work and retain exported buffers." },
  { "push",
    (PyCFunction)Pipeline_push,
    METH_O,
    "Push a sequence of {uri, aabb} dicts. Returns "
    "{consumed: int, status: str}." },
  { "pop",
    (PyCFunction)Pipeline_pop,
    METH_NOARGS,
    "Block until the next batch is on-device-ready. Returns a Batch." },
  { "stats",
    (PyCFunction)Pipeline_stats,
    METH_NOARGS,
    "Cumulative pipeline metrics as a dict." },
  { "stats_reset",
    (PyCFunction)Pipeline_stats_reset,
    METH_NOARGS,
    "Reset all cumulative counters." },
  { NULL, NULL, 0, NULL },
};

PyTypeObject PipelineType = {
  PyVarObject_HEAD_INIT(NULL, 0).tp_name = "damacy._native.Pipeline",
  .tp_basicsize = sizeof(PipelineObj),
  .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
  .tp_doc = "Native streaming pipeline handle. Implementation detail of "
            "damacy.Pipeline; prefer the wrapper, which adds typed config, "
            "the exception hierarchy, and lazy-generator push semantics.",
  .tp_new = PyType_GenericNew,
  .tp_init = (initproc)Pipeline_init,
  .tp_dealloc = (destructor)Pipeline_dealloc,
  .tp_methods = Pipeline_methods,
  .tp_getset = Pipeline_getset,
};

PyObject*
api_raise_status(enum damacy_status status, const char* what)
{
  return raise_status(status, what);
}

PyObject*
api_pipeline_from_components(struct damacy_planner* planner,
                             struct damacy_executor* executor,
                             const struct damacy_batch_spec* output,
                             const struct damacy_queue_limits* queues,
                             PyObject* dependencies)
{
  PipelineObj* self = (PipelineObj*)PipelineType.tp_alloc(&PipelineType, 0);
  if (!self)
    return NULL;
  Py_INCREF(dependencies);
  self->dependencies = dependencies;
  enum damacy_status status;
  Py_BEGIN_ALLOW_THREADS status =
    damacy_pipeline_create(planner, executor, output, queues, &self->handle);
  Py_END_ALLOW_THREADS if (status != DAMACY_OK)
  {
    Py_DECREF(self);
    return raise_status(status, "create");
  }
  return (PyObject*)self;
}

// ---------- registration ----------

int
api_register_types(PyObject* m)
{
  ADD_TYPE(m, "Pipeline", PipelineType);
  ADD_TYPE(m, "Batch", BatchType);

  // DamacyError(message) — subclass of RuntimeError. Carries .status
  // (int, one of STATUS_*) and .what (str) attributes; the Python
  // wrapper module uses .status to remap to per-status subclasses.
  DamacyError = PyErr_NewExceptionWithDoc(
    "damacy._native.DamacyError",
    "Native damacy error. .status is one of damacy._native.STATUS_*.",
    PyExc_RuntimeError,
    NULL);
  if (!DamacyError)
    return -1;
  Py_INCREF(DamacyError);
  if (PyModule_AddObject(m, "DamacyError", DamacyError) < 0) {
    Py_DECREF(DamacyError);
    Py_DECREF(DamacyError);
    DamacyError = NULL;
    return -1;
  }

  // Status code integer constants. Mirror enum damacy_status; consumed
  // by the Python wrapper to map raw exceptions to typed subclasses.
  static const struct
  {
    const char* name;
    int value;
  } statuses[] = {
    { "STATUS_OK", DAMACY_OK },         { "STATUS_AGAIN", DAMACY_AGAIN },
    { "STATUS_INVAL", DAMACY_INVAL },   { "STATUS_NOTFOUND", DAMACY_NOTFOUND },
    { "STATUS_DTYPE", DAMACY_DTYPE },   { "STATUS_RANK", DAMACY_RANK },
    { "STATUS_IO", DAMACY_IO },         { "STATUS_DECODE", DAMACY_DECODE },
    { "STATUS_CUDA", DAMACY_CUDA },     { "STATUS_OOM", DAMACY_OOM },
    { "STATUS_BUDGET", DAMACY_BUDGET }, { "STATUS_SHUTDOWN", DAMACY_SHUTDOWN },
  };
  for (size_t i = 0; i < sizeof statuses / sizeof statuses[0]; ++i) {
    if (PyModule_AddIntConstant(m, statuses[i].name, statuses[i].value) < 0)
      return -1;
  }

  // dtype enum mirrors damacy_dtype. Tagged DTYPE_ to avoid collision
  // with future U16/I32/... source-side dtype constants.
  if (PyModule_AddIntConstant(m, "DTYPE_F32", DAMACY_F32) < 0)
    return -1;
  if (PyModule_AddIntConstant(m, "DTYPE_BF16", DAMACY_BF16) < 0)
    return -1;

  if (PyModule_AddIntConstant(m, "NUMA_AUTO", DAMACY_NUMA_AUTO) < 0)
    return -1;
  if (PyModule_AddIntConstant(m, "NUMA_DISABLED", DAMACY_NUMA_DISABLED) < 0)
    return -1;
  if (PyModule_AddIntConstant(m, "NUMA_PIN_TO", DAMACY_NUMA_PIN_TO) < 0)
    return -1;

  if (PyModule_AddIntConstant(m, "GDS_AUTO", DAMACY_GDS_AUTO) < 0)
    return -1;
  if (PyModule_AddIntConstant(m, "GDS_ON", DAMACY_GDS_ON) < 0)
    return -1;
  if (PyModule_AddIntConstant(m, "GDS_OFF", DAMACY_GDS_OFF) < 0)
    return -1;

  // Tuning defaults come from damacy_tuning_defaults() (the single source of
  // truth); bounds are mirrored from damacy_limits.h. The Python Config
  // surface uses real values, not 0-sentinels, and validates against the same
  // ranges as validate_config().
  struct damacy_tuning d = damacy_tuning_defaults();
  struct
  {
    const char* name;
    long long value;
  } limits[] = {
    { "DEFAULT_CHUNK_UNCOMPRESSED_BYTES", d.max_chunk_uncompressed_bytes },
    { "DEFAULT_READ_OP_MAX_BYTES", d.max_read_op_bytes },
    { "DEFAULT_MAX_INDEX_BYTES", d.max_index_bytes },
    { "DEFAULT_HOST_BUFFER_WAVES", d.host_buffer_waves },
    { "DEFAULT_MAX_CHUNKS_PER_WAVE", d.max_chunks_per_wave },
    { "DEFAULT_MAX_SUBSTREAMS_PER_CHUNK", d.max_substreams_per_chunk },
    { "DEFAULT_METADATA_IO_CONCURRENCY", d.metadata_io_concurrency },
    { "DEFAULT_IO_THREADS", d.n_io_threads },
    { "DEFAULT_ARRAY_META_CACHE", d.n_array_meta_cache },
    { "DEFAULT_SHARD_INDEX_CACHE", d.n_shard_index_cache },
    { "DEFAULT_CHUNK_LAYOUT_CACHE", d.n_chunk_layout_cache },
    { "DEFAULT_MAX_SHARDS_PER_SAMPLE", d.max_shards_per_sample },
    { "MAX_RANK", DAMACY_MAX_RANK },
    { "MAX_CHUNK_BYTES", DAMACY_MAX_CHUNK_BYTES },
    { "MAX_READ_OP_BYTES", UINT32_MAX },
    { "N_WAVES", DAMACY_N_WAVES },
    { "MAX_HOST_BUFFER_WAVES", DAMACY_MAX_HOST_BUFFER_WAVES },
    { "HARD_MAX_CHUNKS_PER_WAVE", DAMACY_HARD_MAX_CHUNKS_PER_WAVE },
    { "HARD_MAX_SUBSTREAMS_PER_CHUNK", DAMACY_HARD_MAX_SUBSTREAMS_PER_CHUNK },
    { "MAX_METADATA_IO_CONCURRENCY", DAMACY_MAX_METADATA_IO_CONCURRENCY },
    { "MAX_IO_THREADS", DAMACY_MAX_IO_THREADS },
  };
  for (size_t i = 0; i < sizeof(limits) / sizeof(limits[0]); ++i) {
    if (PyModule_AddIntConstant(m, limits[i].name, limits[i].value) < 0)
      return -1;
  }
  return 0;
}
