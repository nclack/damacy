// Pipeline / Batch python bindings around the public C API.
//
// Pipeline(...)   → damacy_create   p.push(s)  → damacy_push
// p.pop()         → damacy_pop      p.flush()  → damacy_flush
// p.stats()       → damacy_stats_get → dict
// batch.release() → damacy_release (tp_dealloc auto-releases)
// batch.info      → dict snapshot of damacy_batch_info
// batch.__dlpack__ → DLPack v0 (default) or v1 capsule, dispatched by
//                     the consumer's `max_version` kwarg.
//
// Long-running C calls (push/pop/flush/destroy/release) drop the GIL.

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "_api.h"
#include "damacy.h"

#include <cuda.h>
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
  PyObject_HEAD struct damacy* handle; // strong ref while not destroyed
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

  return Py_BuildValue("{s:K,s:N,s:s,s:K,s:K}",
                       "device_ptr",
                       (unsigned long long)(uintptr_t)info.device_ptr,
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
  PyObject* batch;                    // strong ref; dropped by deleter
};

// Drop the batch ref and free the payload under the GIL. Shared by both
// v0 and v1 deleters. PyMem_Free uses pymalloc which itself requires the
// GIL, so the free stays inside the GIL window.
static void
dlpack_payload_free(struct dlpack_payload* p)
{
  if (!p)
    return;
  PyGILState_STATE g = PyGILState_Ensure();
  Py_XDECREF(p->batch);
  PyMem_Free(p);
  PyGILState_Release(g);
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
  (void)dl_device;
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

  // Resolve device id from the device pointer.
  int dev_id = 0;
  unsigned int ord = 0;
  if (cuPointerGetAttribute(&ord,
                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            (CUdeviceptr)info.device_ptr) == CUDA_SUCCESS)
    dev_id = (int)ord;

  if (sync_streams_for_consumer(info.ready_stream, stream_obj) != 0)
    return NULL;

  struct dlpack_payload* p = PyMem_Calloc(1, sizeof *p);
  if (!p)
    return PyErr_NoMemory();

  for (int i = 0; i < info.rank; ++i)
    p->shape[i] = info.shape[i];

  Py_INCREF(self);
  p->batch = (PyObject*)self;

  DLTensor dl = {
    .data = info.device_ptr,
    .device = (DLDevice){ .device_type = kDLCUDA, .device_id = dev_id },
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
    Py_DECREF(self);
    PyMem_Free(p);
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
  unsigned int ord = 0;
  if (cuPointerGetAttribute(&ord,
                            CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                            (CUdeviceptr)info.device_ptr) != CUDA_SUCCESS)
    ord = 0;
  return Py_BuildValue("(ii)", (int)kDLCUDA, (int)ord);
}

static PyMethodDef Batch_methods[] = {
  { "release",
    (PyCFunction)(void (*)(void))Batch_release,
    METH_VARARGS | METH_KEYWORDS,
    "release(event=None): return the slot to the pool. With event=None "
    "(default), release is immediate. With event set to an integer CUevent "
    "handle, damacy stream-waits on it before reusing the slot's buffer — "
    "the host returns immediately. Idempotent." },
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
    "DLPack device tuple: (kDLCUDA=2, ordinal)." },
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
  // kws[] / format string / variable list mirror struct damacy_config —
  // keep all three in sync when adding a field.
  static char* kws[] = { "batch_size",
                         "lookahead_batches",
                         "n_io_threads",
                         "n_zarrs_meta_cache",
                         "n_shards_meta_cache",
                         "dtype",
                         "max_chunk_uncompressed_bytes",
                         "max_gpu_memory_bytes",
                         "sample_shape",
                         "device",
                         "host_buffer_waves",
                         "max_read_op_bytes",
                         "max_chunks_per_wave",
                         "max_substreams_per_chunk",
                         "enable_gds",
                         "numa_strategy",
                         "numa_node",
                         "bypass_decode",
                         NULL };
  unsigned int batch_size = 0;
  unsigned int lookahead = 2;
  unsigned int n_io = 4;
  unsigned int n_zarrs_meta = 64;
  unsigned int n_shards_meta = 256;
  PyObject* dtype_obj = NULL;
  unsigned int max_chunk_uncompressed = 0;
  unsigned long long max_gpu_bytes = 0;
  PyObject* sample_shape_obj = NULL;
  int device = -1;
  unsigned int host_buffer_waves = 0;
  unsigned long long max_read_op_bytes = 0;
  unsigned int max_chunks_per_wave = 0;
  unsigned int max_substreams_per_chunk = 0;
  int enable_gds = DAMACY_GDS_AUTO;
  int numa_strategy = DAMACY_NUMA_AUTO;
  int numa_node = -1;
  int bypass_decode = 0;
  if (!PyArg_ParseTupleAndKeywords(args,
                                   kw,
                                   "IIIIIOIKO|iIKIIiiip",
                                   kws,
                                   &batch_size,
                                   &lookahead,
                                   &n_io,
                                   &n_zarrs_meta,
                                   &n_shards_meta,
                                   &dtype_obj,
                                   &max_chunk_uncompressed,
                                   &max_gpu_bytes,
                                   &sample_shape_obj,
                                   &device,
                                   &host_buffer_waves,
                                   &max_read_op_bytes,
                                   &max_chunks_per_wave,
                                   &max_substreams_per_chunk,
                                   &enable_gds,
                                   &numa_strategy,
                                   &numa_node,
                                   &bypass_decode))
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

  enum damacy_dtype dt;
  if (parse_dtype(dtype_obj, &dt) != 0)
    return -1;

  struct damacy_config cfg = {
    .batch_size = batch_size,
    .lookahead_batches = lookahead,
    .dtype = dt,
    .device = device,
    .tuning = {
      .n_io_threads = n_io,
      .n_zarrs_meta_cache = n_zarrs_meta,
      .n_shards_meta_cache = n_shards_meta,
      .max_chunk_uncompressed_bytes = max_chunk_uncompressed,
      .max_read_op_bytes = (uint64_t)max_read_op_bytes,
      .max_gpu_memory_bytes = (uint64_t)max_gpu_bytes,
      .host_buffer_waves = (uint8_t)host_buffer_waves,
      .max_chunks_per_wave = (uint32_t)max_chunks_per_wave,
      .max_substreams_per_chunk = (uint16_t)max_substreams_per_chunk,
      .enable_gds = (enum damacy_gds_mode)enable_gds,
      .numa_strategy = (enum damacy_numa_strategy)numa_strategy,
      .numa_node = numa_node,
    },
    .debug = { .bypass_decode = (uint8_t)(bypass_decode ? 1 : 0) },
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
  PyObject* uri = PyDict_GetItemString(obj, "uri");   // borrowed
  PyObject* aabb = PyDict_GetItemString(obj, "aabb"); // borrowed
  if (!uri || !aabb) {
    PyErr_SetString(PyExc_KeyError, "sample requires 'uri' and 'aabb'");
    return -1;
  }
  const char* uri_s = PyUnicode_AsUTF8(uri);
  if (!uri_s)
    return -1;
  if (!PyList_Check(aabb) && !PyTuple_Check(aabb)) {
    PyErr_SetString(PyExc_TypeError,
                    "aabb must be a list or tuple of (beg,end)");
    return -1;
  }
  Py_ssize_t n = PySequence_Fast_GET_SIZE(aabb);
  if (n < 1 || n > DAMACY_MAX_RANK) {
    PyErr_Format(
      PyExc_ValueError, "aabb rank out of range: %zd", (Py_ssize_t)n);
    return -1;
  }

  out->uri = uri_s;
  out->aabb.rank = (uint8_t)n;
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject* item = PySequence_Fast_GET_ITEM(aabb, i);
    long long beg, end;
    if (!PyArg_ParseTuple(item, "LL", &beg, &end)) {
      PyErr_Format(PyExc_ValueError, "aabb[%zd] must be a (beg,end) pair", i);
      return -1;
    }
    out->aabb.dims[i].beg = (int64_t)beg;
    out->aabb.dims[i].end = (int64_t)end;
  }
  return 0;
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
      PyMem_Free(buf);
      return NULL;
    }
  }

  struct damacy_sample_slice slice = { .beg = buf, .end = buf + n };
  struct damacy_push_result r;
  WITH_GIL_RELEASED(r = damacy_push(self->handle, slice));

  Py_ssize_t consumed = (Py_ssize_t)(r.unconsumed.beg - buf);
  PyMem_Free(buf);

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
  return (PyObject*)batch_new(self, b);
}

static PyObject*
Pipeline_flush(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  enum damacy_status s;
  WITH_GIL_RELEASED(s = damacy_flush(self->handle));
  if (s != DAMACY_OK)
    return raise_status(s, "flush");
  Py_RETURN_NONE;
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
    { "h2d", &st.h2d },
    { "decode", &st.decode },
    { "post_decode", &st.post_decode },
    { "decode_gap", &st.decode_gap },
    { "assemble", &st.assemble },
    { "bind_wait", &st.bind_wait },
    { "pop_wait", &st.pop_wait },
    { "flush_wait", &st.flush_wait },
  };
  for (size_t i = 0; i < sizeof metrics / sizeof metrics[0]; ++i)
    if (dict_set_steal(d, metrics[i].name, metric_to_dict(metrics[i].m)) < 0)
      goto Fail;

  const struct
  {
    const char* name;
    unsigned long long val;
  } counters[] = {
    { "zarr_meta_hits", st.zarr_meta_hits },
    { "zarr_meta_misses", st.zarr_meta_misses },
    { "shard_idx_hits", st.shard_idx_hits },
    { "shard_idx_misses", st.shard_idx_misses },
    { "batches_emitted", st.batches_emitted },
    { "batches_truncated", st.batches_truncated },
    { "waves_emitted", st.waves_emitted },
    { "worker_steps", st.worker_steps },
    { "chunks_planned", st.chunks_planned },
    { "chunks_to_load", st.chunks_to_load },
    { "chunks_dispatched", st.chunks_dispatched },
    { "reads_issued", st.reads_issued },
    { "gpu_bytes_committed", st.gpu_bytes_committed },
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

static PyMethodDef Pipeline_methods[] = {
  { "push",
    (PyCFunction)Pipeline_push,
    METH_O,
    "Push a sequence of {uri, aabb} dicts. Returns "
    "{consumed: int, status: str}." },
  { "pop",
    (PyCFunction)Pipeline_pop,
    METH_NOARGS,
    "Block until the next batch is on-device-ready. Returns a Batch." },
  { "flush",
    (PyCFunction)Pipeline_flush,
    METH_NOARGS,
    "Drain in-flight work and ready any partial last batch." },
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
  return 0;
}
