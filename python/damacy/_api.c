// Pipeline / Batch python bindings around the public C API.
//
// Pipeline(...)   → damacy_create   p.push(s)  → damacy_push
// p.pop()         → damacy_pop      p.flush()  → damacy_flush
// p.stats()       → damacy_stats_get → dict
// batch.release() → damacy_release (tp_dealloc auto-releases)
// batch.info      → dict snapshot of damacy_batch_info
// batch.__dlpack__ → DLPack v1 capsule
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

// ---------- DLPack (v1.0) — minimal definitions, no external dep. ----------
// https://dmlc.github.io/dlpack/latest/c_api.html

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

static PyObject*
Batch_release(BatchObj* self, PyObject* Py_UNUSED(ignored))
{
  batch_do_release(self);
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

// Storage layout for one exported tensor. Lives in DLManagedTensorVersioned
// .manager_ctx so the deleter can free it.
struct dlpack_payload
{
  DLManagedTensorVersioned mt;
  int64_t shape[DAMACY_MAX_RANK + 1]; // referenced by mt.dl_tensor.shape
  PyObject* batch;                    // strong ref; dropped by deleter
};

static void
dlpack_deleter(DLManagedTensorVersioned* self)
{
  if (!self)
    return;
  struct dlpack_payload* p = (struct dlpack_payload*)self->manager_ctx;
  // Acquire the GIL; the consumer may call us from any thread.
  PyGILState_STATE g = PyGILState_Ensure();
  Py_XDECREF(p->batch);
  PyGILState_Release(g);
  PyMem_Free(p);
}

static void
dlpack_capsule_destructor(PyObject* capsule)
{
  // Only fires when the capsule was never consumed; otherwise the
  // consumer renames to "used_dltensor_versioned" and runs the deleter.
  const char* name = PyCapsule_GetName(capsule);
  if (!name)
    return;
  if (strcmp(name, "dltensor_versioned") == 0) {
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
    } else if (s == 1 || s == 2) {
      consumer = (CUstream)(uintptr_t)s; // legacy / per-thread
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
  (void)max_version;
  (void)dl_device;
  if (copy != Py_None && PyObject_IsTrue(copy)) {
    PyErr_SetString(PyExc_BufferError,
                    "damacy DLPack: copy=True not supported");
    return NULL;
  }
  RETURN_IF_DESTROYED(self, "Batch has been released");

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

  p->mt.version = (DLPackVersion){ 1, 0 };
  p->mt.manager_ctx = p;
  p->mt.deleter = dlpack_deleter;
  p->mt.flags = 0;
  p->mt.dl_tensor = (DLTensor){
    .data = info.device_ptr,
    .device = (DLDevice){ .device_type = kDLCUDA, .device_id = dev_id },
    .ndim = (int32_t)info.rank,
    .dtype = dlt,
    .shape = p->shape,
    .strides = NULL, // contiguous
    .byte_offset = 0,
  };

  PyObject* cap =
    PyCapsule_New(&p->mt, "dltensor_versioned", dlpack_capsule_destructor);
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
    (PyCFunction)Batch_release,
    METH_NOARGS,
    "Return the batch slot to the pool. Idempotent." },
  { "__dlpack__",
    (PyCFunction)(void (*)(void))Batch_dlpack,
    METH_VARARGS | METH_KEYWORDS,
    "DLPack v1 capsule export. Honors stream=... per the DLPack protocol." },
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
  static char* kws[] = { "batch_size",
                         "lookahead_batches",
                         "n_io_threads",
                         "host_buffer_bytes",
                         "device_buffer_bytes",
                         "n_zarrs_meta_cache",
                         "n_shards_meta_cache",
                         "dtype",
                         "max_chunk_uncompressed_bytes",
                         "max_gpu_memory_bytes",
                         "max_bytes_per_element",
                         "device",
                         "n_compute_threads",
                         NULL };
  unsigned int batch_size = 0;
  unsigned int lookahead = 2;
  unsigned int n_io = 4;
  unsigned long long host_bytes = 0;
  unsigned long long dev_bytes = 0;
  unsigned int n_zarrs_meta = 64;
  unsigned int n_shards_meta = 256;
  PyObject* dtype_obj = NULL;
  unsigned int max_chunk_uncompressed = 0;
  unsigned long long max_gpu_bytes = 0;
  unsigned char max_bytes_per_element = 0;
  int device = -1;
  unsigned int n_compute = 0;
  if (!PyArg_ParseTupleAndKeywords(args,
                                   kw,
                                   "IIIKKIIOI|KBiI",
                                   kws,
                                   &batch_size,
                                   &lookahead,
                                   &n_io,
                                   &host_bytes,
                                   &dev_bytes,
                                   &n_zarrs_meta,
                                   &n_shards_meta,
                                   &dtype_obj,
                                   &max_chunk_uncompressed,
                                   &max_gpu_bytes,
                                   &max_bytes_per_element,
                                   &device,
                                   &n_compute))
    return -1;

  enum damacy_dtype dt;
  if (parse_dtype(dtype_obj, &dt) != 0)
    return -1;

  struct damacy_config cfg = {
    .batch_size = batch_size,
    .lookahead_batches = lookahead,
    .n_io_threads = n_io,
    .host_buffer_bytes = (uint64_t)host_bytes,
    .device_buffer_bytes = (uint64_t)dev_bytes,
    .n_zarrs_meta_cache = n_zarrs_meta,
    .n_shards_meta_cache = n_shards_meta,
    .dtype = dt,
    .max_chunk_uncompressed_bytes = max_chunk_uncompressed,
    .max_gpu_memory_bytes = (uint64_t)max_gpu_bytes,
    .max_bytes_per_element = max_bytes_per_element,
    .device = device,
    .n_compute_threads = n_compute,
  };

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

static PyObject*
Pipeline_stats(PipelineObj* self, PyObject* Py_UNUSED(ignored))
{
  RETURN_IF_DESTROYED(self, "Pipeline has been destroyed");
  struct damacy_stats st;
  damacy_stats_get(self->handle, &st);
  return Py_BuildValue("{s:N,s:N,s:N,s:N,s:N,s:N,s:N,s:N,"
                       "s:K,s:K,s:K,s:K,s:K,s:K,s:K,s:K}",
                       "plan",
                       metric_to_dict(&st.plan),
                       "io",
                       metric_to_dict(&st.io),
                       "h2d",
                       metric_to_dict(&st.h2d),
                       "decompress",
                       metric_to_dict(&st.decompress),
                       "assemble",
                       metric_to_dict(&st.assemble),
                       "pop_wait_io",
                       metric_to_dict(&st.pop_wait_io),
                       "pop_wait_compute",
                       metric_to_dict(&st.pop_wait_compute),
                       "flush_wait",
                       metric_to_dict(&st.flush_wait),
                       "zarr_meta_hits",
                       (unsigned long long)st.zarr_meta_hits,
                       "zarr_meta_misses",
                       (unsigned long long)st.zarr_meta_misses,
                       "shard_idx_hits",
                       (unsigned long long)st.shard_idx_hits,
                       "shard_idx_misses",
                       (unsigned long long)st.shard_idx_misses,
                       "batches_emitted",
                       (unsigned long long)st.batches_emitted,
                       "batches_truncated",
                       (unsigned long long)st.batches_truncated,
                       "waves_emitted",
                       (unsigned long long)st.waves_emitted,
                       "gpu_bytes_committed",
                       (unsigned long long)st.gpu_bytes_committed);
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
    { "STATUS_OK", DAMACY_OK },
    { "STATUS_AGAIN", DAMACY_AGAIN },
    { "STATUS_INVAL", DAMACY_INVAL },
    { "STATUS_NOTFOUND", DAMACY_NOTFOUND },
    { "STATUS_DTYPE", DAMACY_DTYPE },
    { "STATUS_RANK", DAMACY_RANK },
    { "STATUS_IO", DAMACY_IO },
    { "STATUS_DECODE", DAMACY_DECODE },
    { "STATUS_CUDA", DAMACY_CUDA },
    { "STATUS_OOM", DAMACY_OOM },
    { "STATUS_SHUTDOWN", DAMACY_SHUTDOWN },
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
  return 0;
}
