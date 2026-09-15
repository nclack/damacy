#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_api.h"
#include "damacy_pipeline.h"

enum component_kind
{
  READER,
  METADATA_READER,
  METADATA,
  PLANNER,
  EXECUTOR
};

struct component
{
  enum component_kind kind;
  void* value;
  PyObject* dependencies;
};

static const char* names[] = { "damacy.Reader",
                               "damacy.MetadataReader",
                               "damacy.Metadata",
                               "damacy.Planner",
                               "damacy.Executor" };

static void
destroy_value(enum component_kind kind, void* value)
{
  switch (kind) {
    case READER:
      damacy_reader_destroy(value);
      break;
    case METADATA_READER:
      damacy_metadata_reader_destroy(value);
      break;
    case METADATA:
      damacy_metadata_destroy(value);
      break;
    case PLANNER:
      damacy_planner_destroy(value);
      break;
    case EXECUTOR:
      damacy_executor_destroy(value);
      break;
  }
}

static void
component_destroy(PyObject* capsule)
{
  struct component* component =
    PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
  if (!component)
    return;
  Py_BEGIN_ALLOW_THREADS destroy_value(component->kind, component->value);
  Py_END_ALLOW_THREADS Py_XDECREF(component->dependencies);
  PyMem_Free(component);
}

static PyObject*
component_new(enum component_kind kind, void* value, PyObject* dependencies)
{
  struct component* component = PyMem_Calloc(1, sizeof(*component));
  if (!component) {
    destroy_value(kind, value);
    return PyErr_NoMemory();
  }
  component->kind = kind;
  component->value = value;
  Py_XINCREF(dependencies);
  component->dependencies = dependencies;
  PyObject* capsule = PyCapsule_New(component, names[kind], component_destroy);
  if (!capsule) {
    Py_XDECREF(dependencies);
    destroy_value(kind, value);
    PyMem_Free(component);
  }
  return capsule;
}

static void*
component_value(PyObject* capsule, enum component_kind kind)
{
  struct component* component = PyCapsule_GetPointer(capsule, names[kind]);
  return component ? component->value : NULL;
}

static PyObject*
create_reader(PyObject* self, PyObject* args)
{
  (void)self;
  unsigned int workers, reads;
  if (!PyArg_ParseTuple(args, "II", &workers, &reads))
    return NULL;
  struct damacy_reader* reader = NULL;
  enum damacy_status status;
  Py_BEGIN_ALLOW_THREADS status =
    damacy_file_reader_create(workers, reads, &reader);
  Py_END_ALLOW_THREADS if (status != DAMACY_OK) return api_raise_status(
    status, "create reader");
  return component_new(READER, reader, NULL);
}

static PyObject*
create_metadata_reader(PyObject* self, PyObject* args)
{
  (void)self;
  unsigned int concurrency;
  unsigned long long baseline, cap, seed;
  double mu, sigma;
  if (!PyArg_ParseTuple(
        args, "IKddKK", &concurrency, &baseline, &mu, &sigma, &cap, &seed))
    return NULL;
  struct damacy_metadata_reader* reader = NULL;
  struct damacy_latency_model latency = { .baseline_ns = baseline,
                                          .lognormal_mu_ln_ns = mu,
                                          .lognormal_sigma_ln_ns = sigma,
                                          .cap_ns = cap,
                                          .seed = seed };
  enum damacy_status status =
    damacy_file_metadata_reader_create(concurrency, &latency, &reader);
  if (status != DAMACY_OK)
    return api_raise_status(status, "create metadata reader");
  return component_new(METADATA_READER, reader, NULL);
}

static PyObject*
create_metadata(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* reader_object;
  struct damacy_metadata_cache_config cache;
  if (!PyArg_ParseTuple(args,
                        "OII",
                        &reader_object,
                        &cache.array_entries,
                        &cache.shard_entries))
    return NULL;
  struct damacy_metadata_reader* reader =
    component_value(reader_object, METADATA_READER);
  if (!reader)
    return NULL;
  struct damacy_metadata* metadata = NULL;
  enum damacy_status status =
    damacy_zarr_metadata_create(reader, &cache, &metadata);
  if (status != DAMACY_OK)
    return api_raise_status(status, "create metadata");
  return component_new(METADATA, metadata, reader_object);
}

static PyObject*
create_planner(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* metadata_object;
  unsigned long long bytes;
  struct damacy_plan_limits limits;
  if (!PyArg_ParseTuple(args,
                        "OIIIK",
                        &metadata_object,
                        &limits.max_chunks,
                        &limits.max_chunk_bytes,
                        &limits.max_shards_per_sample,
                        &bytes))
    return NULL;
  limits.max_plan_bytes = bytes;
  struct damacy_metadata* metadata = component_value(metadata_object, METADATA);
  if (!metadata)
    return NULL;
  struct damacy_planner* planner = NULL;
  enum damacy_status status =
    damacy_chunk_planner_create(metadata, &limits, &planner);
  if (status != DAMACY_OK)
    return api_raise_status(status, "create planner");
  return component_new(PLANNER, planner, metadata_object);
}

static PyObject*
create_cpu_executor(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* reader_object;
  unsigned long long bytes;
  struct damacy_cpu_config config;
  if (!PyArg_ParseTuple(args,
                        "OIIIKI",
                        &reader_object,
                        &config.decode_workers,
                        &config.max_encoded_chunk_bytes,
                        &config.max_decoded_chunk_bytes,
                        &bytes,
                        &config.chunks_per_input_buffer))
    return NULL;
  config.max_memory_bytes = bytes;
  struct damacy_reader* reader = component_value(reader_object, READER);
  if (!reader)
    return NULL;
  struct damacy_executor* executor = NULL;
  enum damacy_status status =
    damacy_cpu_executor_create(reader, &config, &executor);
  if (status != DAMACY_OK)
    return api_raise_status(status, "create CPU executor");
  return component_new(EXECUTOR, executor, reader_object);
}

static PyObject*
create_cuda_executor(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* reader_object;
  unsigned long long memory, read_bytes, index_bytes;
  unsigned int buffers;
  int numa, gds;
  struct damacy_cuda_config config = { 0 };
  if (!PyArg_ParseTuple(args,
                        "OiKIKIIIIiiiK",
                        &reader_object,
                        &config.device,
                        &memory,
                        &config.max_chunk_bytes,
                        &read_bytes,
                        &config.max_chunks_per_wave,
                        &config.max_substreams_per_chunk,
                        &buffers,
                        &config.chunk_layout_entries,
                        &numa,
                        &config.numa_node,
                        &gds,
                        &index_bytes))
    return NULL;
  if (buffers > UINT8_MAX) {
    PyErr_SetString(PyExc_ValueError, "host_buffer_waves is too large");
    return NULL;
  }
  config.max_gpu_memory_bytes = memory;
  config.max_index_bytes = index_bytes;
  config.max_read_bytes = read_bytes;
  config.host_buffer_waves = (uint8_t)buffers;
  config.numa_strategy = (enum damacy_numa_strategy)numa;
  config.enable_gds = (enum damacy_gds_mode)gds;
  struct damacy_reader* reader = component_value(reader_object, READER);
  if (!reader)
    return NULL;
  struct damacy_executor* executor = NULL;
  enum damacy_status status =
    damacy_cuda_executor_create(reader, &config, &executor);
  if (status != DAMACY_OK)
    return api_raise_status(status, "create CUDA executor");
  return component_new(EXECUTOR, executor, reader_object);
}

static PyObject*
compose_pipeline(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject *planner_object, *executor_object, *shape_object;
  unsigned int dtype;
  struct damacy_batch_spec output = { 0 };
  struct damacy_queue_limits queues;
  if (!PyArg_ParseTuple(args,
                        "OOOIIII",
                        &planner_object,
                        &executor_object,
                        &shape_object,
                        &output.samples_per_batch,
                        &dtype,
                        &queues.lookahead_samples,
                        &queues.prepared_batches))
    return NULL;
  output.dtype = (enum damacy_dtype)dtype;
  struct damacy_planner* planner = component_value(planner_object, PLANNER);
  if (!planner)
    return NULL;
  struct damacy_executor* executor = component_value(executor_object, EXECUTOR);
  if (!executor)
    return NULL;
  PyObject* shape = PySequence_Fast(shape_object, "shape must be a sequence");
  if (!shape)
    return NULL;
  Py_ssize_t rank = PySequence_Fast_GET_SIZE(shape);
  if (rank < 1 || rank > DAMACY_MAX_RANK) {
    Py_DECREF(shape);
    PyErr_SetString(PyExc_ValueError, "invalid output rank");
    return NULL;
  }
  output.sample_rank = (uint8_t)rank;
  for (Py_ssize_t i = 0; i < rank; ++i) {
    output.sample_shape[i] =
      PyLong_AsLongLong(PySequence_Fast_GET_ITEM(shape, i));
    if (PyErr_Occurred()) {
      Py_DECREF(shape);
      return NULL;
    }
  }
  Py_DECREF(shape);
  PyObject* dependencies = PyTuple_Pack(2, planner_object, executor_object);
  if (!dependencies)
    return NULL;
  PyObject* result = api_pipeline_from_components(
    planner, executor, &output, &queues, dependencies);
  Py_DECREF(dependencies);
  return result;
}

static PyMethodDef methods[] = {
  { "create_reader", create_reader, METH_VARARGS, NULL },
  { "create_metadata_reader", create_metadata_reader, METH_VARARGS, NULL },
  { "create_metadata", create_metadata, METH_VARARGS, NULL },
  { "create_planner", create_planner, METH_VARARGS, NULL },
  { "create_cpu_executor", create_cpu_executor, METH_VARARGS, NULL },
  { "create_cuda_executor", create_cuda_executor, METH_VARARGS, NULL },
  { "compose_pipeline", compose_pipeline, METH_VARARGS, NULL },
  { NULL, NULL, 0, NULL }
};

int
components_register(PyObject* module)
{
  return PyModule_AddFunctions(module, methods);
}

struct damacy_metadata_reader*
api_metadata_reader(PyObject* capsule)
{
  return component_value(capsule, METADATA_READER);
}
