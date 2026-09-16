#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "_api.h"
#include "damacy_spatial.h"

static const char image_name[] = "damacy.NgffImage";

static void
image_destroy(PyObject* capsule)
{
  damacy_ngff_image_destroy(PyCapsule_GetPointer(capsule, image_name));
}

static int
put(PyObject* dict, const char* key, PyObject* value)
{
  if (!value)
    return -1;
  int result = PyDict_SetItemString(dict, key, value);
  Py_DECREF(value);
  return result;
}

static PyObject*
integers(const int64_t* values, uint8_t rank)
{
  PyObject* result = PyTuple_New(rank);
  if (!result)
    return NULL;
  for (uint8_t i = 0; i < rank; ++i) {
    PyObject* value = PyLong_FromLongLong(values[i]);
    if (!value) {
      Py_DECREF(result);
      return NULL;
    }
    PyTuple_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject*
doubles(const double* values, uint8_t rank)
{
  PyObject* result = PyTuple_New(rank);
  if (!result)
    return NULL;
  for (uint8_t i = 0; i < rank; ++i) {
    PyObject* value = PyFloat_FromDouble(values[i]);
    if (!value) {
      Py_DECREF(result);
      return NULL;
    }
    PyTuple_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject*
affine(const struct damacy_affine* transform, uint8_t rank)
{
  PyObject* result = PyTuple_New(rank);
  if (!result)
    return NULL;
  for (uint8_t i = 0; i < rank; ++i) {
    double row[DAMACY_MAX_RANK + 1];
    for (uint8_t j = 0; j < rank; ++j)
      row[j] = transform->linear[i][j];
    row[rank] = transform->offset[i];
    PyObject* value = doubles(row, rank + 1);
    if (!value) {
      Py_DECREF(result);
      return NULL;
    }
    PyTuple_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject*
bounds(const struct damacy_aabb* box)
{
  PyObject* result = PyTuple_New(box->rank);
  if (!result)
    return NULL;
  for (uint8_t i = 0; i < box->rank; ++i) {
    PyObject* value = Py_BuildValue(
      "(LL)", (long long)box->dims[i].beg, (long long)box->dims[i].end);
    if (!value) {
      Py_DECREF(result);
      return NULL;
    }
    PyTuple_SET_ITEM(result, i, value);
  }
  return result;
}

static PyObject*
load_image(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject* reader_object;
  const char* uri;
  unsigned int index, levels;
  unsigned long long bytes;
  if (!PyArg_ParseTuple(
        args, "OsIIK", &reader_object, &uri, &index, &levels, &bytes))
    return NULL;
  struct damacy_metadata_reader* reader = api_metadata_reader(reader_object);
  if (!reader)
    return NULL;
  struct damacy_ngff_limits limits = { .max_levels = levels,
                                       .max_metadata_bytes = bytes };
  struct damacy_ngff_image* image = NULL;
  enum damacy_status status;
  Py_BEGIN_ALLOW_THREADS status =
    damacy_ngff_image_load(reader, uri, index, &limits, &image);
  Py_END_ALLOW_THREADS if (status != DAMACY_OK) return api_raise_status(
    status, "load NGFF metadata");
  PyObject* capsule = PyCapsule_New(image, image_name, image_destroy);
  if (!capsule)
    damacy_ngff_image_destroy(image);
  return capsule;
}

static PyObject*
image_info(PyObject* self, PyObject* capsule)
{
  (void)self;
  struct damacy_ngff_image* image = PyCapsule_GetPointer(capsule, image_name);
  if (!image)
    return NULL;
  const struct damacy_ngff_info* info = damacy_ngff_image_info(image);
  PyObject* result = PyDict_New();
  PyObject* axes = PyTuple_New(info->rank);
  PyObject* levels = PyTuple_New(info->level_count);
  if (!result || !axes || !levels)
    goto Fail;
  for (uint8_t i = 0; i < info->rank; ++i) {
    const struct damacy_ngff_axis* axis = &info->axes[i];
    const char* kind = axis->kind == DAMACY_NGFF_SPACE  ? "space"
                       : axis->kind == DAMACY_NGFF_TIME ? "time"
                                                        : "channel";
    PyObject* value = Py_BuildValue("(ssz)", axis->name, kind, axis->unit);
    if (!value)
      goto Fail;
    PyTuple_SET_ITEM(axes, i, value);
  }
  for (uint32_t i = 0; i < info->level_count; ++i) {
    const struct damacy_ngff_level* level = &info->levels[i];
    PyObject* value = PyDict_New();
    if (!value)
      goto Fail;
    PyTuple_SET_ITEM(levels, i, value);
    if (put(value, "uri", PyUnicode_FromString(level->uri)) ||
        put(value, "shape", integers(level->shape, info->rank)) ||
        put(value,
            "scale_to_reference",
            doubles(level->scale_to_reference, info->rank)) ||
        put(value,
            "origin_reference_index",
            doubles(level->origin_reference_index, info->rank)))
      goto Fail;
  }
  if (PyDict_SetItemString(result, "axes", axes) ||
      PyDict_SetItemString(result, "levels", levels) ||
      put(result, "data_type", PyUnicode_FromString(info->data_type)))
    goto Fail;
  Py_DECREF(axes);
  Py_DECREF(levels);
  return result;
Fail:
  Py_XDECREF(result);
  Py_XDECREF(axes);
  Py_XDECREF(levels);
  return NULL;
}

static int
parse_transform(PyObject* object, uint8_t rank, struct damacy_affine* transform)
{
  PyObject* rows = PySequence_Fast(object, "transform must be a sequence");
  if (!rows)
    return -1;
  if (PySequence_Fast_GET_SIZE(rows) != rank) {
    PyErr_SetString(PyExc_ValueError, "transform rank must match output rank");
    Py_DECREF(rows);
    return -1;
  }
  for (uint8_t i = 0; i < rank; ++i) {
    PyObject* row = PySequence_Fast(PySequence_Fast_GET_ITEM(rows, i),
                                    "transform rows must be sequences");
    if (!row) {
      Py_DECREF(rows);
      return -1;
    }
    if (PySequence_Fast_GET_SIZE(row) != rank + 1) {
      PyErr_SetString(PyExc_ValueError, "transform rows need rank + 1 entries");
      Py_DECREF(row);
      Py_DECREF(rows);
      return -1;
    }
    for (uint8_t j = 0; j <= rank; ++j) {
      double value = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(row, j));
      if (PyErr_Occurred()) {
        Py_DECREF(row);
        Py_DECREF(rows);
        return -1;
      }
      if (j == rank)
        transform->offset[i] = value;
      else
        transform->linear[i][j] = value;
    }
    Py_DECREF(row);
  }
  Py_DECREF(rows);
  return 0;
}

static PyObject*
resolution_value(const struct damacy_spatial_resolution* info)
{
  PyObject* result = PyDict_New();
  if (!result)
    return NULL;
  if (put(result, "uri", PyUnicode_FromString(info->uri)) ||
      put(result, "level", PyLong_FromUnsignedLong(info->level)) ||
      put(result, "shape", integers(info->output_shape, info->rank)) ||
      put(result, "source_shape", integers(info->source_shape, info->rank)) ||
      put(result,
          "output_to_source",
          affine(&info->output_to_source, info->rank)) ||
      put(result, "source_bounds_index", bounds(&info->source_bounds_index)) ||
      put(result, "read_bounds_index", bounds(&info->read_bounds_index)) ||
      put(result,
          "requires_resampling",
          PyBool_FromLong(info->requires_resampling))) {
    Py_DECREF(result);
    return NULL;
  }
  return result;
}

static PyObject*
resolve_query(PyObject* self, PyObject* args)
{
  (void)self;
  PyObject *capsule, *shape_object, *transform;
  int filter, boundary, level;
  double value;
  if (!PyArg_ParseTuple(args,
                        "OOOiidi",
                        &capsule,
                        &shape_object,
                        &transform,
                        &filter,
                        &boundary,
                        &value,
                        &level))
    return NULL;
  struct damacy_ngff_image* image = PyCapsule_GetPointer(capsule, image_name);
  if (!image)
    return NULL;
  int64_t output_shape[DAMACY_MAX_RANK];
  PyObject* shape = PySequence_Fast(shape_object, "shape must be a sequence");
  if (!shape)
    return NULL;
  Py_ssize_t rank = PySequence_Fast_GET_SIZE(shape);
  if (rank < 1 || rank > DAMACY_MAX_RANK) {
    Py_DECREF(shape);
    PyErr_SetString(PyExc_ValueError, "invalid output rank");
    return NULL;
  }
  for (Py_ssize_t i = 0; i < rank; ++i) {
    output_shape[i] = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(shape, i));
    if (PyErr_Occurred()) {
      Py_DECREF(shape);
      return NULL;
    }
  }
  Py_DECREF(shape);
  struct damacy_spatial_query query = {
    .sampler = { .filter = (enum damacy_filter)filter,
                 .boundary = (enum damacy_boundary)boundary,
                 .constant_value = value },
    .level = level
  };
  if (parse_transform(transform, (uint8_t)rank, &query.output_to_reference))
    return NULL;
  struct damacy_spatial_resolution resolution;
  enum damacy_status status;
  Py_BEGIN_ALLOW_THREADS status = damacy_spatial_resolve(
    image, &query, (uint8_t)rank, output_shape, &resolution);
  Py_END_ALLOW_THREADS if (status != DAMACY_OK) return api_raise_status(
    status, "resolve spatial query");
  PyObject* result = resolution_value(&resolution);
  damacy_spatial_resolution_clear(&resolution);
  return result;
}

static PyMethodDef methods[] = {
  { "ngff_load", load_image, METH_VARARGS, NULL },
  { "ngff_info", image_info, METH_O, NULL },
  { "spatial_resolve", resolve_query, METH_VARARGS, NULL },
  { NULL, NULL, 0, NULL }
};

int
spatial_register(PyObject* module)
{
  return PyModule_AddFunctions(module, methods);
}
