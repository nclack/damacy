#pragma once

#include "damacy_pipeline.h"
#include <Python.h>

int
api_register_types(PyObject* module);
int
components_register(PyObject* module);
PyObject*
api_raise_status(enum damacy_status status, const char* what);
PyObject*
api_pipeline_from_components(struct damacy_planner* planner,
                             struct damacy_executor* executor,
                             const struct damacy_batch_spec* output,
                             const struct damacy_queue_limits* queues,
                             PyObject* dependencies);
