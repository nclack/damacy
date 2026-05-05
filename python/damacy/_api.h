#pragma once

// Internal: registers the Damacy and Batch types on a module object.
// Returns 0 on success, -1 with a Python error set on failure.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#ifdef __cplusplus
extern "C"
{
#endif

  int api_register_types(PyObject* module);

#ifdef __cplusplus
}
#endif
