# Python C extension. Driven by scikit-build-core (sets DAMACY_PYTHON=ON
# and arranges for the extension's install destination); also buildable
# from a developer cmake configure with -DDAMACY_PYTHON=ON.

find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

python_add_library(
    _native
    MODULE
    WITH_SOABI
    damacy/_native.c
    damacy/_log_sink.c
    damacy/_api.c
)

target_link_libraries(_native PRIVATE damacy log warnings)

# scikit-build-core installs Python.modules into the wheel by their
# install destination. damacy package lives at "damacy/" inside the
# wheel; the extension goes alongside the package's __init__.py.
install(TARGETS _native LIBRARY DESTINATION damacy)
