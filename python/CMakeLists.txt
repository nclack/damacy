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

# Pytest harness for the bindings. Gated on pytest being importable
# from the cmake-located Python interpreter — non-Python-dev
# environments build the extension without registering this test.
# The damacy package itself is imported at test runtime, so the test
# correctly fires after a build/editable-install ordering (see
# Dockerfile). WRITE_ZARR_SCRIPT mirrors the C-side fixture path.
if(BUILD_TESTING)
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pytest"
        RESULT_VARIABLE _damacy_have_pytest
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_damacy_have_pytest EQUAL 0)
        add_test(
            NAME python_pytest
            COMMAND
                ${Python_EXECUTABLE} -m pytest -v --color=yes
                "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(
            python_pytest
            PROPERTIES
                ENVIRONMENT
                    "WRITE_ZARR_SCRIPT=${PROJECT_SOURCE_DIR}/tests/write_zarr.py"
        )
    else()
        message(
            STATUS
            "damacy: pytest not importable from ${Python_EXECUTABLE}; "
            "skipping python_pytest registration"
        )
    endif()
endif()
