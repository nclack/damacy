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
    damacy/_components.c
)

target_link_libraries(_native PRIVATE damacy log warnings)

# scikit-build-core installs Python.modules into the wheel by their
# install destination. damacy package lives at "damacy/" inside the
# wheel; the extension goes alongside the package's __init__.py.
install(TARGETS _native LIBRARY DESTINATION damacy)

set_target_properties(
    _native
    PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/damacy"
)
configure_file(damacy/__init__.py damacy/__init__.py COPYONLY)

if(BUILD_TESTING)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import pytest, pytest_cov"
        RESULT_VARIABLE PYTEST_AVAILABLE
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(PYTEST_AVAILABLE EQUAL 0)
        set(PYTHON_COVERAGE_ARGS
            --cov=damacy
            --cov-report=term-missing:skip-covered
        )
        if(DAMACY_CUDA)
            list(APPEND PYTHON_COVERAGE_ARGS --cov-fail-under=70)
        endif()
        add_test(
            NAME python_pytest
            COMMAND
                "${Python_EXECUTABLE}" -m pytest -v ${PYTHON_COVERAGE_ARGS}
                "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        )
        set_tests_properties(
            python_pytest
            PROPERTIES
                ENVIRONMENT
                    "WRITE_ZARR_SCRIPT=${PROJECT_SOURCE_DIR}/tests/write_zarr.py;PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
        )
    else()
        message(
            STATUS
            "damacy: install pytest and pytest-cov to enable python_pytest"
        )
    endif()
endif()
