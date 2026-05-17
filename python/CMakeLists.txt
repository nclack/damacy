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

# --cov-fail-under floor matches codecov.yml's patch target so the two
# gates (project-total here, per-PR-patch on codecov) read the same number.
if(BUILD_TESTING)
    find_program(UV_EXECUTABLE uv)
    if(UV_EXECUTABLE)
        add_test(
            NAME python_pytest
            COMMAND
                ${UV_EXECUTABLE} run --extra dev pytest -v --color=yes
                --cov=damacy --cov-report=term-missing:skip-covered
                --cov-fail-under=70 "${CMAKE_CURRENT_SOURCE_DIR}/tests"
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
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
            "damacy: uv not on PATH; skipping python_pytest registration"
        )
    endif()
endif()
