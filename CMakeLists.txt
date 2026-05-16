cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0144 NEW)
project(damacy LANGUAGES C CXX)

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(Warnings)
include(Helpers)
include(Fuzz) # declares DAMACY_FUZZ and applies fuzz-mode flags

find_package(Threads REQUIRED)

if(NOT DAMACY_FUZZ)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Nvcomp REQUIRED)
endif()

option(
    DAMACY_PYTHON
    "Build the Python extension module (python/damacy/_native)"
    OFF
)

# NUMA-aware pinned-host allocation + io_queue/scheduler thread affinity
# is built unconditionally on Linux: libnuma is dlopen'd at runtime so a
# single wheel/binary works on hosts with or without libnuma installed.
# The runtime probe in src/numa/numa.c falls back to a no-op when
# libnuma.so.1 isn't loadable.

# DAMACY_NVTX_ENABLED — Phase 6 timeline observability. ON wires the
# nvtx/ wrappers to NVTX v3 (CUDA::nvtx3, header-only) so Nsight Systems
# sees per-wave ranges and named streams. OFF compiles call sites to
# inline no-ops via static inlines in src/nvtx/nvtx.h. Defaults ON for
# non-fuzz builds; src/CMakeLists.txt forces OFF under DAMACY_FUZZ.
option(DAMACY_NVTX_ENABLED "Enable NVTX timeline instrumentation" ON)

# DAMACY_ENABLE_GDS — gates the cuFile/GDS reader backend
# (src/store/store_fs_gds.c). Defaults OFF so developers/CI without
# cufile.h can still configure; flip ON to build the GDS path.
option(DAMACY_ENABLE_GDS "Build cuFile/GDS reader backend" OFF)

# cufile.h is needed at compile time for typedefs; libcufile.so.0 is
# dlopen'd at runtime (see src/store/store_fs_gds.c), not linked. Gated
# on DAMACY_ENABLE_GDS so non-GDS builds don't require cufile.h.
if(NOT DAMACY_FUZZ AND DAMACY_ENABLE_GDS)
    find_package(CuFile REQUIRED)
endif()

# DAMACY_COVERAGE — gcov-style instrumentation on the `warnings` interface
# target so every src lib/test inherits --coverage. Intended for CI; not
# compatible with DAMACY_FUZZ (which uses clang source-based coverage).
option(DAMACY_COVERAGE "Build with gcov coverage instrumentation" OFF)
if(DAMACY_COVERAGE AND NOT DAMACY_FUZZ)
    target_compile_options(
        warnings
        INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:--coverage;-O0;-g>
    )
    target_link_options(warnings INTERFACE --coverage)
endif()

add_subdirectory(src)
add_subdirectory(bench)

# Research playground for the assemble kernel — non-default targets
# (asmbench, damacy_bench_capture). Built explicitly via
# `cmake --build build --target asmbench` etc. The directory may be
# pruned from build contexts that don't ship experiments (e.g. Docker).
if(
    NOT DAMACY_FUZZ
    AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/experiments/asm/CMakeLists.txt"
)
    add_subdirectory(experiments/asm)
endif()

# CTest before python/ so python/CMakeLists.txt can register its
# pytest target alongside the C tests under a single BUILD_TESTING gate.
include(CTest)

if(DAMACY_PYTHON)
    add_subdirectory(python)
endif()

add_subdirectory(tests)
