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

add_subdirectory(src)
add_subdirectory(bench)

# Research playground for the assemble kernel — non-default targets
# (asmbench, damacy_bench_capture). Built explicitly via
# `cmake --build build --target asmbench` etc.
if(NOT DAMACY_FUZZ)
    add_subdirectory(experiments/asm)
endif()

include(CTest)
add_subdirectory(tests)
