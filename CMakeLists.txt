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

find_package(Threads REQUIRED)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(Nvcomp REQUIRED)

add_subdirectory(src)
add_subdirectory(bench)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
