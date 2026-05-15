# FindCuFile.cmake — locate cufile.h. Damacy dlopen's libcufile, so we
# deliberately do not search for / require the library.
#
# Sets:
#   CuFile_FOUND        — TRUE on success
#   CuFile_INCLUDE_DIR  — directory containing cufile.h

set(_cufile_root_hints "")
if(DEFINED ENV{CUFILE_ROOT})
    list(APPEND _cufile_root_hints "$ENV{CUFILE_ROOT}")
endif()
if(DEFINED CUDAToolkit_ROOT)
    list(APPEND _cufile_root_hints "${CUDAToolkit_ROOT}")
endif()
if(DEFINED ENV{CUDA_PATH})
    list(APPEND _cufile_root_hints "$ENV{CUDA_PATH}")
endif()

find_path(
    CuFile_INCLUDE_DIR
    NAMES cufile.h
    HINTS ${_cufile_root_hints}
    PATH_SUFFIXES include targets/x86_64-linux/include
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CuFile REQUIRED_VARS CuFile_INCLUDE_DIR)

mark_as_advanced(CuFile_INCLUDE_DIR)
