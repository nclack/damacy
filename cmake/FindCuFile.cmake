# FindCuFile.cmake — locate cufile.h + libcufile.
#
# Search order:
#   1. CUFILE_ROOT environment variable (nix/devShell, vendored installs)
#   2. CUDAToolkit_ROOT / CUDA_PATH
#   3. Compiler default search paths
#
# Sets:
#   CuFile_FOUND        — TRUE on success
#   CuFile_INCLUDE_DIR  — directory containing cufile.h
#   CuFile_LIBRARY      — full path to libcufile.so
#   CuFile::cufile      — imported target

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

find_library(
    CuFile_LIBRARY
    NAMES cufile
    HINTS ${_cufile_root_hints}
    PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    CuFile
    REQUIRED_VARS CuFile_INCLUDE_DIR CuFile_LIBRARY
)

if(CuFile_FOUND AND NOT TARGET CuFile::cufile)
    add_library(CuFile::cufile UNKNOWN IMPORTED)
    set_target_properties(
        CuFile::cufile
        PROPERTIES
            IMPORTED_LOCATION "${CuFile_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CuFile_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CuFile_INCLUDE_DIR CuFile_LIBRARY)
