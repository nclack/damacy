if(NOT BUILD_TESTING)
    return()
endif()

# Shared fs-backed test scaffolding. expect.h (macros only) needs no
# library; fixture.c does, and brings in crc32c + zarr_shard_index_size
# transitively via zarr. fixture_write_zarr shells out to
# tests/write_zarr.py via uv (--script with PEP-723 inline deps); the
# script path is baked in via WRITE_ZARR_SCRIPT.
add_library(test_fixture STATIC fixture.c fixture.h expect.h)
target_include_directories(
    test_fixture
    PUBLIC ${PROJECT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_definitions(
    test_fixture
    PRIVATE WRITE_ZARR_SCRIPT="${CMAKE_CURRENT_SOURCE_DIR}/write_zarr.py"
)
target_link_libraries(test_fixture PRIVATE warnings)
target_link_libraries(test_fixture PUBLIC log crc32c zarr)

# Each test_<name>.c is its own self-checking executable. Tests print
# OK/FAIL lines and exit nonzero on failure; ctest captures the exit
# code. add_damacy_test lives in cmake/Helpers.cmake.
add_damacy_test(test_json json log)
add_damacy_test(test_lru lru hash)
add_damacy_test(test_zarr_meta_cache zarr_meta_cache zarr store test_fixture)
add_damacy_test(test_zarr_shard_cache zarr_shard_cache zarr store test_fixture)
add_damacy_test(test_planner planner zarr_meta_cache zarr_shard_cache zarr store test_fixture)

# End-to-end smoke test: requires CUDA + nvCOMP, so guarded against
# DAMACY_FUZZ (which disables the CUDA-linked damacy target). Also
# requires `uv` on PATH at test time (provided by flake.nix devShell).
if(NOT DAMACY_FUZZ)
    add_damacy_test(test_damacy damacy test_fixture)
    add_damacy_test(test_assemble assemble CUDA::cuda_driver CUDA::cudart_static)
endif()

if(DAMACY_FUZZ)
    add_subdirectory(fuzz)
endif()
