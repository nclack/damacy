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
add_damacy_test(test_io_queue io_queue log)
add_damacy_test(test_zarr_meta_cache zarr_meta_cache zarr store test_fixture)
add_damacy_test(test_zarr_shard_cache zarr_shard_cache zarr store test_fixture)
add_damacy_test(test_planner planner zarr_meta_cache zarr_shard_cache zarr store test_fixture)
add_damacy_test(test_lookahead lookahead log)

# End-to-end smoke test: requires CUDA + nvCOMP, so guarded against
# DAMACY_FUZZ (which disables the CUDA-linked damacy target). Also
# requires `uv` on PATH at test time (provided by flake.nix devShell).
if(NOT DAMACY_FUZZ)
    add_damacy_test(test_damacy damacy test_fixture)
    add_damacy_test(test_damacy_blosc damacy test_fixture)
    add_damacy_test(test_damacy_caps damacy decoder test_fixture CUDA::cuda_driver)
    add_damacy_test(test_damacy_ctx damacy test_fixture CUDA::cuda_driver)
    add_damacy_test(test_batch_pool batch_pool log CUDA::cuda_driver)
    add_damacy_test(test_gpu_budget gpu_budget log CUDA::cuda_driver)
    add_damacy_test(test_assemble assemble CUDA::cuda_driver CUDA::cudart_static)
    add_damacy_test(test_decoder decoder CUDA::cuda_driver)
    add_damacy_test(test_blosc1 blosc1_host decoder threadpool zarr CUDA::cuda_driver)
    set(_BLOSC1_DIR ${PROJECT_SOURCE_DIR}/experiments/blosc1-gpu)
    target_compile_definitions(
        test_blosc1
        PRIVATE
            BLOSC1_FIXTURES_DIR="${_BLOSC1_DIR}"
            BLOSC1_GEN_FIXTURE_SCRIPT="${_BLOSC1_DIR}/gen_fixture.py"
    )
endif()

if(DAMACY_FUZZ)
    add_subdirectory(fuzz)
endif()
