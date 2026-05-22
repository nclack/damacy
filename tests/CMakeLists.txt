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
add_damacy_test(test_damacy_config damacy_config log)
add_damacy_test(test_lru lru hash)
add_damacy_test(test_pool pool log platform Threads::Threads)
add_damacy_test(test_io_queue io_queue log)
add_damacy_test(test_store_fs store lru hash log)
target_link_libraries(test_store_fs PRIVATE Threads::Threads)
add_damacy_test(test_zarr_meta_cache zarr_meta_cache zarr store test_fixture)
add_damacy_test(test_zarr_shard_cache zarr_shard_cache zarr store test_fixture)

# Concurrent stress test for the LRU caches. Designed to surface
# unsynchronized access to lru_get / lru_put / lru_stats_get; under
# DAMACY_TSAN=ON, ThreadSanitizer reports the race directly. Labeled
# `tsan` so CI can run only this test under the tsan build.
add_damacy_test(
    test_zarr_cache_threading
    zarr_meta_cache
    zarr_shard_cache
    zarr
    store
    test_fixture
)
target_link_libraries(test_zarr_cache_threading PRIVATE Threads::Threads)
set_tests_properties(test_zarr_cache_threading PROPERTIES LABELS tsan)
add_damacy_test(test_planner planner zarr_meta_cache zarr_shard_cache zarr store path_intern test_fixture)
add_damacy_test(test_coalesce planner path_intern log)
add_damacy_test(test_path_intern path_intern hash log)
add_damacy_test(test_group_chunks planner log)
add_damacy_test(test_chunk_layout planner zarr_chunk_layout zarr_meta_cache zarr_shard_cache zarr store path_intern test_fixture)
add_damacy_test(test_lookahead lookahead log platform)
add_damacy_test(test_prefetch_cache prefetch_cache log)
add_damacy_test(test_array_meta array_meta prefetch_cache zarr store test_fixture)
add_damacy_test(test_shard_index shard_index array_meta prefetch_cache zarr store hash test_fixture)
add_damacy_test(test_chunk_layout_cache chunk_layout shard_index array_meta prefetch_cache zarr_chunk_layout zarr store hash test_fixture)
add_damacy_test(test_prefetcher prefetcher chunk_layout shard_index array_meta prefetch_cache zarr_chunk_layout zarr store hash platform test_fixture)
add_damacy_test(test_sample_shard_iterator zarr log)
add_damacy_test(test_scheduler scheduler platform log)

# End-to-end smoke test: requires CUDA + nvCOMP, so guarded against
# DAMACY_FUZZ (which disables the CUDA-linked damacy target). Also
# requires `uv` on PATH at test time (provided by flake.nix devShell).
if(NOT DAMACY_FUZZ)
    add_damacy_test(test_damacy damacy test_fixture)
    target_sources(test_damacy PRIVATE spin_kernel.cu)
    target_link_libraries(test_damacy PRIVATE CUDA::cudart_static)
    add_damacy_test(test_damacy_blosc damacy test_fixture)
    add_damacy_test(test_damacy_caps damacy decoder test_fixture CUDA::cuda_driver)
    add_damacy_test(test_damacy_ctx damacy test_fixture CUDA::cuda_driver)
    add_damacy_test(test_batch_pool batch_pool log CUDA::cuda_driver)
    add_damacy_test(test_wave_pool wave_pool log CUDA::cuda_driver)
    add_damacy_test(test_gpu_budget wave_budget log CUDA::cuda_driver)
    add_damacy_test(test_assemble assemble CUDA::cuda_driver CUDA::cudart_static)
    add_damacy_test(test_decoder decoder CUDA::cuda_driver)
    if(DAMACY_ENABLE_GDS)
        add_damacy_test(test_store_fs_gds store lru hash log CUDA::cuda_driver)
    endif()
endif()

if(DAMACY_FUZZ)
    add_subdirectory(fuzz)
endif()
