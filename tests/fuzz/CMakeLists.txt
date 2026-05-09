# libFuzzer harnesses for the host-side parsers. Built only when
# DAMACY_FUZZ=ON, which also gates out the CUDA-dependent slice of the
# tree (see top-level CMakeLists.txt). Requires clang — gcc has no
# libFuzzer. add_damacy_fuzzer lives in cmake/Helpers.cmake.
if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
    message(
        FATAL_ERROR
        "DAMACY_FUZZ requires the Clang C compiler (got ${CMAKE_C_COMPILER_ID}). "
        "Configure with -DCMAKE_C_COMPILER=clang or use the 'fuzz' preset."
    )
endif()

add_damacy_fuzzer(fuzz_json_input json)
add_damacy_fuzzer(fuzz_json_query json)
add_damacy_fuzzer(fuzz_shard_index zarr)
add_damacy_fuzzer(fuzz_zarr_metadata zarr)
add_damacy_fuzzer(fuzz_blosc1_host blosc1_host)
