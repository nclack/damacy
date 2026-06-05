# bench/ links damacy, which becomes a CUDA-linked target. add_cuda_lib
# is a no-op under DAMACY_FUZZ, so bail out early in that mode.
if(DAMACY_FUZZ)
    return()
endif()

add_executable(damacy_bench main.c)
target_link_libraries(
    damacy_bench
    PRIVATE damacy json json_writer slice strbuf warnings
)
# nvcomp_static drags in libstdc++/libgcc_s; fold them into the binary too.
target_link_options(damacy_bench PRIVATE -static-libstdc++ -static-libgcc)

add_executable(metadata_store_async_bench metadata_store_async_bench.c)
target_link_libraries(
    metadata_store_async_bench
    PRIVATE metadata_store_async warnings Threads::Threads
)
