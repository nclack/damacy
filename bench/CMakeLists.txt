# bench/ links damacy, which becomes a CUDA-linked target. add_cuda_lib
# is a no-op under DAMACY_FUZZ, so bail out early in that mode.
if(DAMACY_FUZZ)
    return()
endif()

add_executable(damacy_bench main.c)
# Threads::Threads for the watchdog thread in main.c.
find_package(Threads REQUIRED)
target_link_libraries(
    damacy_bench
    PRIVATE damacy json json_writer slice strbuf warnings Threads::Threads
)
# nvcomp_static drags in libstdc++/libgcc_s; fold them into the binary too.
target_link_options(damacy_bench PRIVATE -static-libstdc++ -static-libgcc)
