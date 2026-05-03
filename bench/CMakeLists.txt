# bench/ depends on CUDA-linked targets (pipeline, decoder), which are
# skipped under DAMACY_FUZZ. Bail out early in that mode.
if(DAMACY_FUZZ)
    return()
endif()

add_executable(damacy_bench main.c)
target_link_libraries(damacy_bench PRIVATE damacy warnings)
# nvcomp_static drags in libstdc++/libgcc_s; fold them into the binary too.
target_link_options(damacy_bench PRIVATE -static-libstdc++ -static-libgcc)
