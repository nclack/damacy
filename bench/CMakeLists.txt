add_executable(damacy_bench main.c)
target_link_libraries(
    damacy_bench
    PRIVATE pipeline decoder zarr store warnings
)
# nvcomp_static drags in libstdc++/libgcc_s; fold them into the binary too.
target_link_options(damacy_bench PRIVATE -static-libstdc++ -static-libgcc)
