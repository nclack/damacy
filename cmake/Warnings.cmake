# Warnings.cmake — shared warning flags via INTERFACE target.
# Creates: warnings (INTERFACE library)

add_library(warnings INTERFACE)
target_compile_options(
    warnings
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)
