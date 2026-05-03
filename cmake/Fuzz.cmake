# Fuzz.cmake — fuzz-mode option plumbing.
#
# Declares DAMACY_FUZZ and DAMACY_FUZZ_COVERAGE, and (when DAMACY_FUZZ
# is on) applies the sanitizer + coverage flags to the `warnings`
# interface target so every src lib inherits them. Requires
# Warnings.cmake to have been included first.
#
# DAMACY_FUZZ also gates: CUDA enable in the top-level CMakeLists.txt;
# bench/ via an early-return; decoder/pipeline via add_cuda_lib (see
# Helpers.cmake).
#
# Two build modes:
#   DAMACY_FUZZ=OFF (default) — full build: CUDA targets, bench, all tests.
#   DAMACY_FUZZ=ON            — minimal slice (json + slice + tests/fuzz),
#                               clang-only, sanitizers + libFuzzer
#                               instrumentation. No CUDA, no bench.

option(DAMACY_FUZZ "Build libFuzzer harnesses for util/json (clang only)" OFF)
option(
    DAMACY_FUZZ_COVERAGE
    "Add llvm source-based coverage to the fuzz build"
    OFF
)

if(NOT DAMACY_FUZZ)
    return()
endif()

# Every src target inherits these via the `warnings` interface lib.
# -fsanitize=fuzzer-no-link gives libFuzzer-compatible coverage
# instrumentation without injecting a main symbol; the harness
# executables themselves add -fsanitize=fuzzer (see Helpers.cmake)
# to pull in the runtime + main.
target_compile_options(
    warnings
    INTERFACE
        -fsanitize=fuzzer-no-link,address,undefined
        -fno-omit-frame-pointer
)
target_link_options(warnings INTERFACE -fsanitize=address,undefined)

if(DAMACY_FUZZ_COVERAGE)
    target_compile_options(
        warnings
        INTERFACE -fprofile-instr-generate -fcoverage-mapping
    )
    target_link_options(
        warnings
        INTERFACE -fprofile-instr-generate -fcoverage-mapping
    )
endif()
