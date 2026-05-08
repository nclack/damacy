# Helpers.cmake — project-specific helper functions, kept in one
# CMAKE_MODULE_PATH-discoverable place so static analyzers and LSPs find
# them and so they don't have to be redefined inside nested
# CMakeLists.txt files.
#
# Provides:
#   add_src_lib(TARGET SOURCES src... [LINKS lib...])
#   add_platform_sources(TARGET [BASENAME])
#   add_damacy_test(NAME [LINKS...])
#   add_damacy_fuzzer(NAME)
#
# Requires `warnings` to already be defined (see Warnings.cmake).

# Static library + warnings + src/ as the public include root.
function(add_src_lib TARGET)
    cmake_parse_arguments(ARG "" "" "SOURCES;LINKS" ${ARGN})
    add_library(${TARGET} STATIC ${ARG_SOURCES})
    target_include_directories(${TARGET} PUBLIC ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(${TARGET} PRIVATE warnings)
    if(ARG_LINKS)
        target_link_libraries(${TARGET} PUBLIC ${ARG_LINKS})
    endif()
endfunction()

# Same as add_src_lib but a no-op under DAMACY_FUZZ (fuzz mode doesn't
# enable CUDA, so any CUDA-linking target would fail to configure).
function(add_cuda_lib TARGET)
    if(DAMACY_FUZZ)
        return()
    endif()
    add_src_lib(${TARGET} ${ARGN})
endfunction()

# Appends basename.win32.c or basename.posix.c (or .darwin.c if it exists)
# to TARGET's sources. BASENAME defaults to TARGET.
function(add_platform_sources TARGET)
    if(ARGC GREATER 1)
        set(BASE ${ARGV1})
    else()
        set(BASE ${TARGET})
    endif()
    if(WIN32)
        target_sources(${TARGET} PRIVATE ${BASE}.win32.c)
    elseif(APPLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${BASE}.darwin.c")
        target_sources(${TARGET} PRIVATE ${BASE}.darwin.c)
    else()
        target_sources(${TARGET} PRIVATE ${BASE}.posix.c)
    endif()
endfunction()

# tests/test_<NAME>.c → self-checking executable registered with ctest.
function(add_damacy_test NAME)
    add_executable(${NAME} ${NAME}.c)
    target_link_libraries(${NAME} PRIVATE warnings ${ARGN})
    add_test(NAME ${NAME} COMMAND ${NAME})
endfunction()

# tests/fuzz/<NAME>.c → libFuzzer harness with sanitizer flags. Extra
# args are forwarded as PRIVATE link libraries so each harness pulls in
# only the targets it exercises.
function(add_damacy_fuzzer NAME)
    set(_FUZZ_FLAGS
        -fsanitize=fuzzer,address,undefined
        -fno-omit-frame-pointer
        -g
        -O1
    )
    add_executable(${NAME} ${NAME}.c)
    target_link_libraries(${NAME} PRIVATE warnings ${ARGN})
    target_compile_options(${NAME} PRIVATE ${_FUZZ_FLAGS})
    target_link_options(${NAME} PRIVATE ${_FUZZ_FLAGS})
endfunction()
