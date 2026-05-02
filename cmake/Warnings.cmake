# Warnings.cmake — shared warning flags via INTERFACE target
#
# Creates: warnings (INTERFACE library)
# Provides: add_platform_sources(TARGET [BASENAME])

add_library(warnings INTERFACE)
target_compile_options(
    warnings
    INTERFACE
        $<$<COMPILE_LANGUAGE:C>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options=-Wall,-Wextra>
)

# add_platform_sources(TARGET [BASENAME])
#   Appends basename.win32.c or basename.posix.c to the target's sources.
#   BASENAME defaults to TARGET.
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
