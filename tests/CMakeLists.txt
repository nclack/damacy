# Each test_<name>.c is its own self-checking executable. Tests print
# OK/FAIL lines and exit nonzero on failure; ctest captures the exit code.
function(add_damacy_test name)
    add_executable(${name} ${name}.c)
    target_link_libraries(${name} PRIVATE warnings ${ARGN})
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_damacy_test(test_json json)
