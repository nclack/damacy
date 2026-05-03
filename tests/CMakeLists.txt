if(NOT BUILD_TESTING)
    return()
endif()

# Each test_<name>.c is its own self-checking executable. Tests print
# OK/FAIL lines and exit nonzero on failure; ctest captures the exit
# code. add_damacy_test lives in cmake/Helpers.cmake.
add_damacy_test(test_json json)
add_damacy_test(test_lru lru)

if(DAMACY_FUZZ)
    add_subdirectory(fuzz)
endif()
