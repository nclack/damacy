if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(test_cold_read test_cold_read.c ../cold_read.c)
    target_include_directories(test_cold_read PRIVATE ..)
    target_link_libraries(test_cold_read PRIVATE json slice warnings)
    add_test(NAME test_cold_read COMMAND test_cold_read)
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(
        NAME test_benchmark_cpu
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark.py
                $<TARGET_FILE:damacy_bench> cpu
    )
    set_tests_properties(test_benchmark_cpu PROPERTIES TIMEOUT 45)
    if(DAMACY_CUDA)
        add_test(
            NAME test_benchmark_cuda
            COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_benchmark.py
                    $<TARGET_FILE:damacy_bench> cuda
        )
        set_tests_properties(test_benchmark_cuda PROPERTIES TIMEOUT 45)
    endif()
endif()
