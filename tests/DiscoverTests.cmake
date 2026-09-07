if(NOT EXISTS "${SSQP_TEST_BINARY}")
    add_test(ssqp_tests_not_built "${SSQP_TEST_BINARY}")
    return()
endif()
execute_process(COMMAND "${SSQP_TEST_BINARY}" --list
    WORKING_DIRECTORY "${SSQP_TEST_SOURCE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE cases ERROR_VARIABLE errors
    TIMEOUT 60)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "SSQP test discovery failed (${result}): ${errors}")
endif()
string(STRIP "${cases}" cases)
string(REPLACE "\n" ";" cases "${cases}")
foreach(case IN LISTS cases)
    add_test("ssqp.${case}" "${SSQP_TEST_BINARY}" --filter "${case}")
    set_tests_properties("ssqp.${case}" PROPERTIES
        WORKING_DIRECTORY "${SSQP_TEST_SOURCE}"
        TIMEOUT 300
        ENVIRONMENT "ASAN_OPTIONS=detect_leaks=1:halt_on_error=1;UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1")
endforeach()
# Confirm an unmatched filter is an error, so a misspelled selection cannot
# produce a green build with zero executed tests.
add_test(ssqp.runner_empty_selection "${SSQP_TEST_BINARY}" --filter __no_such_test__)
set_tests_properties(ssqp.runner_empty_selection PROPERTIES
    WORKING_DIRECTORY "${SSQP_TEST_SOURCE}" WILL_FAIL TRUE TIMEOUT 60)
