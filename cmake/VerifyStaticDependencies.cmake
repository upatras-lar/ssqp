# Verify that the built Python extension does not pull in vendored or Boost
# shared libraries. The platform C++ and C runtimes remain shared on normal
# Linux installations and are intentionally allowed.

if(NOT DEFINED SSQP_READELF_EXECUTABLE OR NOT EXISTS "${SSQP_READELF_EXECUTABLE}")
    message(FATAL_ERROR "readelf is required to verify ssqp's dynamic dependencies")
endif()

if(NOT DEFINED SSQP_PYTHON_MODULE OR NOT EXISTS "${SSQP_PYTHON_MODULE}")
    message(FATAL_ERROR "Python module does not exist: ${SSQP_PYTHON_MODULE}")
endif()

if(NOT DEFINED SSQP_STATIC_LIBRARY OR NOT EXISTS "${SSQP_STATIC_LIBRARY}")
    message(FATAL_ERROR "Static ssqp library does not exist: ${SSQP_STATIC_LIBRARY}")
endif()

if(NOT DEFINED SSQP_AR OR NOT EXISTS "${SSQP_AR}")
    message(FATAL_ERROR "The archiver is required to inspect ${SSQP_STATIC_LIBRARY}")
endif()

execute_process(
    COMMAND "${SSQP_READELF_EXECUTABLE}" -d "${SSQP_PYTHON_MODULE}"
    RESULT_VARIABLE _readelf_result
    OUTPUT_VARIABLE _dynamic_section
    ERROR_VARIABLE _readelf_error
)
if(NOT _readelf_result EQUAL 0)
    message(FATAL_ERROR
        "Could not inspect ${SSQP_PYTHON_MODULE}: ${_readelf_error}")
endif()

# A static archive has no ELF dynamic section by definition. Check its file
# type separately so the target also guards against accidentally changing the
# main ssqp target to SHARED.
execute_process(
    COMMAND "${SSQP_AR}" t "${SSQP_STATIC_LIBRARY}"
    RESULT_VARIABLE _ar_result
    OUTPUT_VARIABLE _archive_members
    ERROR_VARIABLE _ar_error
)
if(NOT _ar_result EQUAL 0 OR NOT _archive_members)
    message(FATAL_ERROR
        "${SSQP_STATIC_LIBRARY} is not a readable static archive: ${_ar_error}")
endif()

set(_forbidden_shared_regex
    "lib(boost_[^]]*|pinocchio[^]]*|urdfdom[^]]*|hpipm[^]]*|blasfeo[^]]*)\\.so")
if(_dynamic_section MATCHES "${_forbidden_shared_regex}")
    message(FATAL_ERROR
        "ssqpy links a non-static SSQP dependency:\n${CMAKE_MATCH_0}")
endif()

message(STATUS
    "Static dependency check passed: ${SSQP_STATIC_LIBRARY} and ${SSQP_PYTHON_MODULE}")
