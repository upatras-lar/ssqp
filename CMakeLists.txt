cmake_minimum_required(VERSION 3.18)
project(ssqp VERSION 0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC -march=native)

# --- Options ---
option(WITH_HPIPM "Link against HPIPM and BLASFEO" OFF)
option(PINOCCHIO_STATIC "Use a static Pinocchio build" OFF)
option(ENABLE_PROFILING "Enable Tracy profiling" OFF)
option(DISABLE_LOGGING "Add -DDISABLE_LOGGING compile flag" OFF)
option(BUILD_EXAMPLES "Build example executables" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(BUILD_PYTHON "Build the Python extension" ON)

# --- Output directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Core library ---
file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
list(APPEND LIB_SOURCES "external/tracy/public/TracyClient.cpp")

add_library(ssqp STATIC ${LIB_SOURCES})

target_include_directories(ssqp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include
    SYSTEM PUBLIC
    /usr/include/eigen3
)

if(ENABLE_PROFILING)
    target_compile_options(ssqp PUBLIC -g -O2 -pthread -fno-omit-frame-pointer)
    target_compile_definitions(ssqp PUBLIC TRACY_ENABLE TRACY_MEMORY)
    target_link_libraries(ssqp PUBLIC pthread)
else()
    target_compile_options(ssqp PUBLIC -O3)
endif()

if(DISABLE_LOGGING)
    target_compile_definitions(ssqp PUBLIC DISABLE_LOGGING)
endif()

if(WITH_HPIPM)
    target_include_directories(ssqp SYSTEM PRIVATE
        /opt/blasfeo/include
        /opt/hpipm/include
    )
    target_link_directories(ssqp PUBLIC /opt/blasfeo/lib /opt/hpipm/lib)
    target_link_libraries(ssqp PRIVATE hpipm blasfeo)
    target_compile_definitions(ssqp PUBLIC WITH_HPIPM=1)
endif()

if(PINOCCHIO_STATIC)
    set(PINOCCHIO_STATIC_DIR "$ENV{HOME}/pinocchio-static")
    target_include_directories(ssqp SYSTEM PRIVATE ${PINOCCHIO_STATIC_DIR}/include)
    target_link_libraries(ssqp PRIVATE
        ${PINOCCHIO_STATIC_DIR}/lib/libpinocchio_default.a
        ${PINOCCHIO_STATIC_DIR}/lib/libpinocchio_parsers.a
    )
else()
    # Require Pinocchio 3.9.0 exactly; fails the configure step otherwise
    find_package(pinocchio 3.9.0 EXACT REQUIRED)

    find_library(PINOCCHIO_PARSERS_LIB NAMES pinocchio_parsers)
    if(NOT PINOCCHIO_PARSERS_LIB)
        message(FATAL_ERROR
            "Pinocchio 3.9.0 was found, but the pinocchio_parsers module could not be located. "
            "Rebuild Pinocchio with URDF/parsers support enabled (BUILD_WITH_URDF_SUPPORT=ON) "
            "before configuring ssqp.")
    endif()
    set(PINOCCHIO_PARSERS_TARGET ${PINOCCHIO_PARSERS_LIB})

    target_link_libraries(ssqp PRIVATE ${PINOCCHIO_PARSERS_TARGET})
endif()

# --- Examples ---
if(BUILD_EXAMPLES)
    file(GLOB_RECURSE EXAMPLE_SOURCES "examples/trajgen/c++/*.cpp")

    foreach(EXAMPLE_FILE ${EXAMPLE_SOURCES})
        file(RELATIVE_PATH REL_PATH
            "${CMAKE_CURRENT_SOURCE_DIR}/examples/trajgen/c++"
            ${EXAMPLE_FILE}
        )
        string(REGEX REPLACE "\\.cpp$" "" REL_BASE ${REL_PATH})
        string(REPLACE "/" "-" TARGET_NAME ${REL_BASE})
        get_filename_component(EXE_NAME ${REL_BASE} NAME)
        get_filename_component(EXE_DIR ${REL_BASE} DIRECTORY)

        add_executable(${TARGET_NAME} ${EXAMPLE_FILE})
        target_link_libraries(${TARGET_NAME} PRIVATE ssqp)

        if(EXE_DIR)
            set_target_properties(${TARGET_NAME} PROPERTIES
                OUTPUT_NAME ${EXE_NAME}
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${EXE_DIR}"
            )
        else()
            set_target_properties(${TARGET_NAME} PROPERTIES
                OUTPUT_NAME ${EXE_NAME}
                RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            )
        endif()
    endforeach()
endif()

# --- Tests ---
if(BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES "tests/src/*.cpp")
    add_executable(run_tests "tests/run_tests.cpp" ${TEST_SOURCES})

    target_include_directories(run_tests PRIVATE tests/include)
    target_link_libraries(run_tests PRIVATE ssqp)
    add_test(NAME ssqp_tests COMMAND run_tests)

    set_target_properties(run_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endif()

# --- Python bindings ---
if(BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import nanobind; print(nanobind.cmake_dir())"
        OUTPUT_VARIABLE nanobind_DIR_FOUND
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )

    set(nanobind_DIR ${nanobind_DIR_FOUND} CACHE PATH "nanobind CMake directory" FORCE)

    if(nanobind_DIR)
        find_package(nanobind REQUIRED)

        file(GLOB_RECURSE BINDINGS_SOURCES "python/*.cpp")
        nanobind_add_module(ssqpy ${BINDINGS_SOURCES})
        target_link_libraries(ssqpy PRIVATE ssqp)

        set_target_properties(ssqpy PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings"
        )

        add_custom_command(TARGET ssqpy POST_BUILD
            COMMAND ${Python_EXECUTABLE} -m nanobind.stubgen
            -m ssqpy
            -M ${CMAKE_BINARY_DIR}/bindings/py.typed
            -r
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bindings
            COMMENT "Generating Python stubs for ssqpy..."
        )
    else()
        message(WARNING "nanobind Python package not found. Skipping Python bindings.")
    endif()
endif()
