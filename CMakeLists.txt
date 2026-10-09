cmake_minimum_required(VERSION 3.22)
project(ssqp VERSION 0.0 LANGUAGES CXX)

# Only affects single-config generators (Makefiles, Ninja); ssqp's own
# optimization level is set explicitly below regardless (see
# ENABLE_PROFILING), so this is mainly so `cmake --install` doesn't report
# an empty configuration name when the user didn't pass -DCMAKE_BUILD_TYPE.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Turn off for binaries that must run on CPUs other than the build machine's
option(SSQP_MARCH_NATIVE "Compile with -march=native" ON)

if(SSQP_MARCH_NATIVE)
    set(SSQP_ARCH_FLAG "-march=native")
else()
    set(SSQP_ARCH_FLAG "")
endif()
add_compile_options(-fPIC ${SSQP_ARCH_FLAG})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# --- Options ---
option(WITH_HPIPM "Link against HPIPM and BLASFEO" ON)
option(ENABLE_PROFILING "Enable Tracy profiling" OFF)
option(DISABLE_LOGGING "Add -DDISABLE_LOGGING compile flag" OFF)
option(BUILD_EXAMPLES "Build example executables" OFF)
option(BUILD_TESTS "Build tests" ON)
option(BUILD_PYTHON "Build the Python extension" ON)
option(ENABLE_SANITIZERS "Enable address, undefined-behavior, and leak checks" OFF)

# If set, link against a prebuilt Pinocchio package (as produced by the
# `package-pinocchio` target) instead of fetching and building it from
# source. Such a package bundles its own Boost headers/static libs, so this
# also removes ssqp's Boost dependency. Must match the target architecture.
set(SSQP_PREBUILT_PINOCCHIO_DIR "" CACHE PATH "Path to a prebuilt Pinocchio package (see cmake/VendorPinocchio.cmake)")

# Same idea for BLASFEO/HPIPM (only relevant if WITH_HPIPM is ON), as
# produced by the `package-hpipm` target.
set(SSQP_PREBUILT_HPIPM_DIR "" CACHE PATH "Path to a prebuilt HPIPM/BLASFEO package (see cmake/VendorHpipm.cmake)")

# BLASFEO kernel selection when building it from source (see
# cmake/VendorHpipm.cmake)
option(SSQP_BLASFEO_AUTO_TARGET "Let BLASFEO pick its kernels for the build machine" ON)
set(SSQP_BLASFEO_TARGET "GENERIC" CACHE STRING "BLASFEO target used when SSQP_BLASFEO_AUTO_TARGET is OFF")

# --- Output directories ---
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# --- Core library ---
find_package(Eigen3 REQUIRED NO_MODULE)

file(GLOB_RECURSE LIB_SOURCES "src/*.cpp")
list(APPEND LIB_SOURCES "external/tracy/public/TracyClient.cpp")

add_library(ssqp STATIC ${LIB_SOURCES})

# --- Embedded URDF presets ---
# Bakes select URDFs into the binary so Model::Structure::from_preset can
# build a model without the caller supplying a urdf_path. Each preset only
# regenerates when its own URDF (or the generator script) changes, not on
# every configure.
set(SSQP_PRESET_NAMES go2 a1 panda g1 talos double_pendulum)
set(SSQP_PRESET_URDFS  ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/go2/go2.urdf
                       ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/a1/a1.urdf
                       ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/panda/panda.urdf
                       ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/g1/g1_23dof.urdf
                       ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/talos/talos.urdf
                       ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/double_pendulum/double_pendulum.urdf)
set(SSQP_PRESET_SYMBOLS go2_urdf a1_urdf panda_urdf g1_urdf talos_urdf double_pendulum_urdf)

list(LENGTH SSQP_PRESET_NAMES SSQP_PRESET_COUNT)
math(EXPR SSQP_PRESET_LAST_INDEX "${SSQP_PRESET_COUNT} - 1")

set(SSQP_PRESET_URDF_HEADERS "")
foreach(_i RANGE ${SSQP_PRESET_LAST_INDEX})
    list(GET SSQP_PRESET_NAMES ${_i} _name)
    list(GET SSQP_PRESET_URDFS ${_i} _urdf)
    list(GET SSQP_PRESET_SYMBOLS ${_i} _symbol)
    set(_header "${CMAKE_CURRENT_BINARY_DIR}/generated/model/presets/${_name}_urdf.hpp")

    add_custom_command(
        OUTPUT ${_header}
        COMMAND ${CMAKE_COMMAND}
        -DIN_URDF=${_urdf}
        -DOUT_HEADER=${_header}
        -DSYMBOL=${_symbol}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateEmbeddedUrdf.cmake
        DEPENDS ${_urdf} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateEmbeddedUrdf.cmake
        COMMENT "Embedding ${_name} URDF preset"
    )
    list(APPEND SSQP_PRESET_URDF_HEADERS ${_header})
endforeach()

add_custom_target(ssqp_generate_presets DEPENDS ${SSQP_PRESET_URDF_HEADERS})
add_dependencies(ssqp ssqp_generate_presets)

target_include_directories(ssqp
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public
    ${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(ssqp PUBLIC Eigen3::Eigen)

if(ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    target_compile_options(ssqp PUBLIC -O1 -g -fno-omit-frame-pointer
        -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_options(ssqp PUBLIC -fsanitize=address,undefined)
elseif(ENABLE_PROFILING)
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
    include(VendorHpipm)
    target_link_libraries(ssqp PRIVATE hpipm_vendored)
    target_compile_definitions(ssqp PUBLIC WITH_HPIPM=1)
endif()

include(VendorPinocchio)
target_link_libraries(ssqp PRIVATE pinocchio_vendored)

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

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/src/*.cpp")
    add_executable(run_tests "tests/run_tests.cpp" ${TEST_SOURCES})

    target_include_directories(run_tests PRIVATE tests/include)
    target_link_libraries(run_tests PRIVATE ssqp)
    # Discover at CTest time so new cases require no manually maintained list.
    # Each case runs in its own process: a segfault does not hide later cases.
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/discover_ssqp_tests.cmake"
        CONTENT "set(SSQP_TEST_BINARY \"$<TARGET_FILE:run_tests>\")\nset(SSQP_TEST_SOURCE \"${CMAKE_CURRENT_SOURCE_DIR}\")\ninclude(\"${CMAKE_CURRENT_SOURCE_DIR}/tests/DiscoverTests.cmake\")\n")
    set_property(DIRECTORY APPEND PROPERTY TEST_INCLUDE_FILES
        "${CMAKE_CURRENT_BINARY_DIR}/discover_ssqp_tests.cmake")

    set_target_properties(run_tests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
    )
endif()

# --- Python bindings ---
if(BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

    include(VendorNanobind)

    file(GLOB_RECURSE BINDINGS_SOURCES "python/*.cpp")
    nanobind_add_module(ssqpy ${BINDINGS_SOURCES})
    target_link_libraries(ssqpy PRIVATE ssqp)

    set_target_properties(ssqpy PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings"
    )

    # The Python extension is a shared object, so it necessarily keeps the
    # platform C/C++ runtime dependencies. This target makes sure that the
    # libraries owned by SSQP and its vendored dependencies remain static.
    find_program(SSQP_READELF_EXECUTABLE readelf)
    if(SSQP_READELF_EXECUTABLE)
        add_custom_target(verify-static-dependencies
            COMMAND ${CMAKE_COMMAND}
            -DSSQP_READELF_EXECUTABLE=${SSQP_READELF_EXECUTABLE}
            -DSSQP_PYTHON_MODULE=$<TARGET_FILE:ssqpy>
            -DSSQP_STATIC_LIBRARY=$<TARGET_FILE:ssqp>
            -DSSQP_AR=${CMAKE_AR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyStaticDependencies.cmake
            DEPENDS ssqpy
            COMMENT "Checking SSQP and Python dependencies are static"
            VERBATIM
        )
    else()
        message(WARNING
            "readelf was not found; verify-static-dependencies is unavailable")
    endif()

    # Keep the caller's PYTHONPATH, which pip's isolated build environment
    # uses to expose its build requirements (e.g. NumPy)
    set(SSQP_STUBGEN_PYTHONPATH ${NB_STUBGEN_PYTHONPATH})
    if(DEFINED ENV{PYTHONPATH})
        string(APPEND SSQP_STUBGEN_PYTHONPATH ":$ENV{PYTHONPATH}")
    endif()

    add_custom_command(TARGET ssqpy POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${SSQP_STUBGEN_PYTHONPATH}" --
        ${Python_EXECUTABLE} -m nanobind.stubgen
        -m ssqpy
        -M ${CMAKE_BINARY_DIR}/bindings/py.typed
        -r
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bindings
        COMMENT "Generating Python stubs for ssqpy..."
    )

    # `cmake --install .` (after building) drops ssqpy straight into
    # site-packages for whichever Python interpreter it was built against
    # (Python_SITEARCH is an absolute path, so this installs there directly
    # regardless of CMAKE_INSTALL_PREFIX), so it's importable from a normal
    # Python environment without manually managing PYTHONPATH. When built as
    # a wheel by scikit-build-core, the install prefix is the wheel root
    if(SKBUILD)
        set(SSQP_PYTHON_INSTALL_DIR .)
    else()
        set(SSQP_PYTHON_INSTALL_DIR ${Python_SITEARCH})
    endif()
    install(TARGETS ssqpy LIBRARY DESTINATION ${SSQP_PYTHON_INSTALL_DIR})
    install(FILES ${CMAKE_BINARY_DIR}/bindings/py.typed DESTINATION ${SSQP_PYTHON_INSTALL_DIR})
    install(DIRECTORY ${CMAKE_BINARY_DIR}/bindings/ssqpy/ DESTINATION ${SSQP_PYTHON_INSTALL_DIR}/ssqpy)
endif()
