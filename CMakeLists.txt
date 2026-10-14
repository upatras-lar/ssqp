cmake_minimum_required(VERSION 3.22)
project(ssqp VERSION 0.1.1 LANGUAGES CXX)

# Only affects single-config generators (Makefiles, Ninja); ssqp's own
# optimization level is set explicitly below regardless (see
# SSQP_ENABLE_PROFILING), so this is mainly so `cmake --install` doesn't report
# an empty configuration name when the user didn't pass -DCMAKE_BUILD_TYPE.
# Left alone when ssqp is added to another project, which owns this setting
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Turn off for binaries that must run on CPUs other than the build machine's
option(SSQP_MARCH_NATIVE "Compile with -march=native" ON)
set(SSQP_ARCH_FLAG "" CACHE STRING "Architecture flag used when SSQP_MARCH_NATIVE is OFF, e.g. -march=x86-64-v3")

if(SSQP_MARCH_NATIVE)
    set(SSQP_ARCH_FLAG "-march=native")
endif()
add_compile_options(-fPIC ${SSQP_ARCH_FLAG})

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# --- Options ---
# Tests and Python bindings default to off when ssqp is added to another
# project with add_subdirectory or FetchContent
option(SSQP_WITH_HPIPM "Link against HPIPM and BLASFEO" ON)
option(SSQP_ENABLE_PROFILING "Enable Tracy profiling" OFF)
option(SSQP_DISABLE_LOGGING "Add -DDISABLE_LOGGING compile flag" OFF)
option(SSQP_BUILD_EXAMPLES "Build example executables" OFF)
option(SSQP_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(SSQP_BUILD_PYTHON "Build the Python extension" ${PROJECT_IS_TOP_LEVEL})
option(SSQP_ENABLE_SANITIZERS "Enable address, undefined-behavior, and leak checks" OFF)

# If set, link against a prebuilt Pinocchio package (as produced by the
# `package-pinocchio` target) instead of fetching and building it from
# source. Such a package bundles its own Boost headers/static libs, so this
# also removes ssqp's Boost dependency. Must match the target architecture.
set(SSQP_PREBUILT_PINOCCHIO_DIR "" CACHE PATH "Path to a prebuilt Pinocchio package (see cmake/VendorPinocchio.cmake)")

# Same idea for BLASFEO/HPIPM (only relevant if SSQP_WITH_HPIPM is ON), as
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
# Same name as the installed package target (see C++ package below)
add_library(ssqp::ssqp ALIAS ssqp)

# Code using ssqp's headers must be compiled for the same architecture, or
# Eigen's alignment can differ between the two
target_compile_options(ssqp INTERFACE ${SSQP_ARCH_FLAG})

# --- Embedded URDF presets ---
# Bakes select URDFs into the binary so Model::Structure::from_preset can
# build a model without the caller supplying a urdf_path. Each preset only
# regenerates when its own URDF (or the generator script) changes, not on
# every configure.
set(SSQP_PRESET_NAMES go2 a1 panda g1 talos double_pendulum)
set(SSQP_PRESET_URDFS ${CMAKE_CURRENT_SOURCE_DIR}/examples/assets/urdf/go2/go2.urdf
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

# The installed headers keep the repository layout under include/ssqp, since
# the headers include the template implementations as "src/..."
include(GNUInstallDirs)
set(SSQP_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/ssqp)

target_include_directories(ssqp
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/tracy/public>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/json/single_include>
    $<INSTALL_INTERFACE:${SSQP_INSTALL_INCLUDEDIR}/include>
    $<INSTALL_INTERFACE:${SSQP_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${SSQP_INSTALL_INCLUDEDIR}/external/tracy/public>
    $<INSTALL_INTERFACE:${SSQP_INSTALL_INCLUDEDIR}/external/json/single_include>
    PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)
target_link_libraries(ssqp PUBLIC Eigen3::Eigen)

if(SSQP_ENABLE_SANITIZERS)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        message(FATAL_ERROR "SSQP_ENABLE_SANITIZERS requires GCC or Clang")
    endif()
    target_compile_options(ssqp PUBLIC -O1 -g -fno-omit-frame-pointer
        -fsanitize=address,undefined -fno-sanitize-recover=all)
    target_link_options(ssqp PUBLIC -fsanitize=address,undefined)
elseif(SSQP_ENABLE_PROFILING)
    target_compile_options(ssqp PUBLIC -g -O2 -pthread -fno-omit-frame-pointer)
    target_compile_definitions(ssqp PUBLIC TRACY_ENABLE TRACY_MEMORY)
    target_link_libraries(ssqp PUBLIC pthread)
else()
    # Private, so code using ssqp follows its own build type
    target_compile_options(ssqp PRIVATE -O3)
endif()

if(SSQP_DISABLE_LOGGING)
    target_compile_definitions(ssqp PUBLIC DISABLE_LOGGING)
endif()

# The vendored static libraries ssqp links, in link order, which are
# installed next to libssqp.a for the installed package
set(SSQP_VENDORED_LIBS "")

if(SSQP_WITH_HPIPM)
    include(VendorHpipm)
    target_link_libraries(ssqp PRIVATE $<BUILD_INTERFACE:hpipm_vendored>)
    target_compile_definitions(ssqp PUBLIC WITH_HPIPM=1)
    list(APPEND SSQP_VENDORED_LIBS ${SSQP_HPIPM_LIBS})
endif()

include(VendorPinocchio)
target_link_libraries(ssqp PRIVATE $<BUILD_INTERFACE:pinocchio_vendored>)
list(APPEND SSQP_VENDORED_LIBS -Wl,--start-group ${SSQP_PINOCCHIO_LIBS} -Wl,--end-group)

# --- Examples ---
if(SSQP_BUILD_EXAMPLES)
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
if(SSQP_BUILD_TESTS)
    enable_testing()

    file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS "tests/src/*.cpp")
    add_executable(run_tests "tests/run_tests.cpp" ${TEST_SOURCES})

    target_include_directories(run_tests PRIVATE tests/include)
    target_link_libraries(run_tests PRIVATE ssqp)
    # The tests instantiate ssqp's templates themselves, so they are
    # optimized like the library to keep them fast in any build type
    if(NOT SSQP_ENABLE_SANITIZERS AND NOT SSQP_ENABLE_PROFILING)
        target_compile_options(run_tests PRIVATE -O3)
    endif()
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
if(SSQP_BUILD_PYTHON)
    # SKBUILD_SABI_COMPONENT is set by scikit-build-core for stable ABI wheels
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT})

    include(VendorNanobind)

    file(GLOB_RECURSE BINDINGS_SOURCES "python/*.cpp")
    # Uses the stable ABI on Python 3.12+ when Development.SABIModule was found
    # above, so one wheel covers later versions
    nanobind_add_module(ssqpy STABLE_ABI ${BINDINGS_SOURCES})
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

    # `cmake --install . --component python` (after building) drops ssqpy straight into
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
    install(TARGETS ssqpy LIBRARY DESTINATION ${SSQP_PYTHON_INSTALL_DIR} COMPONENT python)
    install(FILES ${CMAKE_BINARY_DIR}/bindings/py.typed DESTINATION ${SSQP_PYTHON_INSTALL_DIR} COMPONENT python)
    install(DIRECTORY ${CMAKE_BINARY_DIR}/bindings/ssqpy/ DESTINATION ${SSQP_PYTHON_INSTALL_DIR}/ssqpy
        COMPONENT python)
endif()

# --- C++ package ---
# `cmake --install . --component cpp` installs libssqp.a, the vendored static
# libraries it needs, the headers and a CMake package under
# CMAKE_INSTALL_PREFIX, so other projects can use find_package(ssqp) and link
# ssqp::ssqp. Skipped for wheels, which only ship the Python module
if(NOT SKBUILD)
    include(CMakePackageConfigHelpers)

    set(SSQP_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ssqp)
    set(SSQP_INSTALL_VENDORDIR ${CMAKE_INSTALL_LIBDIR}/ssqp)

    # The installed package links the installed copies of the vendored
    # libraries instead of the ones in the build tree
    foreach(_item ${SSQP_VENDORED_LIBS})
        if(_item MATCHES "^-")
            target_link_libraries(ssqp INTERFACE $<INSTALL_INTERFACE:${_item}>)
        else()
            install(FILES ${_item} DESTINATION ${SSQP_INSTALL_VENDORDIR} COMPONENT cpp)
            get_filename_component(_name ${_item} NAME)
            target_link_libraries(ssqp INTERFACE
                $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${SSQP_INSTALL_VENDORDIR}/${_name}>)
        endif()
    endforeach()

    install(TARGETS ssqp EXPORT ssqpTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT cpp)
    install(DIRECTORY include/ DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/include COMPONENT cpp)
    install(DIRECTORY src/ DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/src COMPONENT cpp
        FILES_MATCHING PATTERN "*.tpp")
    install(DIRECTORY external/tracy/public/ DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/external/tracy/public
        COMPONENT cpp FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
    install(DIRECTORY external/json/single_include/
        DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/external/json/single_include COMPONENT cpp)
    install(FILES external/tracy/LICENSE DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/external/tracy COMPONENT cpp)
    install(FILES external/json/LICENSE.MIT DESTINATION ${SSQP_INSTALL_INCLUDEDIR}/external/json COMPONENT cpp)
    install(FILES LICENSE THIRD_PARTY_NOTICES.md DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT cpp)

    install(EXPORT ssqpTargets NAMESPACE ssqp:: DESTINATION ${SSQP_INSTALL_CMAKEDIR} COMPONENT cpp)
    configure_package_config_file(cmake/ssqpConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/ssqpConfig.cmake
        INSTALL_DESTINATION ${SSQP_INSTALL_CMAKEDIR})
    # The API is not stable yet, so only the same minor version is compatible
    write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/ssqpConfigVersion.cmake
        COMPATIBILITY SameMinorVersion)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/ssqpConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/ssqpConfigVersion.cmake
        DESTINATION ${SSQP_INSTALL_CMAKEDIR} COMPONENT cpp)
endif()
