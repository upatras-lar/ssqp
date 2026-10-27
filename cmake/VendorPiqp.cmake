# Provides the `piqp_vendored` INTERFACE target for the PIQP QP solver backend.
# PIQP is header-only, but its multistage KKT solver is only compiled in
# when PIQP_HAS_BLASFEO is defined, so BLASFEO is linked as well. The two
# parts come from:
#
# - PIQP: fetched as source (pinned to a specific commit, since its multistage
#   KKT solver only recovers from failed factorizations after the latest
#   release), or, if
#   SSQP_PREBUILT_PIQP_DIR is set, taken from the include/ directory of that
#   package. No PIQP library is built, and the template instantiation
#   sources are not used.
#
# - BLASFEO: shared with the HPIPM backend when SSQP_WITH_HPIPM is on, so it
#   is built and linked once. Otherwise it is built here, pinned to the same
#   commit as in cmake/VendorHpipm.cmake, or taken from lib/ and include/ of
#   SSQP_PREBUILT_PIQP_DIR. See that file for how SSQP_BLASFEO_AUTO_TARGET and
#   SSQP_BLASFEO_TARGET select the kernels.
#
# Without SSQP_PREBUILT_PIQP_DIR, the `package-piqp` target (see bottom of
# this file) assembles such a package. It holds the PIQP headers, and also
# BLASFEO when SSQP_WITH_HPIPM is off, so a package made with HPIPM on is
# only usable by builds with HPIPM on.
#
# Sets SSQP_PIQP_LIBS to the static libraries that are not already provided by
# the HPIPM backend.

if(NOT DEFINED SSQP_VENDOR_PREFIX_PIQP)
    set(SSQP_VENDOR_PREFIX_PIQP "${CMAKE_BINARY_DIR}/vendor-piqp" CACHE PATH "Install prefix for vendored BLASFEO used by PIQP")
endif()

add_library(piqp_vendored INTERFACE)
set(SSQP_PIQP_LIBS "")

# --- PIQP headers ---
if(SSQP_PREBUILT_PIQP_DIR)
    set(_ssqp_piqp_include_dir ${SSQP_PREBUILT_PIQP_DIR}/include)
else()
    include(FetchContent)

    # Only the headers are used, so the PIQP project itself is not added
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()

    FetchContent_Declare(ssqp_piqp_src
        GIT_REPOSITORY https://github.com/PREDICT-EPFL/piqp.git
        GIT_TAG 439f6ffb2c8c76f89ca09cd0b9db7bc2e071d54f
        UPDATE_DISCONNECTED TRUE
    )
    FetchContent_Populate(ssqp_piqp_src)
    set(_ssqp_piqp_include_dir ${ssqp_piqp_src_SOURCE_DIR}/include)
endif()

# --- BLASFEO ---
if(SSQP_WITH_HPIPM)
    set(_ssqp_piqp_blasfeo_dir ${_ssqp_hpipm_dir})
    if(TARGET blasfeo_ext)
        add_dependencies(piqp_vendored blasfeo_ext)
    endif()
elseif(SSQP_PREBUILT_PIQP_DIR)
    set(_ssqp_piqp_blasfeo_dir ${SSQP_PREBUILT_PIQP_DIR})
else()
    include(ExternalProject)

    set(_ssqp_piqp_blasfeo_dir ${SSQP_VENDOR_PREFIX_PIQP})

    if(SSQP_BLASFEO_AUTO_TARGET)
        set(_ssqp_blasfeo_target X64_AUTOMATIC)
    else()
        set(_ssqp_blasfeo_target ${SSQP_BLASFEO_TARGET})
    endif()

    if(_ssqp_blasfeo_target STREQUAL "GENERIC")
        message(WARNING
            "BLASFEO is built with the GENERIC target, so PIQP will be "
            "considerably slower. Set SSQP_BLASFEO_TARGET to a target matching "
            "the machines it will run on, or turn SSQP_BLASFEO_AUTO_TARGET on.")
    endif()

    ExternalProject_Add(piqp_blasfeo_ext
        GIT_REPOSITORY https://github.com/giaf/blasfeo.git
        GIT_TAG 60493284741b3e4e5b0bf25155221d4b5f0b2232
        UPDATE_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/piqp_blasfeo
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX_PIQP}
            -DTARGET=${_ssqp_blasfeo_target}
            -DBLASFEO_EXAMPLES=OFF
            -DBLASFEO_TESTING=OFF
            -DBLASFEO_BENCHMARKS=OFF
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX_PIQP}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blasfeo${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    add_dependencies(piqp_vendored piqp_blasfeo_ext)
endif()

if(NOT SSQP_WITH_HPIPM)
    set(SSQP_PIQP_LIBS
        ${_ssqp_piqp_blasfeo_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blasfeo${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

target_include_directories(piqp_vendored SYSTEM INTERFACE
    ${_ssqp_piqp_include_dir}
    ${_ssqp_piqp_blasfeo_dir}/include)
target_compile_definitions(piqp_vendored INTERFACE PIQP_HAS_BLASFEO)
target_link_libraries(piqp_vendored INTERFACE Eigen3::Eigen ${SSQP_PIQP_LIBS})

if(NOT SSQP_PREBUILT_PIQP_DIR)
    # --- package-piqp: assemble a redistributable copy of the above ---
    set(SSQP_PIQP_PACKAGE_DIR "${CMAKE_BINARY_DIR}/prebuilt/piqp" CACHE PATH
        "Output directory for the 'package-piqp' redistributable prebuilt package target")

    set(_ssqp_piqp_package_commands
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${SSQP_PIQP_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SSQP_PIQP_PACKAGE_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_piqp_include_dir} ${SSQP_PIQP_PACKAGE_DIR}/include
    )

    # With HPIPM on, BLASFEO comes from the HPIPM package instead
    if(NOT SSQP_WITH_HPIPM)
        list(APPEND _ssqp_piqp_package_commands
            COMMAND ${CMAKE_COMMAND} -E make_directory ${SSQP_PIQP_PACKAGE_DIR}/lib
            COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_piqp_blasfeo_dir}/include ${SSQP_PIQP_PACKAGE_DIR}/include
            COMMAND ${CMAKE_COMMAND} -E copy ${SSQP_PIQP_LIBS} ${SSQP_PIQP_PACKAGE_DIR}/lib/
        )
    endif()

    add_custom_target(package-piqp
        ${_ssqp_piqp_package_commands}
        COMMENT "Assembling redistributable prebuilt PIQP package at ${SSQP_PIQP_PACKAGE_DIR}"
        VERBATIM
    )

    if(NOT SSQP_WITH_HPIPM)
        add_dependencies(package-piqp piqp_blasfeo_ext)
    endif()
endif()
