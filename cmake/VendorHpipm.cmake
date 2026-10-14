# Provides the `hpipm_vendored` INTERFACE target, pointing at a directory
# containing include/ and lib/*.a for BLASFEO and HPIPM. That directory
# comes from one of two places:
#
# - From source (default): fetched and built right here into
#   ${SSQP_VENDOR_PREFIX_HPIPM}, so the HPIPM QP solver backend can be built
#   without any manual setup. Pinned to specific commits since neither
#   project publishes up-to-date release tags. This mode also defines a
#   `package-hpipm` target (see bottom of this file) that assembles a
#   redistributable copy of the result for the mode below.
#
# - Prebuilt (SSQP_PREBUILT_HPIPM_DIR set): a pre-existing package produced
#   by `package-hpipm` on a matching architecture is used as-is — no
#   network access and no compiler chain for BLASFEO/HPIPM needed. Use this
#   if building them from source isn't practical on the target machine
#   (e.g. BLASFEO's kernel selection failing on an unusual CPU).
#
# BLASFEO's TARGET selects a hand-written assembly kernel set. With
# SSQP_BLASFEO_AUTO_TARGET on, X64_AUTOMATIC probes a list of candidates at
# configure time and picks the first that works on the build machine, so the
# result only runs on machines with the same CPU features. With it off,
# SSQP_BLASFEO_TARGET is used instead (e.g. X64_INTEL_HASWELL for AVX2+FMA,
# or GENERIC for plain C that runs anywhere but is much slower).
# HPIPM only exposes a coarse AVX/GENERIC choice for its own sources and
# builds AVX with -mavx, so it uses AVX only when the BLASFEO target does.
# A prebuilt package is only portable across machines with compatible kernel
# support for whichever TARGET it was actually built with.

if(NOT DEFINED SSQP_VENDOR_PREFIX_HPIPM)
    set(SSQP_VENDOR_PREFIX_HPIPM "${CMAKE_BINARY_DIR}/vendor-hpipm" CACHE PATH "Install prefix for vendored BLASFEO/HPIPM")
endif()

add_library(hpipm_vendored INTERFACE)

if(SSQP_PREBUILT_HPIPM_DIR)
    set(_ssqp_hpipm_dir ${SSQP_PREBUILT_HPIPM_DIR})
else()
    include(ExternalProject)

    set(_ssqp_hpipm_dir ${SSQP_VENDOR_PREFIX_HPIPM})

    if(SSQP_BLASFEO_AUTO_TARGET)
        set(_ssqp_blasfeo_target X64_AUTOMATIC)
    else()
        set(_ssqp_blasfeo_target ${SSQP_BLASFEO_TARGET})
    endif()

    if(_ssqp_blasfeo_target MATCHES "^X64_(AUTOMATIC|INTEL_(HASWELL|SANDY_BRIDGE|SKYLAKE_X)|AMD_BULLDOZER)$")
        set(_ssqp_hpipm_target AVX)
    else()
        set(_ssqp_hpipm_target GENERIC)
    endif()

    if(_ssqp_blasfeo_target STREQUAL "GENERIC")
        message(WARNING
            "BLASFEO is built with the GENERIC target, so HPIPM will be "
            "considerably slower. Set SSQP_BLASFEO_TARGET to a target matching "
            "the machines it will run on, or turn SSQP_BLASFEO_AUTO_TARGET on.")
    endif()

    ExternalProject_Add(blasfeo_ext
        GIT_REPOSITORY https://github.com/giaf/blasfeo.git
        GIT_TAG 60493284741b3e4e5b0bf25155221d4b5f0b2232
        # The dependency is pinned; avoid a phony Git update step on every build.
        UPDATE_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/blasfeo
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX_HPIPM}
            -DTARGET=${_ssqp_blasfeo_target}
            -DBLASFEO_EXAMPLES=OFF
            -DBLASFEO_TESTING=OFF
            -DBLASFEO_BENCHMARKS=OFF
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX_HPIPM}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blasfeo${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(hpipm_ext
        GIT_REPOSITORY https://github.com/giaf/hpipm.git
        GIT_TAG 16a8db59f595bda642f460aa0f62e308f4b76db3
        UPDATE_COMMAND ""
        DEPENDS blasfeo_ext
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/hpipm
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX_HPIPM}
            -DTARGET=${_ssqp_hpipm_target}
            -DHPIPM_TESTING=OFF
            -DBLASFEO_PATH=${SSQP_VENDOR_PREFIX_HPIPM}
            -DHPIPM_BLASFEO_LIB=Static
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX_HPIPM}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hpipm${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    add_dependencies(hpipm_vendored hpipm_ext)
endif()

# From here on, hpipm_vendored is defined the same way regardless of mode —
# just pointed at ${_ssqp_hpipm_dir}, which is either what was just built
# above or the prebuilt package handed to us.
target_include_directories(hpipm_vendored SYSTEM INTERFACE ${_ssqp_hpipm_dir}/include)

set(_hpipm_lib_paths
    ${_ssqp_hpipm_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}hpipm${CMAKE_STATIC_LIBRARY_SUFFIX}
    ${_ssqp_hpipm_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blasfeo${CMAKE_STATIC_LIBRARY_SUFFIX}
)
target_link_libraries(hpipm_vendored INTERFACE ${_hpipm_lib_paths})
set(SSQP_HPIPM_LIBS ${_hpipm_lib_paths})

if(NOT SSQP_PREBUILT_HPIPM_DIR)
    # --- package-hpipm: assemble a redistributable copy of the above ---
    set(SSQP_HPIPM_PACKAGE_DIR "${CMAKE_BINARY_DIR}/prebuilt/hpipm" CACHE PATH
        "Output directory for the 'package-hpipm' redistributable prebuilt package target")

    add_custom_target(package-hpipm
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${SSQP_HPIPM_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SSQP_HPIPM_PACKAGE_DIR}/include ${SSQP_HPIPM_PACKAGE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_hpipm_dir}/include ${SSQP_HPIPM_PACKAGE_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy ${_hpipm_lib_paths} ${SSQP_HPIPM_PACKAGE_DIR}/lib/
        DEPENDS hpipm_ext
        COMMENT "Assembling redistributable prebuilt HPIPM/BLASFEO package at ${SSQP_HPIPM_PACKAGE_DIR}"
        VERBATIM
    )
endif()
