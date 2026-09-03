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
# BLASFEO's TARGET selects a hand-written assembly kernel set for the build
# machine's microarchitecture; X64_AUTOMATIC probes a list of candidates at
# configure time and picks the first that actually works, which keeps this
# portable across machines rather than assuming e.g. AVX2/FMA are present.
# HPIPM only exposes a coarse AVX/GENERIC choice for its own sources; AVX
# covers effectively any machine this project would run on. A prebuilt
# package is only portable across machines with compatible kernel support
# for whichever TARGET it was actually built with.

if(NOT DEFINED SSQP_VENDOR_PREFIX_HPIPM)
    set(SSQP_VENDOR_PREFIX_HPIPM "${CMAKE_BINARY_DIR}/vendor-hpipm" CACHE PATH "Install prefix for vendored BLASFEO/HPIPM")
endif()

add_library(hpipm_vendored INTERFACE)

if(SSQP_PREBUILT_HPIPM_DIR)
    set(_ssqp_hpipm_dir ${SSQP_PREBUILT_HPIPM_DIR})
else()
    include(ExternalProject)

    set(_ssqp_hpipm_dir ${SSQP_VENDOR_PREFIX_HPIPM})

    ExternalProject_Add(blasfeo_ext
        GIT_REPOSITORY https://github.com/giaf/blasfeo.git
        GIT_TAG 60493284741b3e4e5b0bf25155221d4b5f0b2232
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/blasfeo
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX_HPIPM}
            -DTARGET=X64_AUTOMATIC
            -DBLASFEO_EXAMPLES=OFF
            -DBLASFEO_TESTING=OFF
            -DBLASFEO_BENCHMARKS=OFF
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX_HPIPM}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}blasfeo${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(hpipm_ext
        GIT_REPOSITORY https://github.com/giaf/hpipm.git
        GIT_TAG 87ef7a6ab773427a8c416030a2b9d075a744c223
        DEPENDS blasfeo_ext
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/hpipm
        CMAKE_ARGS
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=OFF
            -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX_HPIPM}
            -DTARGET=AVX
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
