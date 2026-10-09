# Provides the `pinocchio_vendored` INTERFACE target, pointing at a
# directory containing include/ and lib/*.a for Pinocchio and its URDF
# parser's dependency chain (urdfdom_headers, console_bridge, tinyxml2,
# urdfdom). That directory comes from one of two places:
#
# - From source (default): fetched and built right here into
#   ${SSQP_VENDOR_PREFIX}, so that ssqp can be built with nothing beyond a
#   C++17 compiler, CMake, Eigen3 and the compiler toolchain. Boost's
#   filesystem and serialization libraries are always linked statically. This
#   mode also defines a `package-pinocchio` target (see bottom of this file)
#   that assembles a redistributable copy of the result for the mode below.
#
# - Prebuilt (SSQP_PREBUILT_PINOCCHIO_DIR set): a pre-existing package
#   produced by `package-pinocchio` on a matching architecture is used as-is
#   — no network access, no compiler chain for Pinocchio/urdfdom, and —
#   since that package bundles a copy of the Boost headers and static libs
#   it was built against — no Boost dependency either. Use this if building
#   Pinocchio (or its dependency chain) from source isn't practical on the
#   target machine.
#
# Either way, the resulting static archives end up embedded in ssqp's final
# artifacts (including the ssqpy shared object).
#
# Pinocchio and urdfdom both hard-code SHARED in their add_library() calls
# regardless of BUILD_SHARED_LIBS, so both are patched to build STATIC.

if(NOT DEFINED SSQP_VENDOR_PREFIX)
    set(SSQP_VENDOR_PREFIX "${CMAKE_BINARY_DIR}/vendor" CACHE PATH "Install prefix for vendored dependencies")
endif()

# The static archives making up a full Pinocchio build, in link order (a
# dependent before its dependencies).
set(_ssqp_pinocchio_libs
    pinocchio_parsers
    pinocchio_default
    pinocchio_visualizers
    urdfdom_model
    urdfdom_world
    urdfdom_sensor
    console_bridge
    tinyxml2
)

add_library(pinocchio_vendored INTERFACE)
target_compile_definitions(pinocchio_vendored INTERFACE
    PINOCCHIO_WITH_URDFDOM
    PINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
    BOOST_MPL_LIMIT_LIST_SIZE=30
    BOOST_MPL_LIMIT_VECTOR_SIZE=30
    BOOST_ALL_NO_LIB
    BOOST_FILESYSTEM_STATIC_LINK
    BOOST_SERIALIZATION_STATIC_LINK
)

if(SSQP_PREBUILT_PINOCCHIO_DIR)
    set(_ssqp_pinocchio_dir ${SSQP_PREBUILT_PINOCCHIO_DIR})
else()
    include(ExternalProject)
    include(FetchContent)

    # The component repositories are fetched as source only and compiled by
    # our own STATIC targets below. FetchContent_MakeAvailable would add the
    # component projects, whose standalone CMake files expect the full Boost
    # superproject, so direct population is intentional here.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()

    # A source fallback is used when either the Boost headers or the two
    # required static archives are unavailable. The fallback version is only
    # used when no system Boost headers exist; with system headers, the
    # component sources are fetched at the matching Boost version.
    set(SSQP_BOOST_VERSION "1.86.0" CACHE STRING
        "Boost version to fetch when system Boost is unavailable")

    # We only need Boost's headers and two archives. Avoid find_package(Boost)
    # here: CMake 4 removed its FindBoost module, while the config package is
    # not needed and does not consistently expose static component archives.
    find_path(_ssqp_boost_include_dir
        NAMES boost/version.hpp
        HINTS ${Boost_INCLUDE_DIR} ${Boost_INCLUDE_DIRS}
    )

    # Ask find_library for archives only. CMake otherwise prefers shared
    # libraries on most Unix platforms, even when a static archive is also
    # installed.
    set(_ssqp_saved_find_library_suffixes ${CMAKE_FIND_LIBRARY_SUFFIXES})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    find_library(_ssqp_system_boost_filesystem
        NAMES boost_filesystem
        HINTS ${Boost_LIBRARY_DIRS})
    find_library(_ssqp_system_boost_serialization
        NAMES boost_serialization
        HINTS ${Boost_LIBRARY_DIRS})
    set(CMAKE_FIND_LIBRARY_SUFFIXES ${_ssqp_saved_find_library_suffixes})

    set(_ssqp_boost_static_targets "")
    set(_ssqp_boost_package_deps "")
    set(_ssqp_boost_from_source FALSE)

    # The system archives must also be linkable into a shared object: ssqpy
    # is a Python extension module, and distro-packaged static Boost
    # archives (e.g. Debian/Ubuntu) are typically compiled without -fPIC,
    # which breaks that link even though a plain static ssqp build would
    # succeed. Probe for this before committing to the fast path below,
    # forcing every object into the link via --whole-archive since any
    # single non-PIC object anywhere in either archive is enough to fail it.
    set(_ssqp_system_boost_is_pic FALSE)
    if(_ssqp_boost_include_dir
       AND _ssqp_system_boost_filesystem
       AND _ssqp_system_boost_serialization)
        set(_ssqp_boost_pic_probe_dir ${CMAKE_BINARY_DIR}/CMakeFiles/ssqp_boost_pic_probe)
        file(MAKE_DIRECTORY ${_ssqp_boost_pic_probe_dir})
        file(WRITE ${_ssqp_boost_pic_probe_dir}/probe.cpp "")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -fPIC -shared
                -o ${_ssqp_boost_pic_probe_dir}/probe.so
                ${_ssqp_boost_pic_probe_dir}/probe.cpp
                -Wl,--whole-archive
                ${_ssqp_system_boost_filesystem} ${_ssqp_system_boost_serialization}
                -Wl,--no-whole-archive
            RESULT_VARIABLE _ssqp_boost_pic_probe_result
            OUTPUT_QUIET ERROR_QUIET
        )
        if(_ssqp_boost_pic_probe_result EQUAL 0)
            set(_ssqp_system_boost_is_pic TRUE)
        else()
            message(STATUS
                "System Boost static archives are not position-independent; building Boost from source instead")
        endif()
    endif()

    if(_ssqp_system_boost_is_pic)
        message(STATUS
            "Using system Boost static archives: ${_ssqp_system_boost_filesystem}; ${_ssqp_system_boost_serialization}")
        # Use the system archives directly when both required static
        # libraries are available.
        add_library(ssqp_boost_filesystem STATIC IMPORTED GLOBAL)
        set_target_properties(ssqp_boost_filesystem PROPERTIES
            IMPORTED_LOCATION ${_ssqp_system_boost_filesystem}
            INTERFACE_INCLUDE_DIRECTORIES ${_ssqp_boost_include_dir}
        )
        add_library(ssqp_boost_serialization STATIC IMPORTED GLOBAL)
        set_target_properties(ssqp_boost_serialization PROPERTIES
            IMPORTED_LOCATION ${_ssqp_system_boost_serialization}
            INTERFACE_INCLUDE_DIRECTORIES ${_ssqp_boost_include_dir}
        )
        list(APPEND _ssqp_boost_static_targets
            ssqp_boost_filesystem
            ssqp_boost_serialization
        )
        get_filename_component(_ssqp_boost_library_dir
            ${_ssqp_system_boost_filesystem} DIRECTORY)
    else()
        # Build PIC static archives so the same Boost objects can be linked
        # into both libssqp.a and the ssqpy shared module.
        set(_ssqp_boost_from_source TRUE)

        if(_ssqp_boost_include_dir)
            file(STRINGS ${_ssqp_boost_include_dir}/boost/version.hpp
                _ssqp_boost_version_line
                REGEX "^#define[ \t]+BOOST_VERSION[ \t]+[0-9]+")
            string(REGEX MATCH "[0-9]+$" _ssqp_boost_version_number
                "${_ssqp_boost_version_line}")
            math(EXPR _ssqp_boost_version_major
                "${_ssqp_boost_version_number} / 100000")
            math(EXPR _ssqp_boost_version_minor
                "(${_ssqp_boost_version_number} / 100) % 1000")
            math(EXPR _ssqp_boost_version_patch
                "${_ssqp_boost_version_number} % 100")
            set(_ssqp_boost_source_version
                ${_ssqp_boost_version_major}.${_ssqp_boost_version_minor}.${_ssqp_boost_version_patch})
            message(STATUS
                "Boost static archives unavailable; fetching Boost ${_ssqp_boost_source_version} component sources")

            FetchContent_Declare(ssqp_boost_filesystem_src
                GIT_REPOSITORY https://github.com/boostorg/filesystem.git
                GIT_TAG boost-${_ssqp_boost_source_version}
                GIT_SHALLOW TRUE
                UPDATE_DISCONNECTED TRUE
            )
            FetchContent_Declare(ssqp_boost_serialization_src
                GIT_REPOSITORY https://github.com/boostorg/serialization.git
                GIT_TAG boost-${_ssqp_boost_source_version}
                GIT_SHALLOW TRUE
                UPDATE_DISCONNECTED TRUE
            )
            FetchContent_Populate(ssqp_boost_filesystem_src)
            FetchContent_Populate(ssqp_boost_serialization_src)
            set(_ssqp_boost_filesystem_src_dir
                ${ssqp_boost_filesystem_src_SOURCE_DIR}/src)
            set(_ssqp_boost_serialization_src_dir
                ${ssqp_boost_serialization_src_SOURCE_DIR}/src)
        else()
            # The Boost release archive contains the headers and the two
            # component source trees, avoiding a dependency on a system Boost
            # installation altogether.
            message(STATUS
                "Boost is unavailable; fetching Boost ${SSQP_BOOST_VERSION} sources")
            string(REPLACE "." "_" _ssqp_boost_archive_version
                   ${SSQP_BOOST_VERSION})
            FetchContent_Declare(ssqp_boost_src
                URL https://archives.boost.io/release/${SSQP_BOOST_VERSION}/source/boost_${_ssqp_boost_archive_version}.tar.gz
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
            FetchContent_Populate(ssqp_boost_src)
            set(_ssqp_boost_include_dir ${ssqp_boost_src_SOURCE_DIR})
            set(_ssqp_boost_filesystem_src_dir
                ${ssqp_boost_src_SOURCE_DIR}/libs/filesystem/src)
            set(_ssqp_boost_serialization_src_dir
                ${ssqp_boost_src_SOURCE_DIR}/libs/serialization/src)
        endif()

        file(GLOB _ssqp_boost_filesystem_srcs CONFIGURE_DEPENDS
            ${_ssqp_boost_filesystem_src_dir}/*.cpp)
        list(FILTER _ssqp_boost_filesystem_srcs EXCLUDE REGEX
            "windows_file_codecvt\\.cpp$")
        add_library(ssqp_boost_filesystem STATIC EXCLUDE_FROM_ALL
            ${_ssqp_boost_filesystem_srcs})
        target_include_directories(ssqp_boost_filesystem SYSTEM PRIVATE
            ${_ssqp_boost_include_dir})
        target_compile_definitions(ssqp_boost_filesystem PRIVATE
            BOOST_FILESYSTEM_STATIC_LINK BOOST_ALL_NO_LIB)
        set_target_properties(ssqp_boost_filesystem PROPERTIES
            OUTPUT_NAME boost_filesystem
            ARCHIVE_OUTPUT_DIRECTORY ${SSQP_VENDOR_PREFIX}/lib
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )

        file(GLOB _ssqp_boost_serialization_srcs CONFIGURE_DEPENDS
            ${_ssqp_boost_serialization_src_dir}/*.cpp)
        add_library(ssqp_boost_serialization STATIC EXCLUDE_FROM_ALL
            ${_ssqp_boost_serialization_srcs})
        target_include_directories(ssqp_boost_serialization SYSTEM PRIVATE
            ${_ssqp_boost_include_dir})
        target_compile_definitions(ssqp_boost_serialization PRIVATE
            BOOST_SERIALIZATION_STATIC_LINK BOOST_ALL_NO_LIB)
        set_target_properties(ssqp_boost_serialization PROPERTIES
            OUTPUT_NAME boost_serialization
            ARCHIVE_OUTPUT_DIRECTORY ${SSQP_VENDOR_PREFIX}/lib
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD 20
            CXX_STANDARD_REQUIRED ON
        )
        list(APPEND _ssqp_boost_static_targets
            ssqp_boost_filesystem
            ssqp_boost_serialization
        )
        list(APPEND _ssqp_boost_package_deps
            ssqp_boost_filesystem
            ssqp_boost_serialization
        )
        set(_ssqp_boost_library_dir ${SSQP_VENDOR_PREFIX}/lib)
    endif()

    set(_ssqp_pinocchio_dir ${SSQP_VENDOR_PREFIX})

    # Pinocchio's Eigen-based types (e.g. DataTpl's internal aligned containers)
    # have a layout that depends on which vectorization ISA is active at compile
    # time. ssqp itself is compiled with SSQP_ARCH_FLAG (see top-level
    # CMakeLists.txt), so Pinocchio has to be built with the exact same flag —
    # otherwise the two sides disagree on alignment for the same extern-template
    # types and calling into Pinocchio's precompiled code corrupts memory
    # (observed as "double free or corruption" inside DataTpl's constructor).
    set(_ssqp_vendor_cmake_args
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_FLAGS=${SSQP_ARCH_FLAG}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_INSTALL_PREFIX=${SSQP_VENDOR_PREFIX}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_PREFIX_PATH=${SSQP_VENDOR_PREFIX}
        # Same Eigen as ssqp, since CMAKE_PREFIX_PATH above hides the caller's
        -DEigen3_DIR=${Eigen3_DIR}
        -DBUILD_TESTING=OFF
        -DBoost_USE_STATIC_LIBS=ON
        -DBoost_USE_STATIC_RUNTIME=OFF
        -DBoost_NO_SYSTEM_PATHS=ON
        -DBoost_INCLUDE_DIR=${_ssqp_boost_include_dir}
        -DBoost_LIBRARY_DIR=${_ssqp_boost_library_dir}
    )

    # CMake 4 rejects projects whose declared policy version predates 3.5.
    # The pinned console_bridge 1.0.2 release declares 3.0.2, so tell the
    # child configure to use a compatible policy baseline without modifying
    # the fetched source tree.
    list(APPEND _ssqp_vendor_cmake_args
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    )

    ExternalProject_Add(urdfdom_headers_ext
        GIT_REPOSITORY https://github.com/ros/urdfdom_headers.git
        # urdfdom 4.0.0 still includes and builds the model-state API,
        # which was removed from urdfdom_headers 2.x and later.
        GIT_TAG 1.1.1
        GIT_SHALLOW TRUE
        # The dependency is pinned; avoid a phony Git update step on every build.
        UPDATE_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/urdfdom_headers
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
    )

    ExternalProject_Add(console_bridge_ext
        GIT_REPOSITORY https://github.com/ros/console_bridge.git
        GIT_TAG 1.0.2
        GIT_SHALLOW TRUE
        UPDATE_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/console_bridge
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}console_bridge${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(tinyxml2_ext
        GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
        GIT_TAG 10.1.0
        GIT_SHALLOW TRUE
        UPDATE_COMMAND ""
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/tinyxml2
        CMAKE_ARGS ${_ssqp_vendor_cmake_args} -Dtinyxml2_SHARED_LIBS=OFF -Dtinyxml2_BUILD_TESTING=OFF
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tinyxml2${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(urdfdom_ext
        GIT_REPOSITORY https://github.com/ros/urdfdom.git
        GIT_TAG 4.0.0
        GIT_SHALLOW TRUE
        UPDATE_COMMAND ""
        DEPENDS urdfdom_headers_ext console_bridge_ext tinyxml2_ext
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/urdfdom
        PATCH_COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/patch-urdfdom.cmake
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
        BUILD_BYPRODUCTS
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}urdfdom_model${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}urdfdom_world${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}urdfdom_sensor${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(pinocchio_ext
        GIT_REPOSITORY https://github.com/stack-of-tasks/pinocchio.git
        GIT_TAG v3.9.0
        GIT_SHALLOW TRUE
        GIT_SUBMODULES "cmake"
        GIT_SUBMODULES_RECURSE TRUE
        UPDATE_COMMAND ""
        DEPENDS urdfdom_ext
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/pinocchio
        PATCH_COMMAND sed -i "s/set(LIBRARY_TYPE SHARED)/set(LIBRARY_TYPE STATIC)/" src/CMakeLists.txt
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
            -DBUILD_WITH_COLLISION_SUPPORT=OFF
            -DBUILD_WITH_SDF_SUPPORT=OFF
            -DBUILD_WITH_URDF_SUPPORT=ON
            -DBUILD_PYTHON_INTERFACE=OFF
            -DBUILD_EXAMPLES=OFF
            -DBUILD_BENCHMARK=OFF
            -DBUILD_UTILS=OFF
            -DINSTALL_DOCUMENTATION=OFF
            -DENABLE_TEMPLATE_INSTANTIATION=ON
        BUILD_BYPRODUCTS
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}pinocchio_default${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}pinocchio_parsers${CMAKE_STATIC_LIBRARY_SUFFIX}
            ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}pinocchio_visualizers${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    # If Boost was fetched and compiled by this build, make it available
    # before the child projects configure and search for Boost libraries.
    if(_ssqp_boost_from_source)
        add_dependencies(urdfdom_ext ${_ssqp_boost_package_deps})
        add_dependencies(pinocchio_ext ${_ssqp_boost_package_deps})
    endif()
    add_dependencies(pinocchio_vendored pinocchio_ext)
endif()

# From here on, pinocchio_vendored is defined the same way regardless of
# mode — just pointed at ${_ssqp_pinocchio_dir}, which is either what was
# just built above or the prebuilt package handed to us.
target_include_directories(pinocchio_vendored SYSTEM INTERFACE ${_ssqp_pinocchio_dir}/include)

# Static archives don't record their own dependencies, so the whole
# transitive chain has to be listed explicitly. Wrapped in --start-group/
# --end-group since several of these reference each other.
set(_pinocchio_lib_paths "")
foreach(_lib ${_ssqp_pinocchio_libs})
    list(APPEND _pinocchio_lib_paths
        ${_ssqp_pinocchio_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${_lib}${CMAKE_STATIC_LIBRARY_SUFFIX})
endforeach()

if(SSQP_PREBUILT_PINOCCHIO_DIR)
    # Boost is bundled in the prebuilt package too, right alongside everything else.
    list(APPEND _pinocchio_lib_paths
        ${_ssqp_pinocchio_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_filesystem${CMAKE_STATIC_LIBRARY_SUFFIX}
        ${_ssqp_pinocchio_dir}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}boost_serialization${CMAKE_STATIC_LIBRARY_SUFFIX})
endif()

target_link_libraries(pinocchio_vendored INTERFACE
    -Wl,--start-group ${_pinocchio_lib_paths} ${_ssqp_boost_static_targets} -Wl,--end-group
)

if(NOT SSQP_PREBUILT_PINOCCHIO_DIR)
    # --- package-pinocchio: assemble a redistributable copy of the above ---
    #
    # Bundles a copy of the Boost headers and static libs this was built
    # against, so a machine using the resulting package has no Boost
    # dependency at all — not even at compile time, which a plain link-only
    # bundle wouldn't cover, since ssqp's own PinocchioWrapper.cpp includes
    # Pinocchio headers that use Boost types directly (e.g. constructing
    # pinocchio::Data by value requires its full definition, which pulls in
    # boost::variant).
    #
    set(SSQP_PINOCCHIO_PACKAGE_DIR "${CMAKE_BINARY_DIR}/prebuilt/pinocchio" CACHE PATH
        "Output directory for the 'package-pinocchio' redistributable prebuilt package target")

    add_custom_target(package-pinocchio
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${SSQP_PINOCCHIO_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SSQP_PINOCCHIO_PACKAGE_DIR}/include ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_pinocchio_dir}/include ${SSQP_PINOCCHIO_PACKAGE_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_boost_include_dir}/boost ${SSQP_PINOCCHIO_PACKAGE_DIR}/include/boost
        COMMAND ${CMAKE_COMMAND} -E copy ${_pinocchio_lib_paths} ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib/
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:ssqp_boost_filesystem> $<TARGET_FILE:ssqp_boost_serialization>
            ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib/
        DEPENDS pinocchio_ext ${_ssqp_boost_package_deps}
        COMMENT "Assembling redistributable prebuilt Pinocchio package (headers + static libs, Boost bundled) at ${SSQP_PINOCCHIO_PACKAGE_DIR}"
        VERBATIM
    )
endif()
