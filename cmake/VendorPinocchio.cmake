# Provides the `pinocchio_vendored` INTERFACE target, pointing at a
# directory containing include/ and lib/*.a for Pinocchio and its URDF
# parser's dependency chain (urdfdom_headers, console_bridge, tinyxml2,
# urdfdom). That directory comes from one of two places:
#
# - From source (default): fetched and built right here into
#   ${SSQP_VENDOR_PREFIX}, so that ssqp can be built with nothing beyond a
#   C++17 compiler, CMake, Eigen3 and Boost. This mode also defines a
#   `package-pinocchio` target (see bottom of this file) that assembles a
#   redistributable copy of the result for the mode below.
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
)

if(SSQP_PREBUILT_PINOCCHIO_DIR)
    set(_ssqp_pinocchio_dir ${SSQP_PREBUILT_PINOCCHIO_DIR})
else()
    include(ExternalProject)

    find_package(Boost REQUIRED COMPONENTS filesystem serialization)

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
        -DBUILD_TESTING=OFF
    )

    ExternalProject_Add(urdfdom_headers_ext
        GIT_REPOSITORY https://github.com/ros/urdfdom_headers.git
        GIT_TAG 3.0.1
        GIT_SHALLOW TRUE
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/urdfdom_headers
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
    )

    ExternalProject_Add(console_bridge_ext
        GIT_REPOSITORY https://github.com/ros/console_bridge.git
        GIT_TAG 1.0.2
        GIT_SHALLOW TRUE
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/console_bridge
        CMAKE_ARGS ${_ssqp_vendor_cmake_args}
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}console_bridge${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(tinyxml2_ext
        GIT_REPOSITORY https://github.com/leethomason/tinyxml2.git
        GIT_TAG 10.1.0
        GIT_SHALLOW TRUE
        PREFIX ${CMAKE_BINARY_DIR}/_vendor/tinyxml2
        CMAKE_ARGS ${_ssqp_vendor_cmake_args} -Dtinyxml2_SHARED_LIBS=OFF -Dtinyxml2_BUILD_TESTING=OFF
        BUILD_BYPRODUCTS ${SSQP_VENDOR_PREFIX}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}tinyxml2${CMAKE_STATIC_LIBRARY_SUFFIX}
    )

    ExternalProject_Add(urdfdom_ext
        GIT_REPOSITORY https://github.com/ros/urdfdom.git
        GIT_TAG 4.0.0
        GIT_SHALLOW TRUE
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
    target_link_libraries(pinocchio_vendored INTERFACE -Wl,--start-group ${_pinocchio_lib_paths} -Wl,--end-group)
else()
    target_link_libraries(pinocchio_vendored INTERFACE
        -Wl,--start-group ${_pinocchio_lib_paths} -Wl,--end-group
        Boost::filesystem
        Boost::serialization
    )
endif()

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
    # The system's Boost static libs (the only ones most distros ship) are
    # not built with -fPIC, so they can't be linked into ssqpy (a shared
    # object) — only into plain executables like run_tests, where the
    # problem doesn't show up. There's no PIC variant to fall back to (this
    # is also why the default from-source build doesn't hit this: it links
    # against Boost's *shared* .so via the Boost::filesystem/serialization
    # imported targets below, which needs no PIC static archive at all).
    # So filesystem and serialization — the only two Boost libraries ssqp
    # actually needs — are recompiled here from their own upstream sources,
    # at the tag matching the system Boost version so headers and object
    # code agree, with -fPIC. They need -std=c++20 (their own atomic_ref
    # usage needs it) independently of the rest of this project's C++17,
    # which is safe: it doesn't change the object's C++ ABI.
    include(FetchContent)
    FetchContent_Declare(boost_filesystem_pic_src
        GIT_REPOSITORY https://github.com/boostorg/filesystem.git
        GIT_TAG boost-${Boost_VERSION}
        GIT_SHALLOW TRUE
    )
    FetchContent_Declare(boost_serialization_pic_src
        GIT_REPOSITORY https://github.com/boostorg/serialization.git
        GIT_TAG boost-${Boost_VERSION}
        GIT_SHALLOW TRUE
    )
    # FetchContent_MakeAvailable would also add_subdirectory() these — their
    # CMakeLists.txt assumes it's being built inside the full Boost
    # superproject (e.g. serialization's links against a Boost::array
    # target that only exists there), so FetchContent_Populate is used
    # instead to fetch the source without invoking their own build.
    FetchContent_Populate(boost_filesystem_pic_src)
    FetchContent_Populate(boost_serialization_pic_src)

    file(GLOB _boost_filesystem_pic_srcs ${boost_filesystem_pic_src_SOURCE_DIR}/src/*.cpp)
    list(FILTER _boost_filesystem_pic_srcs EXCLUDE REGEX "windows_file_codecvt\\.cpp$")
    add_library(boost_filesystem_pic STATIC EXCLUDE_FROM_ALL ${_boost_filesystem_pic_srcs})
    target_include_directories(boost_filesystem_pic SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
    target_compile_definitions(boost_filesystem_pic PRIVATE BOOST_FILESYSTEM_STATIC_LINK BOOST_ALL_NO_LIB)
    set_target_properties(boost_filesystem_pic PROPERTIES
        OUTPUT_NAME boost_filesystem
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    file(GLOB _boost_serialization_pic_srcs ${boost_serialization_pic_src_SOURCE_DIR}/src/*.cpp)
    add_library(boost_serialization_pic STATIC EXCLUDE_FROM_ALL ${_boost_serialization_pic_srcs})
    target_include_directories(boost_serialization_pic SYSTEM PRIVATE ${Boost_INCLUDE_DIRS})
    target_compile_definitions(boost_serialization_pic PRIVATE BOOST_ALL_NO_LIB)
    set_target_properties(boost_serialization_pic PROPERTIES
        OUTPUT_NAME boost_serialization
        POSITION_INDEPENDENT_CODE ON
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )

    list(GET Boost_INCLUDE_DIRS 0 _boost_include_dir)
    set(SSQP_PINOCCHIO_PACKAGE_DIR "${CMAKE_BINARY_DIR}/prebuilt/pinocchio" CACHE PATH
        "Output directory for the 'package-pinocchio' redistributable prebuilt package target")

    add_custom_target(package-pinocchio
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${SSQP_PINOCCHIO_PACKAGE_DIR}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${SSQP_PINOCCHIO_PACKAGE_DIR}/include ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_ssqp_pinocchio_dir}/include ${SSQP_PINOCCHIO_PACKAGE_DIR}/include
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${_boost_include_dir}/boost ${SSQP_PINOCCHIO_PACKAGE_DIR}/include/boost
        COMMAND ${CMAKE_COMMAND} -E copy ${_pinocchio_lib_paths} ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib/
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_FILE:boost_filesystem_pic> $<TARGET_FILE:boost_serialization_pic>
            ${SSQP_PINOCCHIO_PACKAGE_DIR}/lib/
        DEPENDS pinocchio_ext boost_filesystem_pic boost_serialization_pic
        COMMENT "Assembling redistributable prebuilt Pinocchio package (headers + static libs, Boost bundled and PIC-compiled) at ${SSQP_PINOCCHIO_PACKAGE_DIR}"
        VERBATIM
    )
endif()
