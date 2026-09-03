# Patches applied to urdfdom before it's configured, run via `cmake -P` (as
# an ExternalProject PATCH_COMMAND, so its working directory is urdfdom's
# source directory) so the edits don't depend on shell/sed quoting.
#
# 1. urdfdom hard-codes SHARED in add_library() regardless of
#    BUILD_SHARED_LIBS, so it's forced to STATIC directly.
# 2. Once static, urdfdom's exported CMake targets carry console_bridge and
#    tinyxml2 into their INTERFACE_LINK_LIBRARIES (private deps of a static
#    library still have to reach the final linker), but urdfdom's own
#    generated package config never find_package()s either of them, so
#    consumers hit "target console_bridge::console_bridge... not found" at
#    configure time unless something upstream happened to find it already.
#    PKG_DEPENDS is what urdfdom-config.cmake.in loops over to find_package
#    its dependencies, so both are added there.

file(READ "urdf_parser/CMakeLists.txt" _urdf_parser_cmakelists)
string(REPLACE
    "add_library(\${add_urdfdom_library_LIBNAME} SHARED"
    "add_library(\${add_urdfdom_library_LIBNAME} STATIC"
    _urdf_parser_cmakelists "${_urdf_parser_cmakelists}")
file(WRITE "urdf_parser/CMakeLists.txt" "${_urdf_parser_cmakelists}")

file(READ "CMakeLists.txt" _urdfdom_cmakelists)
string(REPLACE
    "set(PKG_DEPENDS urdfdom_headers)"
    "set(PKG_DEPENDS urdfdom_headers console_bridge tinyxml2)"
    _urdfdom_cmakelists "${_urdfdom_cmakelists}")
file(WRITE "CMakeLists.txt" "${_urdfdom_cmakelists}")
