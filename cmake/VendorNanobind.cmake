# Fetches nanobind from source so the Python bindings can be built without
# requiring it to be pip-installed beforehand.
#
# Pinned to the 2.x line (not the current 3.x) because python/'s bindings
# use nanobind::detail::ticket's 3-argument constructor directly, which 3.x
# changed to take an extra hash argument.

include(FetchContent)

set(NB_TEST OFF CACHE BOOL "Disable nanobind's own test suite" FORCE)

FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG v2.15.0
    GIT_SHALLOW TRUE
    GIT_SUBMODULES "ext/robin_map"
)
FetchContent_MakeAvailable(nanobind)

# `python -m nanobind.stubgen` (used below for the ssqpy stub-generation
# step) needs `nanobind` to be an importable Python package. That normally
# comes from pip-installing the nanobind PyPI package into whatever
# interpreter CMake's find_package(Python) picks up, but nothing guarantees
# that's been done in this environment (in particular, the fetched nanobind
# above satisfies the C++/CMake side only). nanobind's own src/ directory
# *is* that package (verbatim: __init__.py, stubgen.py, no compiled
# component required) — it's just not named "nanobind" or importable as-is.
# Expose it via a small directory containing a "nanobind" symlink to that
# source, and prepend it to PYTHONPATH when stubgen is invoked.
set(NB_STUBGEN_PYTHONPATH "${CMAKE_BINARY_DIR}/nanobind-stubgen-pythonpath")
file(MAKE_DIRECTORY ${NB_STUBGEN_PYTHONPATH})
file(REMOVE ${NB_STUBGEN_PYTHONPATH}/nanobind)
file(CREATE_LINK ${nanobind_SOURCE_DIR}/src ${NB_STUBGEN_PYTHONPATH}/nanobind SYMBOLIC)
