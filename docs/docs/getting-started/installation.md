# Installation

## Installing with pip

Prebuilt wheels are available for Linux x86_64 and Python 3.10 or newer:

```bash
pip install ssqp
```

The package provides the `ssqpy` module. The wheels require a CPU with AVX2 and FMA (Intel Haswell, AMD Excavator or
newer). On older CPUs, importing `ssqpy` fails with an illegal instruction error, so build from source instead, as
described below. Building from source is also needed for the C++ library, other platforms, or non-default
[CMake options](#cmake-options).

## Dependencies

SSQP only expects a couple of dependencies to already be present on the system:

* [Eigen3](https://eigen.tuxfamily.org/): linear algebra
* [NumPy](https://numpy.org/): only required if `SSQP_BUILD_PYTHON` is enabled (the default); `ssqpy`'s bindings convert
  between Eigen and NumPy arrays, so NumPy must be importable in whatever Python interpreter CMake picks up (via
  `find_package(Python)`; pass `-DPython_EXECUTABLE=/path/to/python` to `cmake` to pin a specific one) both when
  building and when later using `ssqpy`

!!! tip
    Building Pinocchio (and, if enabled, HPIPM/BLASFEO and PIQP) from source is the default, but SSQP also supports building
    against **prebuilt** packages for them instead; see [Building against prebuilt
    dependencies](#building-against-prebuilt-dependencies) below.

Everything else is fetched and built from source automatically as part of the CMake build, statically linked, and
baked into the final artifacts (including the `ssqpy` shared object), with no system-wide install, package manager,
or manual build step required:

* [Boost](https://www.boost.org/) `filesystem`/`serialization`: always linked statically into the final artifacts,
  and used from the system when a suitable install is found there, falling back automatically otherwise, in one of
  two ways: if Boost headers are present but not both required static archives, CMake fetches and builds just the
  `filesystem`/`serialization` component sources at the matching, already-installed Boost version; if no system
  Boost is found at all, CMake fetches and builds a full Boost release instead (the `SSQP_BOOST_VERSION` CMake
  option, default `1.86.0`, picks which one). None of this applies when building against a prebuilt Pinocchio
  package (see below), which bundles its own Boost static libs instead.
* [Pinocchio](https://github.com/stack-of-tasks/pinocchio): rigid body dynamics algorithms, along with the
  dependency chain needed for its URDF parser (urdfdom, urdfdom_headers, console_bridge, tinyxml2)
* [nanobind](https://github.com/wjakob/nanobind): Python bindings, fetched automatically if `SSQP_BUILD_PYTHON` is
  enabled (the default)
* [BLASFEO](https://github.com/giaf/blasfeo) and [HPIPM](https://github.com/giaf/hpipm): fetched automatically if
  the `SSQP_WITH_HPIPM` CMake option is enabled, for the HPIPM-based QP solver (see [QP Solvers](../concepts/qp-solvers.md))
* [PIQP](https://github.com/PREDICT-EPFL/piqp): header-only, fetched automatically if the `SSQP_WITH_PIQP` CMake
  option is enabled, for the PIQP-based QP solver (see [QP Solvers](../concepts/qp-solvers.md)). Its multistage solver
  also needs BLASFEO, which is shared with HPIPM when both are enabled and fetched and built for PIQP otherwise
* [nlohmann/json](https://github.com/nlohmann/json) and [Tracy](https://github.com/wolfpld/tracy): vendored under
  [`external/`](https://github.com/upatras-lar/ssqp/tree/main/external) as git submodules

Make sure to clone submodules along with the repository:

```bash
git clone --recurse-submodules git@github.com:upatras-lar/ssqp.git
```

The dependencies fetched by CMake (see [`cmake/`](https://github.com/upatras-lar/ssqp/tree/main/cmake)) require
network access during the build. The first build will take noticeably longer than subsequent ones since it has to
compile Pinocchio and, if enabled, HPIPM/BLASFEO from source (PIQP itself is header-only); later builds only recompile them if their vendored
sources change. The [`.ci/Dockerfile`](https://github.com/upatras-lar/ssqp/blob/main/.ci/Dockerfile) documents a working system
setup (Ubuntu 24.04, Eigen3, Boost) to have in place before running CMake; Boost is included there so CI doesn't pay
the fetch-and-build cost on every run, not because it's strictly required (see [Dependencies](#dependencies) above).

## Building the library

From the repository root:

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

Every dependency SSQP builds itself, Boost included, is linked statically, so the `ssqpy` Python extension only
keeps the platform's C/C++ runtime as a shared dependency. With `SSQP_BUILD_PYTHON=ON` (the default) and
[`readelf`](https://man7.org/linux/man-pages/man1/readelf.1.html) available, verify this for a completed build with:

```bash
cmake --build . --target verify-static-dependencies
```

### CMake options

| Option | Default | Description |
| --- | --- | --- |
| `SSQP_WITH_HPIPM` | `ON` | Fetch, build and link against HPIPM and BLASFEO to enable the [HPIPM QP solver](../concepts/qp-solvers.md) |
| `SSQP_PREBUILT_PINOCCHIO_DIR` | *(empty)* | Path to a prebuilt Pinocchio package instead of building it from source; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) |
| `SSQP_PREBUILT_HPIPM_DIR` | *(empty)* | Path to a prebuilt HPIPM/BLASFEO package instead of building it from source (only relevant if `SSQP_WITH_HPIPM=ON`) |
| `SSQP_WITH_PIQP` | `ON` | Fetch and link against PIQP (and BLASFEO) to enable the [PIQP QP solver](../concepts/qp-solvers.md#piqp) |
| `SSQP_PIQP_OPENMP` | `OFF` | Build the PIQP backend with OpenMP, enabling its [parallel multistage solver](../concepts/qp-solvers.md#piqp); adds a shared dependency on the OpenMP runtime (only relevant if `SSQP_WITH_PIQP=ON`) |
| `SSQP_PREBUILT_PIQP_DIR` | *(empty)* | Path to a prebuilt PIQP package instead of fetching it, along with BLASFEO when `SSQP_WITH_HPIPM=OFF` (only relevant if `SSQP_WITH_PIQP=ON`) |
| `SSQP_BOOST_VERSION` | `1.86.0` | Boost release fetched when no system Boost headers are available (see [Dependencies](#dependencies) above) |
| `SSQP_ENABLE_PROFILING` | `OFF` | Enable [Tracy](https://github.com/wolfpld/tracy) profiling |
| `SSQP_DISABLE_LOGGING` | `OFF` | Compile out SSQP's logging (see `ssqp::set_log_level`) |
| `SSQP_BUILD_EXAMPLES` | `OFF` | Build the C++ example executables under `examples/trajgen/c++` |
| `SSQP_BUILD_TESTS` | `ON` | Build the unit test suite under `tests/`; `OFF` when SSQP is [added to another project](#adding-ssqp-to-a-cmake-project) |
| `SSQP_BUILD_PYTHON` | `ON` | Build the `ssqpy` Python extension via nanobind; `OFF` when SSQP is [added to another project](#adding-ssqp-to-a-cmake-project) |

Pass options as `-D<OPTION>=<VALUE>` to `cmake`, e.g.:

```bash
cmake -DSSQP_WITH_HPIPM=ON -DSSQP_BUILD_EXAMPLES=ON ..
```

### CMake targets

Build a specific target instead of everything with `cmake --build . --target <target>`:

| Target | Description |
| --- | --- |
| `ssqp` | The static C++ library; always built |
| `ssqpy` | The `ssqpy` Python extension module (`SSQP_BUILD_PYTHON=ON`, the default) |
| `run_tests` | The unit/regression test executable under `tests/` (`SSQP_BUILD_TESTS=ON`, the default) |
| *(per-file, e.g.* `analytical-di`*)* | One executable per C++ example under `examples/trajgen/c++` (`SSQP_BUILD_EXAMPLES=ON`); named after the file's path relative to that directory, with `/` replaced by `-` |
| `install` | Installs the C++ package (`cpp` component, see [Using the C++ library](#using-the-c-library)) and the built `ssqpy` (`python` component, see [Using the Python bindings](#using-the-python-bindings)) |
| `verify-static-dependencies` | Confirms `ssqp`/`ssqpy` link no shared Boost, Pinocchio, URDF, HPIPM, or BLASFEO libraries (`SSQP_BUILD_PYTHON=ON` and [`readelf`](https://man7.org/linux/man-pages/man1/readelf.1.html) available) |
| `package-pinocchio` | Assembles a redistributable prebuilt Pinocchio package; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) below (unavailable when `SSQP_PREBUILT_PINOCCHIO_DIR` is set) |
| `package-hpipm` | Assembles a redistributable prebuilt HPIPM/BLASFEO package; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) below (`SSQP_WITH_HPIPM=ON`, unavailable when `SSQP_PREBUILT_HPIPM_DIR` is set) |
| `package-piqp` | Assembles a redistributable prebuilt PIQP package, which also holds BLASFEO when `SSQP_WITH_HPIPM=OFF`; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) below (`SSQP_WITH_PIQP=ON`, unavailable when `SSQP_PREBUILT_PIQP_DIR` is set) |

## Building against prebuilt dependencies

Building Pinocchio (and, if `SSQP_WITH_HPIPM` is on, HPIPM/BLASFEO) from source is the default, but that isn't always
practical, since the target machine may lack network access, or its CPU/toolchain combination may trip up one of
their builds. As an alternative, SSQP provides prebuilt packages for Pinocchio, HPIPM/BLASFEO and PIQP that a build
can link against directly instead of building them from source.

<!-- TODO: where to get the prebuilt packages -->

Point a build at them:

```bash
mkdir build && cd build
cmake -DSSQP_WITH_HPIPM=ON \
      -DSSQP_PREBUILT_PINOCCHIO_DIR=/path/to/prebuilt/pinocchio \
      -DSSQP_PREBUILT_HPIPM_DIR=/path/to/prebuilt/hpipm \
      -DSSQP_PREBUILT_PIQP_DIR=/path/to/prebuilt/piqp \
      ..
cmake --build .
```

This build only compiles ssqp itself, needing no network access and no Pinocchio/urdfdom/HPIPM/BLASFEO compiler
toolchain. The Pinocchio package also bundles the Boost headers and static libs it was built against (`ssqp`'s own
sources include Pinocchio headers that use Boost types directly, not just precompiled code, so this is needed even
though the linking itself is static), so `SSQP_PREBUILT_PINOCCHIO_DIR` can be used without Boost installed at all
either: unlike the default from-source build (see [Dependencies](#dependencies) above), this mode never has to
detect, fetch, or build Boost itself. `SSQP_PREBUILT_HPIPM_DIR` can be set independently of
`SSQP_PREBUILT_PINOCCHIO_DIR` and vice versa; each falls back to building from source if left unset. The same goes
for `SSQP_PREBUILT_PIQP_DIR`, except that a PIQP package only holds BLASFEO when it was made with
`SSQP_WITH_HPIPM=OFF`. With HPIPM on, PIQP uses the BLASFEO from the HPIPM build or package, so a PIQP package made
that way only works for builds that also have HPIPM on.

A prebuilt package is tied to the architecture, compiler, and (for HPIPM/BLASFEO) CPU microarchitecture it was built
on. It isn't a portable, run-anywhere artifact, just a way to build once and reuse the result across machines with
matching setups.

!!! note
    Instead of using a package we provide, you can also produce your own. On a machine that can build from source
    matching your target architecture, a build can package what it just built into a redistributable directory:
    `cmake --build . --target package-pinocchio`, `cmake --build . --target package-hpipm` and
    `cmake --build . --target package-piqp` produce `build/prebuilt/pinocchio/{include,lib}`,
    `build/prebuilt/hpipm/{include,lib}` and `build/prebuilt/piqp` respectively. Copy any of these directories to
    another machine and point `SSQP_PREBUILT_PINOCCHIO_DIR`/`SSQP_PREBUILT_HPIPM_DIR`/`SSQP_PREBUILT_PIQP_DIR` at it,
    same as above.

## Using the C++ library

SSQP is a static library, `libssqp.a`, exposed to CMake as the `ssqp::ssqp` target. You can install a prebuilt
release, install your own build, or build SSQP as part of your project. In every case your code is compiled with
SSQP's architecture flag (`-march=native` for your own builds, see `SSQP_MARCH_NATIVE`), since Eigen's memory
alignment must match between SSQP and the code calling it.

### Installing a release

Each [GitHub release](https://github.com/upatras-lar/ssqp/releases) provides prebuilt packages for Linux x86_64.
Like the pip wheels, they are built with `-march=x86-64-v3` and need a CPU with AVX2 and FMA, and they include the
HPIPM QP solver.

On Ubuntu 22.04 or 24.04, download the `.deb` for your Ubuntu version and install it with apt, which also installs
Eigen:

```bash
sudo apt install ./libssqp-dev_{{ ssqp_version }}-1-ubuntu24.04_amd64.deb
```

Your project then finds it with `find_package(ssqp)`, as shown [below](#installing-your-own-build).

On other distributions, use the `ssqp-v{{ ssqp_version }}-linux-x86_64.tar.gz` archive, which also contains the
Python module, the examples and the [prebuilt dependency packages](#building-against-prebuilt-dependencies). Extract
it and pass its directory to your project with `-DCMAKE_PREFIX_PATH`, or let CMake download it:

```cmake
include(FetchContent)
FetchContent_Declare(ssqp
    URL https://github.com/upatras-lar/ssqp/releases/download/v{{ ssqp_version }}/ssqp-v{{ ssqp_version }}-linux-x86_64.tar.gz
)
FetchContent_MakeAvailable(ssqp)
find_package(ssqp REQUIRED PATHS ${ssqp_SOURCE_DIR} NO_DEFAULT_PATH)

target_link_libraries(my_app PRIVATE ssqp::ssqp)
```

The archive needs Eigen 3.4 on the system.

### Installing your own build

Install the library, its headers and a CMake package into a prefix of your choice:

```bash
cmake --install build --component cpp --prefix ~/.local
```

This installs `libssqp.a` under `lib/`, the static libraries it links (Pinocchio, its URDF parser, Boost and, with
`SSQP_WITH_HPIPM=ON`, HPIPM/BLASFEO) under `lib/ssqp/`, the headers under `include/ssqp/` and the CMake package
under `lib/cmake/ssqp/`. Your project then finds it with:

```cmake
find_package(ssqp REQUIRED)
target_link_libraries(my_app PRIVATE ssqp::ssqp)
```

Pass `-DCMAKE_PREFIX_PATH=~/.local` when configuring your project if the prefix isn't one CMake searches by
default. The package needs Eigen3 on the system, compatible with the version SSQP was built against, but no
Pinocchio, Boost or HPIPM.

### Adding SSQP to a CMake project

Alternatively, build SSQP as part of your project with
[`FetchContent`](https://cmake.org/cmake/help/latest/module/FetchContent.html) (or `add_subdirectory` on a checkout):

```cmake
include(FetchContent)
FetchContent_Declare(ssqp
    GIT_REPOSITORY https://github.com/upatras-lar/ssqp.git
    GIT_TAG v{{ ssqp_version }}
)
FetchContent_MakeAvailable(ssqp)

target_link_libraries(my_app PRIVATE ssqp::ssqp)
```

The [CMake options](#cmake-options) above work the same here, set before `FetchContent_MakeAvailable` or on the
command line. `SSQP_BUILD_TESTS` and `SSQP_BUILD_PYTHON` default to `OFF` in this case. The first build compiles
Pinocchio as described in [Dependencies](#dependencies), unless `SSQP_PREBUILT_PINOCCHIO_DIR` points to a
[prebuilt package](#building-against-prebuilt-dependencies).

## Using the Python bindings

Once built with `SSQP_BUILD_PYTHON=ON` (the default), the `ssqpy` extension module is produced under `build/bindings/`.
There are two ways to make it importable:

**Install it** into whichever Python interpreter it was built against (the one `find_package(Python)` picked up at
configure time; pass `-DPython_EXECUTABLE=/path/to/python` to pin a specific one, e.g. a virtualenv's):

```bash
cmake --install build --component python
```

This drops `ssqpy` (plus its type stubs) directly into that interpreter's `site-packages`, so `import ssqpy` works
from anywhere, in any script or shell, with no environment setup. This is the same interpreter build already needs
NumPy in (see [Dependencies](#dependencies) above). Installing to a different one than you built against risks an
ABI mismatch, since `ssqpy` is not a portable, version-independent artifact.

**Or, without installing**, add `build/bindings` directly to your `PYTHONPATH` for the current shell only:

```bash
export PYTHONPATH="$(pwd)/build/bindings:${PYTHONPATH}"
```

The [`examples/`](https://github.com/upatras-lar/ssqp/tree/main/examples) directory provides a
[`setup_env.sh`](https://github.com/upatras-lar/ssqp/blob/main/examples/setup_env.sh) script that does this (and
sets up other example-specific environment variables); source it from within `examples/` before running any Python
example:

```bash
cd examples
source setup_env.sh
python trajgen/python/di.py
```

See [Quickstart](quickstart.md) to write your own script against `ssqpy`.
