# Installation

Building SSQP is a work in progress: there is no packaged release yet, and the library must be built from source.

## Dependencies

SSQP only expects a couple of dependencies to already be present on the system:

* [Eigen3](https://eigen.tuxfamily.org/): linear algebra
* [NumPy](https://numpy.org/): only required if `BUILD_PYTHON` is enabled (the default); `ssqpy`'s bindings convert
  between Eigen and NumPy arrays, so NumPy must be importable in whatever Python interpreter CMake picks up (via
  `find_package(Python)`; pass `-DPython_EXECUTABLE=/path/to/python` to `cmake` to pin a specific one) both when
  building and when later using `ssqpy`

!!! tip
    Building Pinocchio (and, if enabled, HPIPM/BLASFEO) from source is the default, but SSQP also supports building
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
* [nanobind](https://github.com/wjakob/nanobind): Python bindings, fetched automatically if `BUILD_PYTHON` is
  enabled (the default)
* [BLASFEO](https://github.com/giaf/blasfeo) and [HPIPM](https://github.com/giaf/hpipm): fetched automatically if
  the `WITH_HPIPM` CMake option is enabled, for the HPIPM-based QP solver (see [QP Solvers](../concepts/qp-solvers.md))
* [nlohmann/json](https://github.com/nlohmann/json) and [Tracy](https://github.com/wolfpld/tracy): vendored under
  [`external/`](https://github.com/upatras-lar/ssqp/tree/main/external) as git submodules

Make sure to clone submodules along with the repository:

```bash
git clone --recurse-submodules git@github.com:upatras-lar/ssqp.git
```

The dependencies fetched by CMake (see [`cmake/`](https://github.com/upatras-lar/ssqp/tree/main/cmake)) require
network access during the build. The first build will take noticeably longer than subsequent ones since it has to
compile Pinocchio and, if enabled, HPIPM/BLASFEO from source; later builds only recompile them if their vendored
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
keeps the platform's C/C++ runtime as a shared dependency. With `BUILD_PYTHON=ON` (the default) and
[`readelf`](https://man7.org/linux/man-pages/man1/readelf.1.html) available, verify this for a completed build with:

```bash
cmake --build . --target verify-static-dependencies
```

### CMake options

| Option | Default | Description |
| --- | --- | --- |
| `WITH_HPIPM` | `OFF` | Fetch, build and link against HPIPM and BLASFEO to enable the [HPIPM QP solver](../concepts/qp-solvers.md) |
| `SSQP_PREBUILT_PINOCCHIO_DIR` | *(empty)* | Path to a prebuilt Pinocchio package instead of building it from source; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) |
| `SSQP_PREBUILT_HPIPM_DIR` | *(empty)* | Path to a prebuilt HPIPM/BLASFEO package instead of building it from source (only relevant if `WITH_HPIPM=ON`) |
| `SSQP_BOOST_VERSION` | `1.86.0` | Boost release fetched when no system Boost headers are available (see [Dependencies](#dependencies) above) |
| `ENABLE_PROFILING` | `OFF` | Enable [Tracy](https://github.com/wolfpld/tracy) profiling |
| `DISABLE_LOGGING` | `OFF` | Compile out SSQP's logging (see `ssqp::set_log_level`) |
| `BUILD_EXAMPLES` | `OFF` | Build the C++ example executables under `examples/trajgen/c++` |
| `BUILD_TESTS` | `ON` | Build the unit test suite under `tests/` |
| `BUILD_PYTHON` | `ON` | Build the `ssqpy` Python extension via nanobind |

Pass options as `-D<OPTION>=<VALUE>` to `cmake`, e.g.:

```bash
cmake -DWITH_HPIPM=ON -DBUILD_EXAMPLES=ON ..
```

### CMake targets

Build a specific target instead of everything with `cmake --build . --target <target>`:

| Target | Description |
| --- | --- |
| `ssqp` | The static C++ library; always built |
| `ssqpy` | The `ssqpy` Python extension module (`BUILD_PYTHON=ON`, the default) |
| `run_tests` | The unit/regression test executable under `tests/` (`BUILD_TESTS=ON`, the default) |
| *(per-file, e.g.* `analytical-di`*)* | One executable per C++ example under `examples/trajgen/c++` (`BUILD_EXAMPLES=ON`); named after the file's path relative to that directory, with `/` replaced by `-` |
| `install` | Installs the built `ssqpy` (plus its type stubs) into the configured Python interpreter's `site-packages`; see [Using the Python bindings](#using-the-python-bindings) |
| `verify-static-dependencies` | Confirms `ssqp`/`ssqpy` link no shared Boost, Pinocchio, URDF, HPIPM, or BLASFEO libraries (`BUILD_PYTHON=ON` and [`readelf`](https://man7.org/linux/man-pages/man1/readelf.1.html) available) |
| `package-pinocchio` | Assembles a redistributable prebuilt Pinocchio package; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) below (unavailable when `SSQP_PREBUILT_PINOCCHIO_DIR` is set) |
| `package-hpipm` | Assembles a redistributable prebuilt HPIPM/BLASFEO package; see [Building against prebuilt dependencies](#building-against-prebuilt-dependencies) below (`WITH_HPIPM=ON`, unavailable when `SSQP_PREBUILT_HPIPM_DIR` is set) |

## Building against prebuilt dependencies

Building Pinocchio (and, if `WITH_HPIPM` is on, HPIPM/BLASFEO) from source is the default, but that isn't always
practical, since the target machine may lack network access, or its CPU/toolchain combination may trip up one of
their builds. As an alternative, SSQP provides prebuilt packages for Pinocchio and HPIPM/BLASFEO that a build can
link against directly instead of building either from source.

<!-- TODO: where to get the prebuilt packages -->

Point a build at them:

```bash
mkdir build && cd build
cmake -DWITH_HPIPM=ON \
      -DSSQP_PREBUILT_PINOCCHIO_DIR=/path/to/prebuilt/pinocchio \
      -DSSQP_PREBUILT_HPIPM_DIR=/path/to/prebuilt/hpipm \
      ..
cmake --build .
```

This build only compiles ssqp itself, needing no network access and no Pinocchio/urdfdom/HPIPM/BLASFEO compiler
toolchain. The Pinocchio package also bundles the Boost headers and static libs it was built against (`ssqp`'s own
sources include Pinocchio headers that use Boost types directly, not just precompiled code, so this is needed even
though the linking itself is static), so `SSQP_PREBUILT_PINOCCHIO_DIR` can be used without Boost installed at all
either: unlike the default from-source build (see [Dependencies](#dependencies) above), this mode never has to
detect, fetch, or build Boost itself. `SSQP_PREBUILT_HPIPM_DIR` can be set independently of
`SSQP_PREBUILT_PINOCCHIO_DIR` and vice versa; each falls back to building from source if left unset.

A prebuilt package is tied to the architecture, compiler, and (for HPIPM/BLASFEO) CPU microarchitecture it was built
on. It isn't a portable, run-anywhere artifact, just a way to build once and reuse the result across machines with
matching setups.

!!! note
    Instead of using a package we provide, you can also produce your own. On a machine that can build from source
    matching your target architecture, a build can package what it just built into a redistributable directory:
    `cmake --build . --target package-pinocchio` and `cmake --build . --target package-hpipm` produce
    `build/prebuilt/pinocchio/{include,lib}` and `build/prebuilt/hpipm/{include,lib}` respectively. Copy either
    directory to another machine and point `SSQP_PREBUILT_PINOCCHIO_DIR`/`SSQP_PREBUILT_HPIPM_DIR` at it, same as
    above.

## Using the Python bindings

Once built with `BUILD_PYTHON=ON` (the default), the `ssqpy` extension module is produced under `build/bindings/`.
There are two ways to make it importable:

**Install it** into whichever Python interpreter it was built against (the one `find_package(Python)` picked up at
configure time; pass `-DPython_EXECUTABLE=/path/to/python` to pin a specific one, e.g. a virtualenv's):

```bash
cmake --install build
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
