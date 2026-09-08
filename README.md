# SSQP

SSQP is a structure-exploiting Sequential Quadratic Programming (SQP) solver
for nonlinear optimal control problems. It is designed for trajectory
generation and real-time Model Predictive Control (MPC), with native support
for manifold-valued states such as floating-base orientations.

SSQP solves problems of the form

$$
\begin{aligned}
\min_{x,u}\quad &\sum_{k=0}^{H-1} \ell_k(x_k,u_k) + \ell_H(x_H) \\
\text{subject to}\quad &x_{k+1}=f_k(x_k,u_k), &&0\leq k < H,\\
&c_k(x_k,u_k)\geq 0, &&0\leq k < H,\\
&c_H(x_H)\geq 0.
\end{aligned}
$$

At every SQP iteration, SSQP linearizes the dynamics and constraints and
quadratizes the costs to form a stagewise, banded Quadratic Program (QP). A
line search uses the QP solution to update the nonlinear trajectory.

The full documentation is available at
[150.140.184.35:8000](http://150.140.184.35:8000), including the
[installation guide](http://150.140.184.35:8000/getting-started/installation/),
[Python quickstart](http://150.140.184.35:8000/getting-started/quickstart/),
concepts, examples, and generated [Python](http://150.140.184.35:8000/reference/)
and [C++](http://150.140.184.35:8000/cpp-reference/) API references.

## Features

- Structure-exploiting SQP for finite-horizon optimal control.
- Euclidean and manifold-valued state components.
- Custom or predefined dynamics, costs, constraints, and terrain models.
- Kinematics, ForwardDynamics, InverseDynamics, and ReducedInverseDynamics
  formulations.
- ADMM QP solver by default, with optional HPIPM/BLASFEO support.
- Receding-horizon MPC with warm starts and bounded per-step iteration counts.
- C++17 library and Python bindings through nanobind.
- Deterministic regression, finite-difference, and sanitizer-enabled tests.

## Repository layout

| Path | Contents |
| --- | --- |
| `include/`, `src/` | C++ library headers and implementation |
| `python/` | `ssqpy` Python bindings |
| `examples/trajgen/` | Offline trajectory-generation examples |
| `examples/mujoco_mpc/` | MPC examples using MuJoCo |
| `examples/real_mpc/` | MPC examples for real systems |
| `examples/cloud_pendulum/` | Adaptive-model pendulum examples |
| `examples/benchmarks/` | Solver and formulation benchmarks |
| `examples/visualizer/` | Trajectory visualization utilities |
| `examples/models/` | Custom model implementations and utilities |
| `examples/assets/` | URDF and MJCF model assets |
| `tests/` | Unit, regression, gradient, and sanitizer test infrastructure |
| `external/` | Repository-vendored dependencies such as JSON and Tracy |
| `cmake/` | Dependency and packaging build logic |

## Dependencies

The default source build requires:

- A C++17 compiler
- CMake 3.22 or newer
- Eigen3
- Boost headers and static Filesystem and Serialization archives, when
  available. If the headers or either static archive is unavailable, CMake
  fetches the matching Boost sources and builds PIC static archives
  automatically.
- Git and network access for the first build
- Python 3.10 or newer and NumPy when Python bindings are enabled

Pinocchio and its URDF parser dependencies are fetched and built by CMake.
Nanobind is fetched when the Python bindings are enabled. HPIPM and BLASFEO
are fetched only when `WITH_HPIPM=ON`.

Clone the repository with its submodules:

```sh
git clone --recurse-submodules https://github.com/upatras-lar/ssqp.git
cd ssqp
```

## Building

Configure and build from the repository root:

```sh
cmake -S . -B build
cmake --build build -j"$(nproc)"
```

The first build compiles the pinned external dependencies and can take some
time. Reusing the same build directory keeps those dependencies incremental;
subsequent builds rebuild only targets affected by changed source files.

The default configuration builds the Python bindings and tests. The compiled
library is placed under `build/lib`, the Python module under
`build/bindings`, and the test runner under `build/tests`.

The Boost components used by Pinocchio are always linked statically. The
`ssqpy` extension still uses the normal platform C and C++ runtime libraries,
but it does not depend on shared Boost, Pinocchio, URDF, HPIPM, or BLASFEO
libraries. Verify this for a completed build with:

```sh
cmake --build build --target verify-static-dependencies
```

### CMake options

Pass options during configuration with `-DNAME=VALUE`:

| Option | Default | Description |
| --- | --- | --- |
| `WITH_HPIPM` | `OFF` | Build and enable the HPIPM/BLASFEO QP backend |
| `BUILD_PYTHON` | `ON` | Build the `ssqpy` Python module |
| `BUILD_TESTS` | `ON` | Build the unit and regression tests |
| `BUILD_EXAMPLES` | `OFF` | Build the C++ trajectory-generation examples |
| `ENABLE_PROFILING` | `OFF` | Enable Tracy profiling support |
| `DISABLE_LOGGING` | `OFF` | Compile out SSQP logging |
| `ENABLE_SANITIZERS` | `OFF` | Enable AddressSanitizer and UndefinedBehaviorSanitizer |
| `SSQP_BOOST_VERSION` | `1.86.0` | Boost release fetched when no system Boost headers are available |

For example, to enable HPIPM and the C++ examples:

```sh
cmake -S . -B build -DWITH_HPIPM=ON -DBUILD_EXAMPLES=ON
cmake --build build -j"$(nproc)"
```

The build uses `-march=native`, so generated binaries and prebuilt dependency
packages are tied to the architecture on which they were built.

### Prebuilt dependencies

Pinocchio and, when HPIPM is enabled, HPIPM/BLASFEO can be supplied as
prebuilt packages instead of being compiled from source:

```sh
cmake -S . -B build \
  -DWITH_HPIPM=ON \
  -DSSQP_PREBUILT_PINOCCHIO_DIR=/path/to/prebuilt/pinocchio \
  -DSSQP_PREBUILT_HPIPM_DIR=/path/to/prebuilt/hpipm
cmake --build build -j"$(nproc)"
```

On a machine that has already built the dependencies from source, create the
packages with:

```sh
cmake --build build --target package-pinocchio
cmake --build build --target package-hpipm
```

The resulting packages are written to `build/prebuilt/pinocchio` and
`build/prebuilt/hpipm`. They must match the target compiler, architecture, and
CPU features.

## Python bindings

Install the built module into the Python environment selected by CMake:

```sh
cmake --install build
python -c "import ssqpy; print('ssqpy imported successfully')"
```

For a build-local setup, use `PYTHONPATH` instead:

```sh
export PYTHONPATH="$PWD/build/bindings:$PYTHONPATH"
python -c "import ssqpy; print('ssqpy imported successfully')"
```

The examples provide a helper that sets this path:

```sh
cd examples
source setup_env.sh
python trajgen/python/di.py
```

The `di.py` example defines a double-integrator dynamics model and quadratic
tracking cost, constructs an `ssqpy.model.Model`, and solves it with
`ssqpy.solvers.ssqp`. Custom dynamics subclass
`ssqpy.model.dynamics.StageDynamics`; custom costs subclass
`ssqpy.model.costs.StageCost`.

## Solvers and MPC

The default QP backend is SSQP's ADMM implementation. HPIPM can be selected
after configuring with `-DWITH_HPIPM=ON`:

```python
params = ssqpy.solvers.ssqpParams()
params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
solver = ssqpy.solvers.ssqp(model, params)
```

For receding-horizon control, `ssqpy.solvers.MPC` repeatedly solves the OCP
with a small iteration budget, applies the first control, feeds back the
current state, and warm-starts the next solve:

```python
mpc = ssqpy.solvers.MPC(model, sqp_iters=5, qp_iters=50)
mpc.reinitialize(initial_state)

while running:
    state = read_state()
    x, u = mpc.step(state)
    apply_control(u.stage(0))
```

See the [MPC concept guide](http://150.140.184.35:8000/concepts/mpc/) and
`examples/mujoco_mpc/` for complete control loops.

## Examples

The [examples README](examples/README.md) describes the available examples.
They cover offline trajectory generation, simulated MPC, real-system MPC,
adaptive pendulum models, benchmarks, and trajectory visualization.

The documentation includes walkthroughs for:

- Pendubot swing-up
- Panda circle trajectory generation
- Go2 jumping with contact scheduling
- Panda circle tracking with MuJoCo MPC

## Testing

Build and run the standard test suite:

```sh
cmake -S . -B build -DWITH_HPIPM=ON
cmake --build build --target run_tests -j"$(nproc)"
ctest --test-dir build --output-on-failure -j"$(nproc)"
```

CTest discovers each test case separately, so a crash or timeout in one case
does not prevent the remaining cases from running. To run a subset directly:

```sh
build/tests/run_tests --list
build/tests/run_tests --filter Regression_
build/tests/run_tests --filter gammafGradTest --seed 42
```

Failed test diagnostics are written to `tests/logs/<test-name>.txt`. See
[`tests/README.md`](tests/README.md) for the correctness coverage, deterministic
seeding, finite-difference checks, sanitizer builds, and leak-detection notes.

## Development status

SSQP is under active development. The source API, Python bindings, build
system, and documentation may change as the library is restructured for
performance, cache locality, and maintainability.
