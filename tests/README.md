# Regression tests

Build and run from the repository root:

```sh
cmake -S . -B build -DWITH_HPIPM=ON
cmake --build build --target run_tests -j4
ctest --test-dir build --output-on-failure -j4
```

CTest discovers cases from `run_tests --list` and starts a separate process for
 each case. A crash, timeout, exception, failed assertion, or failed numerical
check fails the case; other cases still run. Failed tests write their diagnostics
to `tests/logs/<test-name>.txt`; a stale log is removed when that test passes.

To run a subset or reproduce a random sample:

```sh
build/tests/run_tests --list
build/tests/run_tests --filter Regression_
build/tests/run_tests --filter gammafGradTest --seed 42
ctest --test-dir build -R 'Regression_(QP|MPC)' --output-on-failure
```

The base seed defaults to 42. Each case derives a stable private seed from its
name, so its samples are unchanged when run alone, reordered, or run alongside
new tests. The runner prints both seeds. Pass the **base seed** to `--seed` to
reproduce a run. An unmatched filter exits with failure. `--filter` matches a
substring. Build with `WITH_HPIPM=OFF` to run only backend-independent and ADMM
cases.

## Correctness coverage

* Existing manifold derivative checks cover robot dynamics, costs and
  constraints, solver modes, integration schemes, floating bases, unactuated
  joints and external forces. Samples use deterministic tangent perturbations
  of valid initial states, preserving manifold constraints.
* Scalar integrator problems check SSQP and every MPC initialization/step
  overload against an independent scalar Riccati solution at horizons 1, 2 and
  8. They also check active control bounds, objective values, dynamics
  feasibility, feedback changes, convergence history and solver reinitialization.
* Small QPs are compared to a dense active-set enumeration oracle. Checks
  include nonsymmetric cross Hessians, affine dynamics, terminal constraints,
  finite and infinite bounds, KKT stationarity and complementarity. Repeated
  solves change the data while retaining the solver object.
* MPC constraints are repeatedly disabled/enabled, with and without an explicit
  caller-side dimension update. Repeated solver construction and destruction
  exercise ownership and resizing under sanitizers.
* Quadratic cost Hessians are compared to differentiated gradients, including
  terminal stages with no controls. FlatVec checks cover heterogeneous stage
  sizes, empty stages, copies, assignment and arithmetic.
* Comparison helpers explicitly reject nonfinite values and wrong shapes.

The gradient checks use **componentwise** absolute and relative tolerances,
independent of the perturbation step. Scalar finite differences refine their
stencil using only function values, then apply Richardson extrapolation when
estimates agree. This reduces truncation error and stencils crossing the kinks
in `abs` and bound-clipping penalties. A scaled tolerance avoids demanding the
same absolute accuracy from derivatives near zero and derivatives in the
hundreds. Exact kink points do not have a unique classical derivative; these
checks target differentiable points rather than defining a new subgradient
contract.

The large archived JSON QP benchmarks remain outside the default suite. The
small independent QP oracles provide fast, explicit correctness checks.

## Memory and undefined behavior

Use a separate build so optimized artifacts remain available:

```sh
cmake -S . -B build/sanitized \
  -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON \
  -DBUILD_PYTHON=OFF -DWITH_HPIPM=ON
cmake --build build/sanitized --target run_tests -j2
ctest --test-dir build/sanitized --output-on-failure -j2
```

`ENABLE_SANITIZERS` instruments the library and test executable with AddressSanitizer
and UndefinedBehaviorSanitizer, debug symbols and frame pointers. CTest enables
leak detection and makes sanitizer findings fatal. LeakSanitizer is available
with AddressSanitizer on supported platforms such as Linux. Use GCC or Clang.
Third-party Pinocchio, HPIPM and BLASFEO builds keep their own compiler settings;
accesses inside uninstrumented third-party code and assembly are not fully
covered. Allocations that survive process shutdown can still be reported as
leaks. No leak suppressions are configured.

These checks cover exercised paths; they cannot prove absence of all memory
bugs or enforce a wall-clock performance budget. Add a small known-solution
case whenever restructuring introduces a new representation or fixes a bug.
