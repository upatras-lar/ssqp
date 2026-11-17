# Changelog

Each release on GitHub takes its notes from the matching section below, so
every new version needs a `## [X.Y.Z] - YYYY-MM-DD` section before its tag is
pushed.

## [0.2] - 2026-11-17

### Breaking changes
- C++ custom costs, constraints and dynamics use a new function interface: inputs are `ConstVecXRef` and outputs are written into arguments, e.g. `evaluate(x, u, out)` instead of returning a `VecX`, and `state_right_plus`/`state_right_minus` likewise. See [Inputs and outputs in C++](https://lar.upatras.gr/ssqp/concepts/costs-constraints-dynamics/#inputs-and-outputs-in-c)
- `StageDynamics::init_state()` and `init_control()` return a const reference
- In Python, the `x`, `u` and multiplier arguments passed to custom functions are read-only views that are only valid during the call, so copy them to keep or modify them
- `SquaredIDTorqueCost` no longer takes a stage `k`. It uses the forces of the stage it is added to, so one instance can only be shared between stages with the same forces
- `Model.getFullAccelerationVector` and `Model.getActuatedTorqueVector` take the stage's `forces` and `wrenches` (see the new `Model.getForces` and `Model.getWrenches`) instead of a stage `k`
- The filter line search only accepts a step that strictly improves cost, gap or constraint violation, so solves can take a different number of iterations than before

### Added
- PIQP as a third QP backend (`QPSolverType.PIQP`), built by default (`SSQP_WITH_PIQP`), with an optional parallel multistage solver (`SSQP_PIQP_OPENMP`)
- `reference_frame` option for frame-based costs and constraints (`ReferenceFrame.World`, `Local`, `Base`, fixed rotations, `ReferenceFrame.yaw` and `ReferenceFrame.normal`), and a `center` option for position bounds
- `ContactConstraint.set_terrain` and `FrameSquaredTerrainHeightErrorCost.set_terrain`, to use a terrain other than the model's
- Optional `state_only_rows()` for constraints: rows that only depend on the state are skipped at stage 0, so an initial state that violates them no longer makes the QP infeasible
- Optional `box_rows()` for constraints, which HPIPM handles as simple bounds instead of general constraints
- `skip_last_termination_check` solver parameter
- `add_gradient`, `add_hessian_*`, `add_hessprod_*` and `*_into` methods on costs, constraints and dynamics, which write into already sized outputs
- `panda_look_at` trajectory generation example
- Go2 real MPC: LIDAR elevation map, terrain-aware footholds and base tilt, forward jump, lift detection and actuator delay compensation

### Changed
- Faster solves: Pinocchio results are cached and shared between costs, constraints and dynamics, cost and constraint merging is cheaper, and the solver reuses its memory between iterations and between MPC steps instead of allocating
- MPC adapts its solver to new dimensions (for example when contacts change) instead of rebuilding it
- The line search reuses the metrics of the point it accepts, and the QP after the last SQP iteration is skipped
- Documentation gallery pages for simulated and real MPC, and PIQP in the solver benchmarks

### Fixed
- `MPC.reinitialize` now discards the previous solution, which was otherwise still used to warm-start the next step
- `ssqp.write` no longer reads an uninitialized floating base flag for models without XML or custom properties

## [0.1.2] - 2026-10-14

### Added
- CMake install option for the C++ library
- Debian packages of the C++ library for Ubuntu 22.04 and 24.04, attached to each release
- Terrain reading with LIDAR in the Go2 real MPC controller

### Changed
- The `-O3` flag is no longer propagated to projects that link against ssqp

## [0.1.1] - 2026-10-12

### Added
- `CITATION.cff` for citing ssqp
- Documentation site build workflow

### Changed
- Updated README and documentation

## [0.1] - 2026-10-12

First public release.
