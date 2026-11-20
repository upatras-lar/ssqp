# C++ API Reference

SSQP is written in C++17; the `ssqpy` Python module documented in the [Python API Reference](../reference/index.md)
is a near 1:1 [nanobind](https://github.com/wjakob/nanobind) wrapping of the library described here. The library can
be used directly from C++ without the Python bindings at all: link against the `ssqp` CMake target (see
[Installation](../getting-started/installation.md)) and include the relevant headers under
[`include/`](https://github.com/upatras-lar/ssqp/tree/main/include).

This reference is generated from the headers' own comments at doc build time, with one page per namespace, so it can't
drift out of sync with the source. It does not enumerate every predefined cost/constraint constructor argument, since
those mirror the [Python bindings](../concepts/costs-constraints-dynamics.md) closely; consult the corresponding
predefined header or the [Python reference](../reference/ssqpy/model/costs.md) for exact parameters.

## Main classes

| Class | Header(s) | Concepts |
| --- | --- | --- |
| [`ssqp::model::Model`](ssqp/model/index.md#ssqp.model.Model), with its nested `Structure`, `Config` and `CustomProperties` | `include/model/Model.hpp` | [Model](../concepts/model.md) |
| [`ssqp::model::Container`](ssqp/model/index.md#ssqp.model.Container) | `include/model/Container.hpp` | |
| [`Function`](ssqp/model/stagewise/index.md#ssqp.model.stagewise.Function), [`StageCost`](ssqp/model/stagewise/index.md#ssqp.model.stagewise.StageCost), [`StageDynamics`](ssqp/model/stagewise/index.md#ssqp.model.stagewise.StageDynamics), [`StageConstraints`](ssqp/model/stagewise/index.md#ssqp.model.stagewise.StageConstraints) (`ssqp::model::stagewise`) | `include/model/stagewise/Function.hpp`, `StageCost.hpp`, `StageDynamics.hpp`, `StageConstraints.hpp` | [Costs, Constraints & Dynamics](../concepts/costs-constraints-dynamics.md) |
| [Predefined costs](ssqp/model/stagewise/predefined/costs.md) and [constraints](ssqp/model/stagewise/predefined/constraints.md) | `include/model/stagewise/predefined/Costs.hpp`, `Constraints.hpp` | [Costs, Constraints & Dynamics](../concepts/costs-constraints-dynamics.md) |
| [`ssqp::model::stagewise::predefined::ReferenceFrame`](ssqp/model/stagewise/predefined/index.md#ssqp.model.stagewise.predefined.ReferenceFrame) | `include/types.hpp` | [Reference frames](../concepts/costs-constraints-dynamics.md#reference-frames) |
| [`Terrain`](ssqp/model/terrain/index.md#ssqp.model.terrain.Terrain) and the [predefined terrains](ssqp/model/terrain/predefined.md) (`ssqp::model::terrain`) | `include/model/terrain/Terrain.hpp`, `include/model/terrain/predefined/` | [Terrain](../concepts/terrain.md) |
| [`ssqp::model::presets`](ssqp/model/presets.md) | `include/model/presets/Presets.hpp` | |
| [`ssqp::solvers::ssqp`](ssqp/solvers.md#ssqp.solvers.ssqp), with its nested `Params`, `Guess` and convergence data | `include/solvers/ssqp.hpp` | [SSQP Solver](../concepts/solver.md) |
| [Filter and merit line search](ssqp/step.md) (`ssqp::step`) | `include/sqp_step/filter.hpp`, `include/sqp_step/merit.hpp` | [SSQP Solver](../concepts/solver.md) |
| [`QPSolver`](ssqp/solvers.md#ssqp.solvers.QPSolver), [`admm_qp`](ssqp/solvers.md#ssqp.solvers.admm_qp), [`hpipm_qp`](ssqp/solvers.md#ssqp.solvers.hpipm_qp), [`piqp_qp`](ssqp/solvers.md#ssqp.solvers.piqp_qp), [`lqr`](ssqp/solvers.md#ssqp.solvers.lqr) (`ssqp::solvers`) | `include/solvers/qp.hpp`, `admm_qp.hpp`, `hpipm_qp.hpp`, `piqp_qp.hpp`, `lqr.hpp` | [QP Solvers](../concepts/qp-solvers.md) |
| [`ssqp::solvers::MPC`](ssqp/solvers.md#ssqp.solvers.MPC) | `include/solvers/mpc.hpp` | [MPC](../concepts/mpc.md) |
| [`ssqp::param_search`](ssqp/param_search/index.md): dynamics parameter search (`dynamics_params::Search`, `GradientDescent`, `GaussNewton`) and `HyperParamSearch` | `include/param_search/search.hpp` | [Parameter Search](../concepts/parameter-search.md) |
| [Core types](ssqp/index.md) (`VecX`, `MatX`, `FlatVec`, log levels, …) | `include/types.hpp` | |

## Namespaces

All library code lives under `ssqp::`, mirroring the Python package layout:

* [`ssqp`](ssqp/index.md): core types (`VecX`, `MatX`, `FlatVec`, …), log levels and their helpers
* [`ssqp::model`](ssqp/model/index.md): `Model`, `Container`, `Dimensions`
* [`ssqp::model::presets`](ssqp/model/presets.md): URDFs baked into the binary at build time, looked up by
  `Model::Structure::from_preset`
* [`ssqp::model::stagewise`](ssqp/model/stagewise/index.md): `StageCost`, `StageDynamics`, `StageConstraints`, `Function`
* [`ssqp::model::stagewise::factory`](ssqp/model/stagewise/factory.md): stage functions made from functors
* [`ssqp::model::stagewise::predefined`](ssqp/model/stagewise/predefined/index.md): `ReferenceFrame`, with the predefined
  [costs](ssqp/model/stagewise/predefined/costs.md) and [constraints](ssqp/model/stagewise/predefined/constraints.md)
  in their own namespaces
* [`ssqp::model::terrain`](ssqp/model/terrain/index.md): `Terrain`, with the
  [predefined terrains](ssqp/model/terrain/predefined.md) in their own namespace
* [`ssqp::solvers`](ssqp/solvers.md): `ssqp`, `MPC`, `QPSolver`, `admm_qp`, `hpipm_qp`, `piqp_qp`, `lqr`
* [`ssqp::step`](ssqp/step.md): the filter and merit line search implementations
* [`ssqp::param_search`](ssqp/param_search/index.md): hyperparameter search, with dynamics parameter search in
  [`dynamics_params`](ssqp/param_search/dynamics_params.md)

## Building against the C++ library

`ssqp` is built as a static library by CMake. Other CMake projects link the `ssqp::ssqp` target, either from an
installed package found with `find_package(ssqp)` or by adding this repository with `FetchContent` or
`add_subdirectory`; the target propagates its include directories, compile definitions and the static libraries it
depends on. See [Using the C++ library](../getting-started/installation.md#using-the-c-library) for both, and
[CMake options](../getting-started/installation.md#cmake-options), particularly `SSQP_WITH_HPIPM` and `SSQP_WITH_PIQP`
if you need the HPIPM or PIQP QP solvers.
