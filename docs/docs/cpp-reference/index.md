# C++ API Reference

SSQP is written in C++17; the `ssqpy` Python module documented in the [Python API Reference](../reference/index.md)
is a near 1:1 [nanobind](https://github.com/wjakob/nanobind) wrapping of the library described here. The library can
be used directly from C++ without the Python bindings at all: link against the `ssqp` CMake target (see
[Installation](../getting-started/installation.md)) and include the relevant headers under
[`include/`](https://github.com/upatras-lar/ssqp/tree/main/include).

This reference is generated from the headers' own Doxygen-style (`/** */`) comments at doc build time: the pages
below embed each class's live public interface, so they can't drift out of sync with the source. It does not
enumerate every predefined cost/constraint constructor argument, since those mirror the
[Python bindings](../concepts/costs-constraints-dynamics.md) closely; consult the corresponding predefined header
or the [Python reference](../reference/model-costs.md) for exact parameters.

## Pages

* [Model](model.md): `ssqp::model::Model`, the problem definition
* [Stagewise Functions](stagewise.md): `Function`, `StageCost`, `StageDynamics`, `StageConstraints`, and the
  predefined implementations
* [Terrain](terrain.md): `Terrain`, `FlatTerrain`
* [The SSQP Solver](solver.md): `ssqp::solvers::ssqp`
* [QP Solvers](qp-solvers.md): `QPSolver`, `admm_qp`, `hpipm_qp`, `lqr`
* [MPC](mpc.md): `ssqp::solvers::MPC`
* [Parameter Search](parameter-search.md): `ssqp::param_search`

## Namespaces

All library code lives under `ssqp::`, mirroring the Python package layout:

* `ssqp::model`: `Model`, `Container`, `Dimensions`
* `ssqp::model::stagewise`: `StageCost`, `StageDynamics`, `StageConstraints`, `Function`, and their predefined
  implementations (in `model/stagewise/predefined/`)
* `ssqp::model::terrain`: `Terrain` and predefined terrains (in `model/terrain/predefined/`)
* `ssqp::model::presets`: URDFs baked into the binary at build time, looked up by `Model::Structure::from_preset`
* `ssqp::solvers`: `ssqp`, `MPC`, `QPSolver`, `admm_qp`, `hpipm_qp`, `lqr`
* `ssqp::step`: the filter and merit line search implementations
* `ssqp::param_search`: dynamics parameter and hyperparameter search

## Header map

| Namespace / class | Header(s) |
| --- | --- |
| `ssqp::model::Model` | `include/model/Model.hpp` |
| `ssqp::model::Container` | `include/model/Container.hpp` |
| `ssqp::model::stagewise::Function` / `StageCost` / `StageDynamics` / `StageConstraints` | `include/model/stagewise/Function.hpp`, `StageCost.hpp`, `StageDynamics.hpp`, `StageConstraints.hpp` |
| Predefined costs / constraints | `include/model/stagewise/predefined/Costs.hpp`, `Constraints.hpp` |
| `ssqp::model::terrain::Terrain` / `FlatTerrain` | `include/model/terrain/Terrain.hpp`, `include/model/terrain/predefined/FlatTerrain.hpp` |
| `ssqp::model::presets` | `include/model/presets/Presets.hpp` |
| `ssqp::solvers::ssqp` | `include/solvers/ssqp.hpp` |
| Filter / merit line search | `include/sqp_step/filter.hpp`, `include/sqp_step/merit.hpp` |
| `ssqp::solvers::QPSolver` / `admm_qp` / `hpipm_qp` | `include/solvers/qp.hpp`, `admm_qp.hpp`, `hpipm_qp.hpp` |
| `ssqp::solvers::lqr` | `include/solvers/lqr.hpp` |
| `ssqp::solvers::MPC` | `include/solvers/mpc.hpp` |
| `ssqp::param_search` | `include/param_search/search.hpp` |
| Core types (`VecX`, `MatX`, `FlatVec`, log levels, …) | `include/types.hpp` |

## Building against the C++ library

`ssqp` is built as a static library by CMake (target `ssqp`); linking against it from another CMake project means
adding this repository (e.g. via `add_subdirectory` or `FetchContent`) and linking the `ssqp` target, which
propagates its public include directories. See [Installation](../getting-started/installation.md) for the required
dependencies and CMake options, particularly `WITH_HPIPM` if you need the HPIPM QP solver, and
`BUILD_PYTHON=OFF` if you only need the C++ library.
