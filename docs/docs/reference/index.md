# Python API Reference

This section is generated directly from the `ssqpy` bindings' docstrings (via
[mkdocstrings](https://mkdocstrings.github.io/)), so it always reflects the module as built. It documents the
Python surface only; see the [C++ API Reference](../cpp-reference/index.md) for the corresponding C++ API, and
[Concepts](../concepts/model.md) for a guided, prose explanation of how the pieces fit together.

* [`ssqpy`](ssqpy/index.md): the top-level module, covering log levels
* [`ssqpy.model`](ssqpy/model/index.md): `Model`, `SolverMode`, `Integration`, `Dimensions`
* [`ssqpy.model.costs`](ssqpy/model/costs.md): `StageCost` and predefined costs
* [`ssqpy.model.constraints`](ssqpy/model/constraints.md): `StageConstraints` and predefined constraints
* [`ssqpy.model.dynamics`](ssqpy/model/dynamics.md): `StageDynamics`
* [`ssqpy.model.terrain`](ssqpy/model/terrain.md): `Terrain`, `FlatTerrain`
* [`ssqpy.solvers`](ssqpy/solvers/index.md): `ssqp`, `MPC`, `ssqpParams`, `admmParams`, `hpipmParams`, `piqpParams`, `FlatVec`,
  convergence data
* [`ssqpy.solvers.search`](ssqpy/solvers/search.md): dynamics and hyperparameter search

!!! note
    Custom subclasses (of `StageCost`, `StageDynamics`, `StageConstraints` or `Terrain`) that you write in Python
    only need to implement the abstract methods described on each base class's page. See the
    [Concepts](../concepts/costs-constraints-dynamics.md) pages for the full contract.
