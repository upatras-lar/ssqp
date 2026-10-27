# Python API Reference

This section is generated directly from the `ssqpy` bindings' docstrings (via
[mkdocstrings](https://mkdocstrings.github.io/)), so it always reflects the module as built. It documents the
Python surface only; see the [C++ API Reference](../cpp-reference/index.md) for the corresponding C++ API, and
[Concepts](../concepts/model.md) for a guided, prose explanation of how the pieces fit together.

* [`ssqpy`](ssqpy.md): the top-level module, covering log levels
* [`ssqpy.model`](model.md): `Model`, `SolverMode`, `Integration`, `Dimensions`
* [`ssqpy.model.costs`](model-costs.md): `StageCost` and predefined costs
* [`ssqpy.model.constraints`](model-constraints.md): `StageConstraints` and predefined constraints
* [`ssqpy.model.dynamics`](model-dynamics.md): `StageDynamics`
* [`ssqpy.model.terrain`](model-terrain.md): `Terrain`, `FlatTerrain`
* [`ssqpy.solvers`](solvers.md): `ssqp`, `MPC`, `ssqpParams`, `admmParams`, `hpipmParams`, `piqpParams`, `FlatVec`,
  convergence data
* [`ssqpy.solvers.search`](solvers-search.md): dynamics and hyperparameter search

!!! note
    Custom subclasses (of `StageCost`, `StageDynamics`, `StageConstraints` or `Terrain`) that you write in Python
    only need to implement the abstract methods described on each base class's page. See the
    [Concepts](../concepts/costs-constraints-dynamics.md) pages for the full contract.
