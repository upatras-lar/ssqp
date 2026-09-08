# Quickstart

This walks through the smallest complete SSQP problem: driving a 1D double integrator to a target position, with
custom (non-robot) dynamics and a custom cost. It corresponds to
[`examples/trajgen/python/di.py`](https://github.com/upatras-lar/ssqp/blob/main/examples/trajgen/python/di.py); the
code blocks below are pulled live from that file at doc build time.

Make sure `ssqpy` is importable first; see [Installation](installation.md).

## 1. Define the dynamics

Custom dynamics subclass `ssqpy.model.dynamics.StageDynamics` and must provide, at minimum, `output_size`,
`state_size`, `control_size`, `evaluate` and `gradient` (Hessian-vector products are optional, defaulting to zero,
and only needed if you want exact second-order information):

```python title="examples/trajgen/python/di.py"
--8<-- "examples/trajgen/python/di.py:8:43"
```

The state is $[p, v]$ (position, velocity), the control is a scalar acceleration $u$, and `evaluate` implements
semi-implicit Euler integration with $dt = 0.1$.

## 2. Define the cost

Custom costs subclass `ssqpy.model.costs.StageCost` and must provide `evaluate`, `gradient`, `hessian_xx`,
`hessian_xu` and `hessian_uu`:

```python title="examples/trajgen/python/di.py"
--8<-- "examples/trajgen/python/di.py:46:68"
```

This penalizes deviation of the position $p$ from a target of $1.0$, applied both at every running stage and at
the terminal stage.

## 3. Build the model and solve

```python title="examples/trajgen/python/di.py"
--8<-- "examples/trajgen/python/di.py:71:92"
```

`model.finalize` accepts a single `dynamics` shared across every stage (as above), or a per-stage list; the same
applies to costs and constraints. See [Model](../concepts/model.md) for the full set of `finalize` arguments (XML
loading, joint limits, initial guesses, etc.).

`solver.solve()` returns the solver's [`SolverState`](../reference/solvers.md), and the resulting state and control
trajectories as [`FlatVec`](../reference/solvers.md) objects, flattened, stagewise vectors that can be indexed with
`.stage(k)` or converted with `.toarray()`.

## Next steps

* [Model](../concepts/model.md): horizons, XML-based robot models, solver modes and integration schemes
* [Costs, Constraints & Dynamics](../concepts/costs-constraints-dynamics.md): the stagewise function interface, and
  predefined costs/constraints that ship with SSQP
* [The SSQP Solver](../concepts/solver.md): solver parameters, line search methods and QP solver selection
* [MPC](../concepts/mpc.md): using SSQP in a receding-horizon control loop
* [Examples](../examples.md): full trajectory generation and MPC examples for real robot models
