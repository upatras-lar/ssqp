# The SSQP Solver

`ssqpy.solvers.ssqp` (`ssqp::solvers::ssqp` in C++) drives the outer SQP loop described on the [Home](../index.md)
page: starting from an initial guess for $\boldsymbol{x}$ and $\boldsymbol{u}$ (provided via
`Model.finalize(init_states=..., init_controls=...)`, or zero), it repeats, until convergence:

1. **Quadratize the cost and linearize the dynamics/constraints** at the current guess (cost gradients/Hessians,
   dynamics/constraint Jacobians) to build a Quadratic Program (QP) that exploits the stagewise (banded) structure
   of the OCP.
2. **Solve the QP** with one of the [QP solvers](qp-solvers.md) to get an update direction.
3. **Line search** to pick a step size along that direction (see below).
4. **Check termination criteria** on the updated guess; stop, or go back to 1.

```python
solver = ssqpy.solvers.ssqp(model)  # or ssqpy.solvers.ssqp(model, **kwargs)
state, x, u = solver.solve()
```

`params`/keyword arguments accepted by the constructor are defined by `ssqpParams`
(see the [API reference](../reference/solvers.md#ssqpy.solvers.ssqpParams) for the exhaustive, current list); the
most commonly tuned are summarized below.

## Termination

* `max_iter`: hard cap on SSQP iterations.
* `tolerance` (and optionally `tolerance_final` / `tolerance_update_steps`, to anneal it linearly over the first N
  steps): convergence tolerance on the Lagrangian's Jacobian norms w.r.t. $\boldsymbol{x}$ and $\boldsymbol{u}$.
* `constraint_tolerance`: separate satisfaction tolerance for constraints, if set positive.
* `stop_after_constraint_satisfaction`: stop as soon as constraints are satisfied, regardless of Lagrangian norms.
* `loose_success` (with `loose_success_tolerance`, `loose_success_stale_iters`): an early-exit heuristic that stops if
  the cost Jacobian norm hasn't improved by more than `loose_success_tolerance` for `loose_success_stale_iters`
  consecutive iterations.

The result reports a `SolverState` (`Success`, `LooseSuccess`, `ConstraintSuccess`, `MaxIterReached`, `Ongoing`, …)
and detailed [`ssqpConvergenceData`](../reference/solvers.md#ssqpy.solvers.ssqpConvergenceData), covering per-iteration
timing, step size, norms, cost and violation, retrievable via `solver.getConvergenceData()`.

## Line search

After solving the QP, SSQP scales the step by $\alpha$ (starting at 1, divided by `inv_alpha_scale_factor` on each
rejected trial, down to `min_alpha`) using one of two methods (`line_search_method`):

* **`Filter`** *(default)*: a filter line search. A step is accepted if it improves on at least one of three
  criteria (Lagrangian norm, stage gap, constraint violation) relative to the recent history (`filter_size` entries
  kept), rather than requiring simultaneous improvement on all of them.
* **`Merit`**: a merit-function line search. Combines norm, stage gap (weighted by `merit_mu_gamma`) and
  constraint violation (weighted by `merit_mu_c`) into a single scalar merit function and applies an Armijo
  sufficient-decrease condition (`merit_armijo_beta`) on it.

If the line search fails to find an acceptable step even at `min_alpha`, SSQP by default increases QP
regularization and retries; set `accept_worse_update` to accept the `min_alpha` step anyway, or
`skip_regularization` to raise an exception immediately instead of regularizing.

## Choosing a QP solver

`qp_solver_type` selects between the two [QP solvers](qp-solvers.md) (`ADMM` or `HPIPM`); `admm_params`/`hpipm_*`
keyword arguments (prefixed `admm_`/`hpipm_` when passed to the `ssqp` constructor) configure the selected one.

## Saving and loading trajectories

`solver.write(filepath)` writes the current solution to a JSON file; `ssqpy.solvers.loadTrajectory(filepath)` reads
one back as a pair of `FlatVec` state/control trajectories, useful for warm-starting later runs or for the
[visualizer example](../examples.md).
