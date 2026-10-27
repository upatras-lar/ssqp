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

## Formulation

SSQP solves the OCP from the [Home](../index.md) page,

$$
\begin{aligned}
    \min_{\boldsymbol{x}, \boldsymbol{u}} \;& J(\boldsymbol{x}, \boldsymbol{u}) = \sum_{k=0}^{H-1} \ell_k(\boldsymbol{x}_k, \boldsymbol{u}_k) + \ell_H(\boldsymbol{x}_H) \\
    \text{s.t.} \;& \boldsymbol{x}_{k+1} = \boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k), \quad 0 \leq k < H \\
    & \boldsymbol{c}^l_k \leq \boldsymbol{c}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \leq \boldsymbol{c}^u_k, \quad 0 \leq k < H \\
    & \boldsymbol{c}^l_H \leq \boldsymbol{c}_H(\boldsymbol{x}_H) \leq \boldsymbol{c}^u_H
\end{aligned}
$$

where $\boldsymbol{x}_0$ is fixed and $\boldsymbol{c}^l_k$, $\boldsymbol{c}^u_k$ are the
[constraint bounds](costs-constraints-dynamics.md#constraints). Given the current guess
$(\boldsymbol{x}, \boldsymbol{u})$ and multiplier estimates $\boldsymbol{\lambda}$, $\boldsymbol{\mu}$ of the
dynamics and constraints, each iteration solves a QP in the update
$(\Delta\boldsymbol{x}, \Delta\boldsymbol{u})$, where $\boldsymbol{\gamma}_k = \boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \ominus \boldsymbol{x}_{k+1}$
is the current dynamics gap:

$$
\begin{aligned}
    \min_{\Delta\boldsymbol{x}, \Delta\boldsymbol{u}} \;& \sum_{k=0}^{H-1}
    \tfrac{1}{2} \begin{bmatrix} \Delta\boldsymbol{x}_k \\ \Delta\boldsymbol{u}_k \end{bmatrix}^\top
    \boldsymbol{Q}_k
    \begin{bmatrix} \Delta\boldsymbol{x}_k \\ \Delta\boldsymbol{u}_k \end{bmatrix}
    + \begin{bmatrix} \boldsymbol{q}^x_k \\ \boldsymbol{q}^u_k \end{bmatrix}^\top
    \begin{bmatrix} \Delta\boldsymbol{x}_k \\ \Delta\boldsymbol{u}_k \end{bmatrix}
    + \tfrac{1}{2} \Delta\boldsymbol{x}_H^\top \boldsymbol{Q}_H \Delta\boldsymbol{x}_H + \boldsymbol{q}^{x\top}_H \Delta\boldsymbol{x}_H \\
    \text{s.t.} \;& \Delta\boldsymbol{x}_0 = \boldsymbol{0}, \quad
    \Delta\boldsymbol{x}_{k+1} = \boldsymbol{A}_k \Delta\boldsymbol{x}_k + \boldsymbol{B}_k \Delta\boldsymbol{u}_k + \boldsymbol{\gamma}_k, \\
    & \boldsymbol{c}^l_k \ominus \boldsymbol{c}_k \leq \boldsymbol{D}_k \Delta\boldsymbol{x}_k + \boldsymbol{E}_k \Delta\boldsymbol{u}_k \leq \boldsymbol{c}^u_k \ominus \boldsymbol{c}_k
\end{aligned}
$$

with $\boldsymbol{A}_k, \boldsymbol{B}_k$ and $\boldsymbol{D}_k, \boldsymbol{E}_k$ the dynamics and constraint Jacobians,
$\boldsymbol{q}_k$ the cost gradients and $\boldsymbol{Q}_k$ the Hessian of the Lagrangian (the cost Hessian plus the
`hessprod_*` terms of the dynamics and constraints against $\boldsymbol{\lambda}_k$ and $\boldsymbol{\mu}_k$). The
terminal stage has no controls, and its constraints use the Jacobian $\boldsymbol{D}_H$ alone. See
[QP Solvers](qp-solvers.md#qp-formulation) for the full form the QP solvers receive.

The QP also returns new multipliers $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$. After the line search picks a step
size $\alpha \in (0, 1]$ (below), the guess is updated through the manifold operator
$\oplus$ (plain addition for Euclidean states):

$$
\boldsymbol{x}_k \leftarrow \boldsymbol{x}_k \oplus \alpha \Delta\boldsymbol{x}_k, \qquad
\boldsymbol{u}_k \leftarrow \boldsymbol{u}_k + \alpha \Delta\boldsymbol{u}_k
$$

## Termination

* `max_iter`: hard cap on SSQP iterations.
* `tolerance` (and optionally `tolerance_final` / `tolerance_update_steps`, to anneal it linearly over the first N
  steps): convergence tolerance on the Lagrangian's Jacobian norms w.r.t. $\boldsymbol{x}$ and $\boldsymbol{u}$.
* `constraint_tolerance`: separate satisfaction tolerance for constraints, if set positive.
* `stop_after_constraint_satisfaction`: stop as soon as constraints are satisfied, regardless of Lagrangian norms.
* `loose_success` (with `loose_success_tolerance`, `loose_success_stale_iters`): an early-exit heuristic that stops if
  the cost Jacobian norm hasn't improved by more than `loose_success_tolerance` for `loose_success_stale_iters`
  consecutive iterations.
* `stop_on_qp_infeasibility` (default on): stop with `PrimalInfeasible` as soon as the QP solver reports an infeasible
  QP, returning the current guess. When off, ADMM skips the infeasibility check and runs as it would without it. Only
  ADMM detects infeasibility.

The result reports a `SolverState` (`Success`, `LooseSuccess`, `ConstraintSuccess`, `PrimalInfeasible`,
`MaxIterReached`, `Ongoing`, …)
and detailed [`ssqpConvergenceData`](../reference/solvers.md#ssqpy.solvers.ssqpConvergenceData), covering per-iteration
timing, step size, norms, cost and violation, retrievable via `solver.getConvergenceData()`.

## Line search

The line search measures a trial point $(\boldsymbol{x}, \boldsymbol{u})$ with three quantities: the total cost
$J(\boldsymbol{x}, \boldsymbol{u})$ from above, the dynamics gap

$$
\Gamma(\boldsymbol{x}, \boldsymbol{u}) = \sum_{k=0}^{H-1} \left\lVert \boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \ominus \boldsymbol{x}_{k+1} \right\rVert_\infty
$$

and the constraint violation

$$
C(\boldsymbol{x}, \boldsymbol{u}) = \sum_{k=0}^{H} \left\lVert \max\left(\boldsymbol{c}^l_k \ominus \boldsymbol{c}_k,\; \boldsymbol{c}_k \ominus \boldsymbol{c}^u_k,\; \boldsymbol{0}\right) \right\rVert_\infty
$$

Starting at $\alpha = 1$, SSQP evaluates the trial point $\boldsymbol{x}^+ = \boldsymbol{x} \oplus \alpha \Delta\boldsymbol{x}$,
$\boldsymbol{u}^+ = \boldsymbol{u} + \alpha \Delta\boldsymbol{u}$. While the trial point is rejected, it divides $\alpha$ by
`inv_alpha_scale_factor` and tries again, until $\alpha$ drops below `min_alpha`. The acceptance test depends on the
method (`line_search_method`):

* **`Filter`** *(default)*: a filter line search. Let $\mathcal{F}$ be the last `filter_size` accepted
    $(J, \Gamma, C)$ triples, and $(J_0, \Gamma_0, C_0)$ those of the current guess. The trial point is rejected if
    it is dominated by any filter entry, i.e. if for some $(J_i, \Gamma_i, C_i) \in \mathcal{F}$

    $$
    J^+ > J_i \;\wedge\; \Gamma^+ > \Gamma_i \;\wedge\; C^+ > C_i
    $$

    and it is otherwise accepted only if it improves on at least one measure of the current guess:

    $$
    J^+ \leq J_0 \;\vee\; \Gamma^+ \leq \Gamma_0 \;\vee\; C^+ \leq C_0
    $$

    The accepted triple is then pushed to the filter. No simultaneous improvement on all three measures is required.
* **`Merit`**: a merit-function line search. The merit function weights the three measures with
    $\mu_\Gamma$ (`merit_mu_gamma`) and $\mu_C$ (`merit_mu_c`),

    $$
    M(\boldsymbol{x}, \boldsymbol{u}) = J + \mu_\Gamma \Gamma + \mu_C C
    $$

    and the trial point is accepted if it satisfies the Armijo sufficient-decrease condition, with
    $\beta$ = `merit_armijo_beta`,

    $$
    M(\boldsymbol{x}^+, \boldsymbol{u}^+) < M(\boldsymbol{x}, \boldsymbol{u}) + \beta \, \alpha \, \nabla M(\boldsymbol{x}, \boldsymbol{u})^\top
    \begin{bmatrix} \Delta\boldsymbol{x} \\ \Delta\boldsymbol{u} \end{bmatrix}
    $$

    where $\nabla M$ uses a smooth (softmax) approximation of the $\infty$-norms in $\Gamma$ and $C$.

If the line search fails to find an acceptable step even at `min_alpha`, SSQP by default increases QP
regularization and retries; set `accept_worse_update` to accept the `min_alpha` step anyway, or
`skip_regularization` to raise an exception immediately instead of regularizing.

## Choosing a QP solver

`qp_solver_type` selects between the three [QP solvers](qp-solvers.md) (`ADMM`, `HPIPM` or `PIQP`);
`admm_params`/`hpipm_*`/`piqp_*` keyword arguments (prefixed `admm_`/`hpipm_`/`piqp_` when passed to the `ssqp`
constructor) configure the selected one.

## Saving and loading trajectories

`solver.write(filepath)` writes the current solution to a JSON file; `ssqpy.solvers.loadTrajectory(filepath)` reads
one back as a pair of `FlatVec` state/control trajectories, useful for warm-starting later runs or for the
[visualizer example](../examples.md).
