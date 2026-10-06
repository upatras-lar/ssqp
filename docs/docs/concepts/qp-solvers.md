# QP Solvers

At each SSQP iteration, quadratizing the cost and linearizing the dynamics/constraints produces a Quadratic Program
with the same stagewise (banded) structure as the original OCP. `QPSolver` (`ssqp::solvers::QPSolver`) is the
abstract interface both QP backends implement:

```cpp
virtual void        reset()                  = 0;
virtual void        increaseRegularization() = 0;
virtual SolverState solve(Solution &sol)     = 0;
virtual uint        iterations() const       = 0;
```

Select the backend via `ssqpParams.qp_solver_type` (`QPSolverType.ADMM` or `QPSolverType.HPIPM`); see
[The SSQP Solver](solver.md#choosing-a-qp-solver).

## QP formulation

At each SSQP iterate $(\boldsymbol{x}, \boldsymbol{u})$, with multiplier estimates $\boldsymbol{\lambda}$ (dynamics)
and $\boldsymbol{\mu}$ (constraints), the QP solvers find the update $(\Delta\boldsymbol{x}, \Delta\boldsymbol{u})$ that
solves

$$
\begin{aligned}
    \min_{\Delta\boldsymbol{x}, \Delta\boldsymbol{u}} \;& \sum_{k=0}^{H-1}
    \left(
    \tfrac{1}{2} \Delta\boldsymbol{x}_k^\top \boldsymbol{Q}^{xx}_k \Delta\boldsymbol{x}_k
    + \Delta\boldsymbol{x}_k^\top \boldsymbol{Q}^{xu}_k \Delta\boldsymbol{u}_k
    + \tfrac{1}{2} \Delta\boldsymbol{u}_k^\top \boldsymbol{Q}^{uu}_k \Delta\boldsymbol{u}_k
    + \boldsymbol{q}^{x\top}_k \Delta\boldsymbol{x}_k + \boldsymbol{q}^{u\top}_k \Delta\boldsymbol{u}_k
    \right) \\
    & + \tfrac{1}{2} \Delta\boldsymbol{x}_H^\top \boldsymbol{Q}^{xx}_H \Delta\boldsymbol{x}_H + \boldsymbol{q}^{x\top}_H \Delta\boldsymbol{x}_H \\
    \text{s.t.} \;& \Delta\boldsymbol{x}_0 = \boldsymbol{0} \\
    & \Delta\boldsymbol{x}_{k+1} = \boldsymbol{A}_k \Delta\boldsymbol{x}_k + \boldsymbol{B}_k \Delta\boldsymbol{u}_k + \boldsymbol{\gamma}_k, \quad 0 \leq k < H \\
    & \boldsymbol{c}^l_k \leq \boldsymbol{D}_k \Delta\boldsymbol{x}_k + \boldsymbol{E}_k \Delta\boldsymbol{u}_k \leq \boldsymbol{c}^u_k, \quad 0 \leq k < H \\
    & \boldsymbol{c}^l_H \leq \boldsymbol{D}_H \Delta\boldsymbol{x}_H \leq \boldsymbol{c}^u_H
\end{aligned}
$$

The data comes from the stagewise functions of the [model](costs-constraints-dynamics.md), all evaluated at the
current iterate:

| Symbol | Meaning |
| --- | --- |
| $\boldsymbol{q}^x_k$, $\boldsymbol{q}^u_k$ | Cost gradients, $\nabla_x \ell_k$ and $\nabla_u \ell_k$ |
| $\boldsymbol{Q}^{xx}_k$, $\boldsymbol{Q}^{xu}_k$, $\boldsymbol{Q}^{uu}_k$ | Hessian of the Lagrangian: the cost Hessian blocks plus the `hessprod_*` terms of the dynamics (against $\boldsymbol{\lambda}_k$) and the constraints (against $\boldsymbol{\mu}_k$) |
| $\boldsymbol{A}_k$, $\boldsymbol{B}_k$ | Dynamics Jacobians with respect to $\boldsymbol{x}_k$ and $\boldsymbol{u}_k$ |
| $\boldsymbol{\gamma}_k$ | Dynamics gap, $\boldsymbol{f}_k(\boldsymbol{x}_k, \boldsymbol{u}_k) \ominus \boldsymbol{x}_{k+1}$ |
| $\boldsymbol{D}_k$, $\boldsymbol{E}_k$ | Constraint Jacobians with respect to $\boldsymbol{x}_k$ and $\boldsymbol{u}_k$ |
| $\boldsymbol{c}^l_k$, $\boldsymbol{c}^u_k$ | Remaining room to the constraint bounds, here $\boldsymbol{c}^l_k \ominus \boldsymbol{c}_k(\boldsymbol{x}_k, \boldsymbol{u}_k)$ and $\boldsymbol{c}^u_k \ominus \boldsymbol{c}_k(\boldsymbol{x}_k, \boldsymbol{u}_k)$ |

The terminal stage has no controls. Stages without constraints have empty $\boldsymbol{D}_k$, $\boldsymbol{E}_k$,
and equality constraints have equal $\boldsymbol{c}^l_k$ and $\boldsymbol{c}^u_k$. Since the initial state is fixed,
$\Delta\boldsymbol{x}_0$ is zero. For a manifold-valued state, all of these quantities live in the tangent space, so
$\Delta\boldsymbol{x}_k \in \mathcal{R}^{n_{dx}}$ (see
[Manifold-valued states](costs-constraints-dynamics.md#manifold-valued-states)). Besides $\Delta\boldsymbol{x}$ and
$\Delta\boldsymbol{u}$, the solver returns the multipliers $\boldsymbol{\lambda}$ and $\boldsymbol{\mu}$ of the QP's
dynamics and constraints, which become the next iterate's estimates.

## ADMM

`admm_qp` is SSQP's own Alternating Direction Method of Multipliers implementation. It requires no extra
dependencies (unlike HPIPM, see below) and is the default. Key parameters (`admmParams`, or `admm_`-prefixed
keyword arguments):

* `init_rho`, `alpha` (over-relaxation), `sigma` (proximal regularization)
* Adaptive $\rho$ update: `rho_update`, `rho_min`/`rho_max`, `adaptive_rho_tolerance`, `rho_update_iter_freq`
* Termination: `max_iter`, `abs_tolerance`/`abs_tolerance_final`, `rel_tolerance`/`rel_tolerance_final`
  (`tolerance_update_steps` anneals both linearly, as with the outer SSQP tolerances), `primal_infs_tolerance`
* `warm_start`: warm-start with the unconstrained solution (disable when using
  [`InverseDynamics`](model.md#solver-mode) mode)
* `reset_rho` / `reset_y`: whether to reset ADMM's $\rho$ and $y$ between successive SSQP formulations
  (disabling either can cause oscillations)

## HPIPM

`hpipm_qp` wraps [HPIPM](https://github.com/giaf/hpipm) (via [BLASFEO](https://github.com/giaf/blasfeo)), an
interior-point QP solver. It requires building SSQP with `-DWITH_HPIPM=ON` (see
[Installation](../getting-started/installation.md#cmake-options)). Key parameters (`hpipmParams`, or
`hpipm_`-prefixed keyword arguments):

* `mode` (`HPIPMMode`: `Robust`, `Speed`, `Balance`): trades off robustness against speed
* `max_iter`, `alpha_min`, `mu0` (initial barrier parameter)
* `tol_stat`, `tol_eq`, `tol_ineq`, `tol_comp`: stationarity/equality/inequality/complementarity tolerances
* `reg_prim`: primal regularization
* `warm_start`, `prediction_correction`, `square_root_riccati`, `split_step`

## LQR

`ssqp::solvers::lqr` is a Riccati-recursion based solver used internally to warm-start/solve the unconstrained
(equality-only) structure of the QP efficiently, exposing both a structural solve (`output_structure`, producing
gain matrices $P, K$) and a full `unconstrained` solve. It is not exposed as a standalone `QPSolverType`; instead it
backs the ADMM solver's per-iteration linear solves.

See the [API reference](../reference/solvers.md) for `ssqpParams`, `admmParams` and `hpipmParams` in full.
