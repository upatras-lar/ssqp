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
