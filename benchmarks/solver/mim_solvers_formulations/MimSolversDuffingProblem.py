"""
MimSolversDuffingProblem: a mim_solvers (CSQP) formulation of the scalable
coupled Duffing oscillator benchmark (see oscillators.py), built directly
on crocoddyl's crocoddyl.ActionModelAbstract extension point rather than
MimSolversProblem's Pinocchio/crocoddyl-multibody machinery

A plain crocoddyl.ActionModelAbstract represents one full discrete
transition directly (state -> next state, plus its cost), unlike the
DifferentialActionModel + IntegratedActionModelEuler split
MimSolversProblem uses for its robot problems, since crocoddyl's own
IntegratedActionModelEuler is semi-implicit (v_next = v + dt*a, q_next = q
+ dt*v_next), which would not match oscillators.py's plain forward-Euler
convention (q_next = q + dt*v, using the *current* v)
Implementing the discrete map directly avoids that mismatch, and also means
none of MimSolversProblem's dt-rescaling of costs (needed only to cancel
IntegratedActionModelEuler's own automatic dt scaling of running-node
costs) applies here, since stage_cost()/terminal_cost() are used as-is
"""

import json
import time

import crocoddyl
import mim_solvers
import numpy as np

from common.oscillators import CoupledDuffingOCP, DuffingConfig


class _DuffingActionModel(crocoddyl.ActionModelAbstract):
    """One full discrete Duffing transition x -> x + dt*f(x,u), plus its
    stage cost, or, when called without u (crocoddyl's terminal-node
    convention), just the terminal cost with x held fixed"""

    def __init__(self, state, problem: CoupledDuffingOCP):
        super().__init__(state, problem.nu)
        self.problem = problem

    def calc(self, data, x, u=None):
        if u is None:
            data.xnext[:] = x
            data.cost = self.problem.terminal_cost(x)
        else:
            data.xnext[:] = self.problem.dynamics(x, u)
            data.cost = self.problem.stage_cost(x, u)

    def calcDiff(self, data, x, u=None):
        nx = self.problem.nx
        if u is None:
            _, g, H = self.problem.terminal_cost_derivatives(x)
            data.Lx[:] = g
            data.Lxx[:, :] = H
        else:
            A, B = self.problem.dynamics_jacobians(x, u)
            data.Fx[:, :] = A
            data.Fu[:, :] = B
            _, g, H = self.problem.stage_cost_derivatives(x, u)
            data.Lx[:] = g[:nx]
            data.Lu[:] = g[nx:]
            data.Lxx[:, :] = H[:nx, :nx]
            data.Lxu[:, :] = H[:nx, nx:]
            data.Luu[:, :] = H[nx:, nx:]


class MimSolversDuffingProblem:
    def __init__(
        self, horizon, dt, duffing_config: DuffingConfig, parameters=None
    ):
        self.H = horizon
        self.dt = dt
        self.problem = CoupledDuffingOCP(duffing_config)

        self.state = crocoddyl.StateVector(self.problem.nx)
        self.nu = self.problem.nu

        self.x0 = None
        self.shooting_problem = None
        self.solver = None
        self.solve_time = None
        self.converged = None
        self.iterations = None
        self.objective = None
        # No path constraints in this problem (matches
        # AligatorDuffingProblem), so there's nothing to report here --
        # the dynamics-transition defect is tracked separately below, as
        # stage_gap
        self.constraint_violation = 0.0
        self.stage_gap = None

    def finalize(self, parameter_dict=None):
        self.x0 = self.problem.initial_state()

        # Every running node shares the same dynamics/cost, so each gets
        # its own model instance (crocoddyl's own convention -- see
        # MimSolversProblem's stage_costs/stage_contacts lists) but all
        # of them are otherwise identical
        running_models = [
            _DuffingActionModel(self.state, self.problem)
            for _ in range(self.H)
        ]
        terminal_model = _DuffingActionModel(self.state, self.problem)

        self.shooting_problem = crocoddyl.ShootingProblem(
            self.x0, running_models, terminal_model
        )

    def solve(self, parameters=None):
        parameters = parameters or {}
        if self.shooting_problem is None:
            self.finalize()

        self.solver = mim_solvers.SolverCSQP(self.shooting_problem)
        self.solver.termination_tolerance = parameters.get("tolerance", 1e-2)
        self.solver.max_qp_iters = parameters.get("max_qp_iters", 100)
        self.solver.eps_abs = parameters.get("eps_abs", 1e-3)
        self.solver.eps_rel = parameters.get("eps_rel", 1e-3)
        self.solver.adaptive_rho_tolerance = parameters.get(
            "adaptive_rho_tolerance", 1
        )
        self.solver.rho_update_interval = parameters.get(
            "rho_update_interval", 25
        )
        self.solver.use_filter_line_search = parameters.get(
            "use_filter_line_search", True
        )
        # Left at SolverCSQP()'s own default unless explicitly passed
        if "reset_rho" in parameters:
            self.solver.reset_rho = parameters["reset_rho"]
        if "reset_y" in parameters:
            self.solver.reset_y = parameters["reset_y"]
        if "update_rho_with_heuristic" in parameters:
            self.solver.update_rho_with_heuristic = parameters[
                "update_rho_with_heuristic"
            ]

        if parameters.get("callbacks", False):
            self.solver.setCallbacks(
                [
                    mim_solvers.CallbackVerbose(),
                    mim_solvers.CallbackLogger(),
                ]
            )

        xs_init = parameters.get("xs_init") or [
            self.x0.copy() for _ in range(self.H + 1)
        ]
        us_init = parameters.get("us_init") or [
            np.zeros(self.nu) for _ in range(self.H)
        ]

        start = time.perf_counter()
        success = self.solver.solve(
            xs_init, us_init, parameters.get("max_iter", 1000)
        )
        self.solve_time = time.perf_counter() - start

        self.converged = bool(success)
        self.iterations = int(self.solver.iter)
        self.objective = float(self.solver.cost)
        self.stage_gap = self._evaluate_stage_gap()

        print(
            f"[mim_solvers] Solve time: {self.solve_time:.3f} s, "
            f"converged: {self.converged}"
        )
        return self.converged

    def _evaluate_stage_gap(self):
        """Sum, over stages, of the infinity norm of how far the reported
        trajectory's x_{k+1} sits from problem.dynamics(x_k, u_k) --
        CSQP's multi-shooting formulation tracks this defect internally
        during optimization and only drives it toward zero as tightly as
        the outer tolerance demands, so it's generally nonzero at
        convergence, same as SSQP's own trajectory_stage_gap"""
        xs = self.solver.xs
        us = self.solver.us
        total = 0.0
        for k in range(self.H):
            defect = np.asarray(xs[k + 1]) - self.problem.dynamics(
                np.asarray(xs[k]), np.asarray(us[k])
            )
            total += float(np.max(np.abs(defect)))
        return total

    def save(self, output_json):
        if self.solver is None:
            raise RuntimeError("save() called before solve()")

        # plot_benchmarks.py (the only consumer of this problem's results
        # so far) only reads the "benchmark" summary below; there's no
        # visualizer for this problem yet, so the full per-stage
        # trajectory schema the robot problems write isn't needed
        output = {
            "benchmark": {
                "format_version": 1,
                "solver": "mim_solvers",
                "success": self.converged,
                "iterations": self.iterations,
                "runtime_ms": self.solve_time * 1000.0,
                "objective": self.objective,
                "constraint_violation": self.constraint_violation,
                "stage_gap": self.stage_gap,
            }
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[mim_solvers] Saved solution to {output_json}")
