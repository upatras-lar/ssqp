"""
SSQPDuffingProblem: an SSQP formulation of the scalable coupled Duffing
oscillator benchmark (see oscillators.py), built directly on ssqpy's
custom-dynamics API (ssqpy.model.dynamics.StageDynamics /
ssqpy.model.costs.StageCost)

Unlike every other benchmark problem, this one has no robot/URDF
underneath it: it's an abstract chain of coupled oscillators with
closed-form dynamics/cost derivatives
It's built as its own small wrapper, reusing ProblemWrapper's
constructor/solve/save shape so run_benchmark.py's dispatch stays
uniform, without inheriting its abstract robot-specific methods
"""

import json

import numpy as np
import ssqpy
import ssqpy.model.costs as costs
import ssqpy.model.dynamics as dynamics

from common.oscillators import CoupledDuffingOCP, DuffingConfig


class _DuffingDynamics(dynamics.StageDynamics):
    def __init__(self, problem: CoupledDuffingOCP):
        super().__init__()
        self.problem = problem

    def output_size(self):
        return self.problem.nx

    def state_size(self):
        return self.problem.nx

    def control_size(self):
        return self.problem.nu

    def evaluate(self, x, u):
        return self.problem.dynamics(x, u)

    def gradient(self, x, u):
        return self.problem.dynamics_jacobians(x, u)

    def hessprod_xx(self, x, u, lam):
        nx = self.problem.nx
        H = self.problem.weighted_dynamics_hessian(x, u, lam)
        return H[:nx, :nx]

    def hessprod_xu(self, x, u, lam):
        nx = self.problem.nx
        H = self.problem.weighted_dynamics_hessian(x, u, lam)
        return H[:nx, nx:]

    def hessprod_uu(self, x, u, lam):
        nx = self.problem.nx
        H = self.problem.weighted_dynamics_hessian(x, u, lam)
        return H[nx:, nx:]


class _DuffingStageCost(costs.StageCost):
    def __init__(self, problem: CoupledDuffingOCP):
        super().__init__()
        self.problem = problem

    def evaluate(self, x, u):
        return self.problem.stage_cost(x, u)

    def gradient(self, x, u):
        nx = self.problem.nx
        _, g, _ = self.problem.stage_cost_derivatives(x, u)
        return g[:nx], g[nx:]

    def hessian_xx(self, x, u):
        nx = self.problem.nx
        _, _, H = self.problem.stage_cost_derivatives(x, u)
        return H[:nx, :nx]

    def hessian_xu(self, x, u):
        nx = self.problem.nx
        _, _, H = self.problem.stage_cost_derivatives(x, u)
        return H[:nx, nx:]

    def hessian_uu(self, x, u):
        nx = self.problem.nx
        _, _, H = self.problem.stage_cost_derivatives(x, u)
        return H[nx:, nx:]


class _DuffingTerminalCost(costs.StageCost):
    """
    Same StageCost interface as _DuffingStageCost, but the terminal
    cost has no u-dependence
    """

    def __init__(self, problem: CoupledDuffingOCP):
        super().__init__()
        self.problem = problem

    def evaluate(self, x, u):
        return self.problem.terminal_cost(x)

    def gradient(self, x, u):
        _, g, _ = self.problem.terminal_cost_derivatives(x)
        return g, np.zeros(u.shape[0])

    def hessian_xx(self, x, u):
        _, _, H = self.problem.terminal_cost_derivatives(x)
        return H

    def hessian_xu(self, x, u):
        nx = self.problem.nx
        return np.zeros((nx, u.shape[0]))

    def hessian_uu(self, x, u):
        return np.zeros((u.shape[0], u.shape[0]))


class SSQPDuffingProblem:
    def __init__(
        self, horizon, dt, duffing_config: DuffingConfig, parameters=None
    ):
        self.H = horizon
        self.dt = dt
        self.problem = CoupledDuffingOCP(duffing_config)
        self.model = ssqpy.model.Model(horizon, dt)

    def finalize(self, parameter_dict=None):
        parameter_dict = parameter_dict or {}

        x0 = self.problem.initial_state()

        self.model.finalize(
            dynamics=[
                _DuffingDynamics(self.problem) for _ in range(self.H + 1)
            ],
            stage_cost=_DuffingStageCost(self.problem),
            terminal_cost=_DuffingTerminalCost(self.problem),
            init_states=x0,
            **parameter_dict,
        )

    def solve(self, parameters):
        if not self.model.finalized:
            self.finalize()

        if not parameters.get("verbose", False):
            ssqpy.setSilentMode()

        ssqp_params = ssqpy.solvers.ssqpParams()

        ssqp_params.max_iter = parameters.get("max_iter", 1000)
        ssqp_params.tolerance = parameters.get("tolerance", 1e-2)

        if "qp_solver" in parameters and parameters["qp_solver"] == "hpipm":
            ssqp_params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
            ssqp_params.hpipmParams.tol_comp = parameters.get(
                "hpipm_tol", 1e-3
            )
            ssqp_params.hpipmParams.tol_stat = parameters.get(
                "hpipm_tol", 1e-3
            )
            ssqp_params.hpipmParams.tol_eq = parameters.get("hpipm_tol", 1e-3)
            ssqp_params.hpipmParams.tol_ineq = parameters.get(
                "hpipm_tol", 1e-3
            )
            if "hpipm_split_step" in parameters:
                ssqp_params.hpipmParams.split_step = parameters[
                    "hpipm_split_step"
                ]
            if "hpipm_prediction_correction" in parameters:
                ssqp_params.hpipmParams.prediction_correction = parameters[
                    "hpipm_prediction_correction"
                ]
            if "hpipm_warm_start" in parameters:
                ssqp_params.hpipmParams.warm_start = parameters[
                    "hpipm_warm_start"
                ]
            if "hpipm_square_root_riccati" in parameters:
                ssqp_params.hpipmParams.square_root_riccati = parameters[
                    "hpipm_square_root_riccati"
                ]
        else:
            ssqp_params.admmParams.max_iter = parameters.get(
                "admm_max_iter", 100
            )
            ssqp_params.admmParams.abs_tolerance = parameters.get(
                "admm_tol", 1e-3
            )
            ssqp_params.admmParams.rel_tolerance = parameters.get(
                "admm_tol", 1e-3
            )
            if "admm_adaptive_rho_tolerance" in parameters:
                ssqp_params.admmParams.adaptive_rho_tolerance = parameters[
                    "admm_adaptive_rho_tolerance"
                ]
            if "admm_rho_update_iter_freq" in parameters:
                ssqp_params.admmParams.rho_update_iter_freq = parameters[
                    "admm_rho_update_iter_freq"
                ]
            if "admm_warm_start" in parameters:
                ssqp_params.admmParams.warm_start = parameters.get(
                    "admm_warm_start", True
                )
            if "admm_reset_rho" in parameters:
                ssqp_params.admmParams.reset_rho = parameters["admm_reset_rho"]
            if "admm_reset_y" in parameters:
                ssqp_params.admmParams.reset_y = parameters["admm_reset_y"]
            if "admm_alpha" in parameters:
                ssqp_params.admmParams.alpha = parameters["admm_alpha"]

        self.solver = ssqpy.solvers.ssqp(self.model, ssqp_params)
        self._benchmark_solver = (
            "ssqp-hpipm"
            if ssqp_params.qp_solver_type == ssqpy.solvers.QPSolverType.HPIPM
            else "ssqp-admm"
        )
        self.status, _, _ = self.solver.solve()

        convergence_data = self.solver.getConvergenceData()

        print(
            f"[SSQPY] Solve time: {convergence_data.total_duration / 1000} ms"
        )

    def save(self, output_json):
        # The native ssqpy.solvers.ssqp.write() writer needs XML-derived
        # joint/frame metadata to serialize a full trajectory, which this
        # custom-dynamics model doesn't have
        #  plot_benchmarks.py only reads the "benchmark" summary below, so
        # that's all this writes
        convergence = self.solver.getConvergenceData()
        successful_states = {
            ssqpy.solvers.SolverState.Success,
            ssqpy.solvers.SolverState.LooseSuccess,
            ssqpy.solvers.SolverState.ConstraintSuccess,
        }
        output = {
            "benchmark": {
                "format_version": 1,
                "solver": self._benchmark_solver,
                "success": self.status in successful_states,
                "iterations": convergence.iterations,
                "runtime_ms": convergence.total_duration / 1000.0,
                "objective": convergence.trajectory_cost,
                "constraint_violation": convergence.trajectory_constraint_violation,
                "stage_gap": convergence.trajectory_stage_gap,
            }
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[SSQPY] Saved solution to {output_json}")
