"""
AligatorDuffingProblem: an Aligator (proxddp) formulation of the scalable
coupled Duffing oscillator benchmark (see oscillators.py), built directly
on aligator's dynamics.ODEAbstract/CostAbstract extension points
There's no robot URDF here, so the continuous dynamics/cost and their analytic
derivatives come straight from CoupledDuffingOCP.

Every stage shares the same dynamics and cost, since neither depends on
k for this problem
"""

import json
from time import perf_counter

import numpy as np

import aligator
from aligator import dynamics, manifolds

from common.oscillators import CoupledDuffingOCP, DuffingConfig


class _DuffingForwardDynamics(dynamics.ODEAbstract):
    """Continuous dynamics xdot = f(x,u), taken directly from
    CoupledDuffingOCP.continuous_dynamics/continuous_jacobians
    Integrated by aligator's IntegratorEuler (x_next = x + dt*xdot, using the
    current state for the update), matching oscillators.py's forward-Euler
    dynamics() convention exactly"""

    def __init__(self, space, problem: CoupledDuffingOCP):
        super().__init__(space, problem.nu)
        self.problem = problem

    def __deepcopy__(self, memo):
        # All mutable state lives on the `data` object aligator passes
        # into forward()/dForward(), not on self, so sharing the same
        # instance across copies is safe
        return self

    def forward(self, x, u, data):
        data.xdot[:] = self.problem.continuous_dynamics(x, u)

    def dForward(self, x, u, data):
        Fx, Fu = self.problem.continuous_jacobians(x, u)
        data.Jx[:, :] = Fx
        data.Ju[:, :] = Fu


class _DuffingStageCost(aligator.CostAbstract):
    def __init__(self, space, nu, problem: CoupledDuffingOCP):
        super().__init__(space, nu)
        self.problem = problem

    def __deepcopy__(self, memo):
        return self

    def evaluate(self, x, u, data):
        data.value = self.problem.stage_cost(x, u)

    def computeGradients(self, x, u, data):
        nx = self.problem.nx
        _, g, _ = self.problem.stage_cost_derivatives(x, u)
        data.Lx[:] = g[:nx]
        data.Lu[:] = g[nx:]

    def computeHessians(self, x, u, data):
        nx = self.problem.nx
        _, _, H = self.problem.stage_cost_derivatives(x, u)
        data.Lxx[:, :] = H[:nx, :nx]
        data.Lxu[:, :] = H[:nx, nx:]
        data.Lux[:, :] = H[nx:, :nx]
        data.Luu[:, :] = H[nx:, nx:]


class _DuffingTerminalCost(aligator.CostAbstract):
    """Same CostAbstract interface as _DuffingStageCost, but with no
    u-dependence: aligator still sizes Lu/Luu/Lxu/Lux to nu, they're
    just left at zero"""

    def __init__(self, space, nu, problem: CoupledDuffingOCP):
        super().__init__(space, nu)
        self.problem = problem

    def __deepcopy__(self, memo):
        return self

    def evaluate(self, x, u, data):
        data.value = self.problem.terminal_cost(x)

    def computeGradients(self, x, u, data):
        _, g, _ = self.problem.terminal_cost_derivatives(x)
        data.Lx[:] = g
        data.Lu[:] = 0.0

    def computeHessians(self, x, u, data):
        _, _, H = self.problem.terminal_cost_derivatives(x)
        data.Lxx[:, :] = H
        data.Lxu[:, :] = 0.0
        data.Lux[:, :] = 0.0
        data.Luu[:, :] = 0.0


class AligatorDuffingProblem:
    def __init__(
        self, horizon, dt, duffing_config: DuffingConfig, parameters=None
    ):
        self.H = horizon
        self.dt = dt
        self.problem = CoupledDuffingOCP(duffing_config)

        self.space = manifolds.VectorSpace(self.problem.nx)
        self.nu = self.problem.nu

        self.x0 = None
        self.traj_problem = None
        self.solver = None
        self.xs_opt = None
        self.us_opt = None
        self.solve_time = None
        self.converged = None
        self.iterations = None
        self.objective = None
        self.constraint_violation = None
        self.stage_gap = None

    def finalize(self, parameter_dict=None):
        self.x0 = self.problem.initial_state()

        ode = _DuffingForwardDynamics(self.space, self.problem)
        discrete_dynamics = dynamics.IntegratorEuler(ode, self.dt)
        stage_cost = _DuffingStageCost(self.space, self.nu, self.problem)
        terminal_cost = _DuffingTerminalCost(self.space, self.nu, self.problem)

        # One StageModel per stage, but all of them wrap the same
        # shared cost/dynamics instances since neither varies with k here
        stages = [
            aligator.StageModel(stage_cost, discrete_dynamics)
            for _ in range(self.H)
        ]

        self.traj_problem = aligator.TrajOptProblem(
            self.x0, stages, terminal_cost
        )

    def solve(self, parameters=None):
        parameters = parameters or {}
        if self.traj_problem is None:
            self.finalize()

        tol = parameters.get("tol", parameters.get("tolerance", 1e-2))
        mu_init = parameters.get("mu_init", 1e-2)
        max_iters = parameters.get("max_iters", 1000)
        verbose = parameters.get("verbose", aligator.VerboseLevel.QUIET)

        self.solver = aligator.SolverProxDDP(tol, mu_init, verbose=verbose)
        self.solver.max_iters = max_iters
        self.solver.setup(self.traj_problem)

        xs_init = [self.x0.copy() for _ in range(self.H + 1)]
        us_init = [np.zeros(self.nu) for _ in range(self.H)]

        start = perf_counter()
        self.converged = self.solver.run(self.traj_problem, xs_init, us_init)
        self.solve_time = perf_counter() - start

        results = self.solver.results
        self.xs_opt = results.xs
        self.us_opt = results.us
        self.iterations = getattr(self.solver, "iter", None)
        if self.iterations is None:
            self.iterations = getattr(results, "num_iters", None)
        result_cost = getattr(results, "traj_cost", None)
        self.objective = (
            float(result_cost) if result_cost is not None else None
        )
        # No path constraints in this problem (unlike CasadiDuffingProblem's
        # direct transcription), so there's nothing to report here -- the
        # dynamics-transition defect is tracked separately below, as
        # stage_gap
        self.constraint_violation = 0.0
        self.stage_gap = self._evaluate_stage_gap()

        print(
            f"[Aligator] Solve time: {self.solve_time:.3f} s, "
            f"converged: {self.converged}"
        )
        return self.converged

    def _evaluate_stage_gap(self):
        """Sum, over stages, of the infinity norm of how far the reported
        trajectory's x_{k+1} sits from problem.dynamics(x_k, u_k) --
        ProxDDP tracks this defect internally during optimization
        (workspace.dyn_slacks) and only drives it toward zero as tightly
        as the outer tolerance demands, so it's generally nonzero at
        convergence, same as SSQP's own trajectory_stage_gap"""
        total = 0.0
        for k in range(self.H):
            defect = np.asarray(self.xs_opt[k + 1]) - self.problem.dynamics(
                np.asarray(self.xs_opt[k]), np.asarray(self.us_opt[k])
            )
            total += float(np.max(np.abs(defect)))
        return total

    def save(self, output_json):
        if self.xs_opt is None:
            raise RuntimeError("save() called before solve()")

        # plot_benchmarks.py (the only consumer of this problem's results
        # so far) only reads the "benchmark" summary below; there's no
        # visualizer for this problem, so the full per-stage
        # trajectory schema the robot problems write isn't needed
        output = {
            "benchmark": {
                "format_version": 1,
                "solver": "aligator",
                "success": bool(self.converged),
                "iterations": self.iterations,
                "runtime_ms": self.solve_time * 1000.0,
                "objective": self.objective,
                "constraint_violation": self.constraint_violation,
                "stage_gap": self.stage_gap,
            }
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[Aligator] Saved solution to {output_json}")
