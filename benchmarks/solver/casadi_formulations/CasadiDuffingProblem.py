"""
CasadiDuffingProblem: a CasADi/Opti formulation of the scalable coupled
Duffing oscillator benchmark (see oscillators.py), built directly on
CasADi's own symbolic ops
"""

import json
import time

import casadi as ca
import numpy as np

from common.oscillators import CoupledDuffingOCP, DuffingConfig

# Same rationale as CasadiProblem's _JIT_OPTIONS: JIT-compile each cached
# dynamics/cost ca.Function individually (built once, called at every
# stage) rather than the whole assembled NLP
_JIT_OPTIONS = {
    "jit": True,
    "compiler": "shell",
    "jit_options": {"compiler": "gcc", "flags": ["-O3"]},
}
_NO_JIT_OPTIONS = {}


class CasadiDuffingProblem:
    def __init__(
        self, horizon, dt, duffing_config: DuffingConfig, parameters=None
    ):
        cp = dict(parameters or {})

        self.H = horizon
        self.dt = dt
        self.problem = CoupledDuffingOCP(duffing_config)
        self.casadi_plugin = cp.get("casadi_plugin", "ipopt")
        self._jit_opts = (
            _JIT_OPTIONS if cp.get("jit", False) else _NO_JIT_OPTIONS
        )

        self.N = horizon + 1

        self._dyn_fn = None
        self._stage_cost_fn = None
        self._terminal_cost_fn = None

        self._opti = None
        self._x = None
        self._u = None
        self._sol = None
        self._nu_list = None
        self.solve_time = None
        self.converged = None
        self.iterations = None
        self.objective = None
        self.constraint_violation = None

    def finalize(self, parameter_dict=None):
        # Nothing to precompute beyond what solve() itself builds; kept
        # only so run_benchmark.py's generic dispatch (which calls
        # finalize() before solve()) has something to call
        pass

    # Cached symbolic dynamics/cost, built once in SX and reused via
    # ca.Function at every stage (same rationale as CasadiProblem's
    # cached kinematics functions)
    def _build_dynamics_fn(self):
        if self._dyn_fn is not None:
            return self._dyn_fn

        p = self.problem
        cfg = p.cfg
        m = p.m

        q = ca.SX.sym("q", m)
        v = ca.SX.sym("v", m)
        u = ca.SX.sym("u", m)

        if cfg.coupling == "local":
            L = ca.DM(p._L)
            kappa = p._local_gain()
            coupling = kappa * (L @ q)
        else:
            q_row = ca.repmat(q.T, m, 1)
            q_col = ca.repmat(q, 1, m)
            delta = q_row - q_col
            c = cfg.coupling_strength / m
            coupling = c * (ca.sin(delta) @ ca.SX.ones(m, 1))

        a = (
            -cfg.damping * v
            - cfg.stiffness * q
            - cfg.cubic * q**3
            + coupling
            + u
        )

        # Explicit (forward) Euler, matching oscillators.py's
        # dynamics(): x_next = x + dt*[v; a], using the *current* v for
        # the position update, not the updated v_next
        v_next = v + self.dt * a
        q_next = q + self.dt * v
        x_next = ca.vertcat(q_next, v_next)

        fn = ca.Function("f_duffing_dyn", [q, v, u], [x_next], self._jit_opts)
        self._dyn_fn = fn
        return fn

    def _cost_normalization(self):
        return (
            1.0 / self.problem.m
            if self.problem.cfg.normalize_cost_by_m
            else 1.0
        )

    def _build_stage_cost_fn(self):
        if self._stage_cost_fn is not None:
            return self._stage_cost_fn

        cfg = self.problem.cfg
        m = self.problem.m

        q = ca.SX.sym("q", m)
        v = ca.SX.sym("v", m)
        u = ca.SX.sym("u", m)

        scale = cfg.stage_cost_scale * self._cost_normalization()
        value = scale * (
            0.5 * cfg.q_weight * ca.sumsqr(q)
            + 0.5 * cfg.v_weight * ca.sumsqr(v)
            + 0.5 * cfg.u_weight * ca.sumsqr(u)
            + 0.25 * cfg.quartic_cost * ca.sumsqr(q**2)
        )

        fn = ca.Function(
            "f_duffing_stage_cost", [q, v, u], [value], self._jit_opts
        )
        self._stage_cost_fn = fn
        return fn

    def _build_terminal_cost_fn(self):
        if self._terminal_cost_fn is not None:
            return self._terminal_cost_fn

        cfg = self.problem.cfg
        m = self.problem.m

        q = ca.SX.sym("q", m)
        v = ca.SX.sym("v", m)

        scale = cfg.terminal_cost_scale * self._cost_normalization()
        value = scale * (
            0.5 * cfg.terminal_q_weight * ca.sumsqr(q)
            + 0.5 * cfg.terminal_v_weight * ca.sumsqr(v)
            + 0.25 * cfg.terminal_quartic_cost * ca.sumsqr(q**2)
        )

        fn = ca.Function(
            "f_duffing_terminal_cost", [q, v], [value], self._jit_opts
        )
        self._terminal_cost_fn = fn
        return fn

    # ------------------------------------------------------------------
    # Transcription + solve
    # ------------------------------------------------------------------
    def solve(self, parameter_dict):
        solve_params = dict(parameter_dict or {})
        plugin = solve_params.pop("casadi_plugin", self.casadi_plugin)
        solve_params.pop("tolerance", None)  # ssqpy-only knob, ignored here

        p = self.problem
        m = p.m
        H = self.H
        N = self.N
        x0 = p.initial_state()

        dyn_fn = self._build_dynamics_fn()
        stage_cost_fn = self._build_stage_cost_fn()
        terminal_cost_fn = self._build_terminal_cost_fn()

        nx_list = [p.nx] * N
        # No control/force variables at the terminal node, matching
        # CasadiProblem's interleaving convention: a size-0 opti
        # variable keeps the x0,u0,x1,u1,...,x_{N-1},u_{N-1} pattern
        # Fatrop's manual structure detection expects
        nu_list = [p.nu] * H + [0]

        opti = ca.Opti()
        x = []
        u = []
        for k in range(N):
            x.append(opti.variable(nx_list[k]))
            u.append(opti.variable(nu_list[k]))

        ng = []
        violation_terms = {k: [] for k in range(N)}
        for k in range(N):
            q_k = x[k][:m]
            v_k = x[k][m:]

            if k < N - 1:
                x_next = dyn_fn(q_k, v_k, u[k])
                opti.subject_to(x[k + 1] == x_next)
                violation_terms[k].append((x_next, x[k + 1], x_next))

            ng.append(0)

            def _add(lb, expr, ub, k=k):
                opti.subject_to((lb <= expr) <= ub)
                ng[-1] += expr.nnz()
                violation_terms[k].append((lb, expr, ub))

            if k == 0:
                _add(ca.DM(x0), x[k], ca.DM(x0))

        J = 0
        for k in range(N):
            q_k = x[k][:m]
            v_k = x[k][m:]
            if k < H:
                J += stage_cost_fn(q_k, v_k, u[k])
            else:
                J += terminal_cost_fn(q_k, v_k)

        opti.minimize(J)

        def _set_cold_start_guess():
            for k in range(N):
                opti.set_initial(x[k], ca.DM(x0))
                if nu_list[k] > 0:
                    opti.set_initial(u[k], ca.DM.zeros(nu_list[k]))

        _set_cold_start_guess()

        if plugin == "fatrop":
            solve_params.setdefault("structure_detection", "manual")
            solve_params.setdefault("nx", nx_list)
            solve_params.setdefault("nu", nu_list)
            solve_params.setdefault("ng", ng)
            solve_params.setdefault("N", H)

        opti.solver(plugin, solve_params)

        if self._jit_opts is _JIT_OPTIONS or plugin == "fatrop":
            # opti.solver() above only selects the plugin/options; the
            # NLP's own derivative functions (nlp_jac_g, nlp_hess_l, ...)
            # aren't actually built, and therefore not JIT-compiled, until
            # this solver instance's first solve() call, so without this
            # warm-up that one-time gcc compile+link cost would land inside
            # the timed region below on every solve(), instead of being paid
            # once ahead of time as --jit intends
            # Fatrop's structure_detection="manual" pays its own one-time
            # cost on that same first solve() call, unrelated to --jit --
            # CasADi has to translate the flat NLP Jacobian/Hessian into
            # Fatrop's per-stage block layout, and that translation scales
            # far worse than linearly with per-stage size, so it needs the
            # same warm-up treatment regardless of whether --jit is set
            # Reusing this same Opti/nlpsol instance for a throwaway
            # untimed solve forces both of these one-time costs now,
            # whether it converges is irrelevant, only that it exercises
            # every derivative path the real solve will need
            # The initial guess is then reset back to the cold-start
            # values so the timed solve below still measures genuine
            # from-scratch convergence, with the same iteration count as
            # without this warm-up
            try:
                opti.solve()
            except RuntimeError:
                pass
            _set_cold_start_guess()

        try:
            start = time.perf_counter()
            sol = opti.solve()
            self.solve_time = time.perf_counter() - start
        except RuntimeError:
            print("CasADi solve failed; last iterate for the final state:")
            print(opti.debug.value(x[-1]))
            raise

        self._opti = opti
        self._x = x
        self._u = u
        self._sol = sol
        self._nu_list = nu_list
        stats = sol.stats()
        self.converged = bool(stats.get("success", True))
        self.iterations = stats.get("iter_count")
        self.objective = float(sol.value(J))
        self.constraint_violation = self._evaluate_violation(
            sol, violation_terms
        )
        print(
            f"[{self.casadi_plugin}] Solve time: {self.solve_time:.3f} s, "
            f"converged: {self.converged}"
        )
        return sol

    def _evaluate_violation(self, sol, violation_terms):
        """Same convention as CasadiProblem._evaluate_violation: per stage,
        the infinity norm of how far outside [lb, ub] the (possibly
        vector-valued) constraint sits, summed across stages"""

        def _num(term):
            if isinstance(term, (ca.MX, ca.SX)):
                return np.asarray(sol.value(term), dtype=float).reshape(-1)
            return np.asarray(ca.DM(term), dtype=float).reshape(-1)

        total = 0.0
        for terms in violation_terms.values():
            stage_worst = 0.0
            for lb, expr, ub in terms:
                lb_arr = _num(lb)
                ub_arr = _num(ub)
                value = _num(expr)
                violation = np.maximum(
                    0.0, np.maximum(lb_arr - value, value - ub_arr)
                )
                if violation.size:
                    stage_worst = max(stage_worst, float(violation.max()))
            total += stage_worst
        return total

    # ------------------------------------------------------------------
    # Saving results
    # ------------------------------------------------------------------
    def save(self, output_json):
        if self._sol is None:
            raise RuntimeError("save() called before solve()")

        # plot_benchmarks.py only reads the "benchmark" summary below
        output = {
            "benchmark": {
                "format_version": 1,
                "solver": self.casadi_plugin,
                "success": self.converged,
                "iterations": self.iterations,
                "runtime_ms": self.solve_time * 1000.0,
                "objective": self.objective,
                "constraint_violation": self.constraint_violation,
            }
        }
        with open(output_json, "w") as f:
            json.dump(output, f, indent=4)

        print(f"[{self.casadi_plugin}] Saved solution to {output_json}")
