import argparse
import os
import subprocess
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

# Must be set before numpy/aligator/etc (imported below) load their BLAS
# backend, since large problems (e.g. duffing_local/global at a big
# --duffing-m) cross BLAS's own auto-parallelization threshold once
# unset, causing thread-pool oversubscription on a many-core host rather
# than a speedup, with solve time dominated by scheduling noise instead
# of actual work
# setdefault() leaves an explicit external setting in place
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from common.benchmark_config import TOLERANCE, tolerance_label
from common.problems import *


def _read_meminfo_kb(key):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    raise RuntimeError(f"{key} not found in /proc/meminfo")


def total_memory_bytes():
    return _read_meminfo_kb("MemTotal") * 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Solver benchmark script")

    parser.add_argument(
        "problem",
        nargs="?",
        choices=[
            "pendubot",
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
            "panda_circle",
            "duffing_local",
            "duffing_global",
        ],
        help="The benchmarking problem to run. Ignored if --all-problems is given.",
    )

    parser.add_argument(
        "-s",
        "--solver",
        nargs="+",
        choices=[
            "ssqp-admm",
            "ssqp-hpipm",
            "fatrop",
            "ipopt",
            "aligator",
            "mim_solvers",
        ],
        default=["ssqp-admm"],
        help=(
            "The solver(s) to benchmark. Provide multiple values to run "
            "each. Ignored if --all-solvers is given."
        ),
    )

    parser.add_argument(
        "--all-solvers",
        action="store_true",
        help="Run every solver on the given problem, instead of just --solver.",
    )

    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Run every problem, instead of just the given problem.",
    )

    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        nargs="+",
        default=[TOLERANCE],
        help=(
            "Convergence tolerance every solver is run with. Provide "
            "multiple values to run a tolerance sweep."
        ),
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print each solver's own per-iteration progress output.",
    )

    parser.add_argument(
        "--jit",
        action="store_true",
        help=(
            "For the CasADi-based solvers (fatrop/ipopt) only: JIT-compile "
            "the per-frame dynamics/kinematics/pose-error functions via gcc "
            "instead of using CasADi's interpreted graph evaluation."
        ),
    )

    parser.add_argument(
        "--jit-opt-level",
        choices=["O0", "O1", "O2", "O3"],
        default="O3",
        help=(
            "gcc optimization level for --jit's generated C code. Lower "
            "this for problems where -O3 takes hours or never finishes "
            "(e.g. duffing at a large --duffing-m); -O0 trades slower "
            "generated code for a compile that actually completes. Only "
            "meaningful with --jit."
        ),
    )

    parser.add_argument(
        "--cold-start",
        action="store_true",
        help=(
            "For the CasADi-based solvers (fatrop/ipopt) only: run one "
            "untimed throwaway solve first so the timed solve doesn't pay "
            "for one-time setup costs (JIT compile, Fatrop's "
            "structure_detection='manual'). Fatrop and any --jit run "
            "already do this unconditionally; this extends it to Ipopt "
            "without --jit."
        ),
    )

    parser.add_argument(
        "--dynamics-mode",
        choices=["forward", "inverse"],
        default="forward",
        help=(
            "Dynamics formulation to solve. 'inverse' is supported for "
            "SSQP (ssqp-admm/ssqp-hpipm), where it maps to SolverMode."
            "InverseDynamics, or SolverMode.ReducedInverseDynamics for "
            "problems with a floating base (see SolverMode's docstring in "
            "include/types.hpp), and for the CasADi-based solvers "
            "(fatrop/ipopt), which formulate the equivalent explicit-"
            "equality-constrained scheme by hand (see CasadiProblem.solve()). "
        ),
    )

    parser.add_argument(
        "--duffing-m",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Only meaningful for duffing_local/duffing_global: number of "
            "coupled oscillators (state dimension nx=2m, control dimension "
            "nu=m). Defaults to the problem's own preset (16) if omitted. "
            "Provide multiple values (together with a multi-valued "
            "--duffing-horizon, or on its own) to run a grid search: every "
            "solver already selected (--solver or --all-solvers) is run "
            "against every (m, horizon) combination in the Cartesian "
            "product of this and --duffing-horizon, instead of a single "
            "run. With --all-problems, the grid applies only to "
            "duffing_local/duffing_global; every other problem still runs "
            "a single run per solver."
        ),
    )

    parser.add_argument(
        "--duffing-horizon",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Only meaningful for duffing_local/duffing_global: OCP horizon. "
            "Defaults to the problem's own preset (80) if omitted. Provide "
            "multiple values to run a grid search against --duffing-m (see "
            "its help)."
        ),
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Skip a run if its output JSON file already exists, instead of "
            "re-running it and overwriting the file. Useful for resuming a "
            "large sweep after a crash or a partial run."
        ),
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help=(
            "Number of runs (problem/solver/tolerance/grid-point "
            "combinations) to execute at once. Each run happens in its own "
            "subprocess (re-invoking this script for a single run), which "
            "also works around a CasADi JIT bug that leaks a file "
            "descriptor per compiled function (see --jit): staying in one "
            "process for a long sweep would otherwise accumulate leaked "
            "descriptors until the OS's open-file limit is hit."
        ),
    )

    parser.add_argument(
        "--memory-budget-fraction",
        type=float,
        default=0.9,
        help=(
            "When running more than one job (see --jobs), cap the total "
            "memory the scheduler will let concurrently-running runs "
            "reserve to this fraction of total system RAM. Each run's own "
            "share is estimated up front (a formula based on m/horizon for "
            "duffing_local/duffing_global, or a flat ~2 GiB for every "
            "other problem, see estimate_memory_bytes()) and reserved the "
            "moment it launches, before the OS shows any real usage, so "
            "the budget accounts for runs that haven't ramped up their "
            "memory use yet -- not just runs the OS already reports as "
            "using memory. A run that doesn't fit is queued until enough "
            "already-running runs finish and release their share."
        ),
    )

    parser.add_argument(
        "--max-memory-retries",
        type=int,
        default=3,
        help=(
            "How many times to re-queue a run that gets killed by the "
            "OS's OOM killer before giving up on it and "
            "reporting it as a failure. Only applies to runs dispatched "
            "as their own subprocess (see --jobs); an OOM kill on a lone "
            "run (neither --all-problems/--all-solvers nor a grid "
            "search) is reported immediately."
        ),
    )

    return parser.parse_args()


args = parse_args()
if not args.all_problems and args.problem is None:
    raise SystemExit("Either a problem or --all-problems must be given.")
if args.jobs < 1:
    raise SystemExit("--jobs must be at least 1")
if not 0 < args.memory_budget_fraction <= 1:
    raise SystemExit(
        "--memory-budget-fraction must be between 0 (exclusive) and 1"
    )
if args.max_memory_retries < 0:
    raise SystemExit("--max-memory-retries must be at least 0")

inverse_dynamics = args.dynamics_mode == "inverse"
tolerances_to_run = args.tolerance

ALL_PROBLEMS = [
    "pendubot",
    "go2_standing",
    "go2_jump",
    "go2_wheelie",
    "go2_frog",
    "panda_circle",
    "duffing_local",
    "duffing_global",
]
problems_to_run = ALL_PROBLEMS if args.all_problems else [args.problem]

ALL_SOLVERS = [
    "ssqp-admm",
    "ssqp-hpipm",
    "fatrop",
    "ipopt",
    "aligator",
    "mim_solvers",
]
# mim_solvers/aligator only build a forward-dynamics problem (see their
# ProblemWrapper implementations); there is no inverse-dynamics formulation
# to select for them
INVERSE_UNSUPPORTED = {"mim_solvers", "aligator"}

# The Duffing oscillator problems have no robot/URDF underneath them, so
# each solver builds them directly on its own custom-dynamics extension
# point
DUFFING_PROBLEMS = {"duffing_local", "duffing_global"}

if (
    not args.all_problems
    and args.problem not in DUFFING_PROBLEMS
    and (args.duffing_m is not None or args.duffing_horizon is not None)
):
    raise SystemExit(
        "--duffing-m/--duffing-horizon only apply to duffing_local/"
        f"duffing_global, not --problem {args.problem}"
    )

duffing_m_list = args.duffing_m if args.duffing_m is not None else [None]
duffing_horizon_list = (
    args.duffing_horizon if args.duffing_horizon is not None else [None]
)
grid_search = len(duffing_m_list) * len(duffing_horizon_list) > 1
if (
    grid_search
    and not args.all_problems
    and args.problem not in DUFFING_PROBLEMS
):
    raise SystemExit(
        "--duffing-m/--duffing-horizon with multiple values need --problem "
        "to be duffing_local or duffing_global, or --all-problems"
    )

duffing_m_scalar = duffing_m_list[0] if not grid_search else None
duffing_horizon_scalar = duffing_horizon_list[0] if not grid_search else None

if args.all_solvers:
    solvers = [
        s
        for s in ALL_SOLVERS
        if not (inverse_dynamics and s in INVERSE_UNSUPPORTED)
    ]
    if inverse_dynamics:
        skipped = [s for s in ALL_SOLVERS if s in INVERSE_UNSUPPORTED]
        print(
            f"Skipping {', '.join(skipped)}: --dynamics-mode inverse is "
            "not supported for them."
        )
else:
    unsupported = [s for s in args.solver if s in INVERSE_UNSUPPORTED]
    if inverse_dynamics and unsupported:
        raise SystemExit(
            "--dynamics-mode inverse is not supported for "
            f"--solver {', '.join(unsupported)} (see --help for why)"
        )
    if (
        not args.all_problems
        and args.problem in DUFFING_PROBLEMS
        and inverse_dynamics
    ):
        raise SystemExit(
            f"--dynamics-mode inverse is not meaningful for {args.problem} "
            "(it has no robot dynamics to invert)"
        )
    solvers = args.solver

# Outer-iteration caps, sized per problem from observed convergence behavior
MAX_ITER_BY_PROBLEM = {
    "pendubot": 500,
    "go2_standing": 500,
    "go2_jump": 1000,
    "go2_wheelie": 1000,
    "go2_frog": 1000,
    "panda_circle": 1000,
    "duffing_local": 300,
    "duffing_global": 300,
}
problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
    "go2_jump": make_go2_jump_problem,
    "go2_wheelie": make_go2_wheelie_problem,
    "go2_frog": make_go2_frog_problem,
    "panda_circle": make_panda_circle_problem,
    "duffing_local": make_duffing_local_problem,
    "duffing_global": make_duffing_global_problem,
}

# Per-(problem, solver) solve_parameters overrides, applied right before
# solve() on top of that solver's generic defaults above
PROBLEM_SOLVER_OVERRIDES = {
    ("pendubot", "ssqp-hpipm"): {
        "hpipm_tol": lambda tolerance: tolerance,
        "hpipm_square_root_riccati": False,
    },
    ("go2_jump", "ssqp-hpipm"): {
        "hpipm_split_step": True,
        "hpipm_prediction_correction": True,
        "hpipm_warm_start": True,
        "hpipm_square_root_riccati": False,
    },
    ("go2_wheelie", "ssqp-hpipm"): {
        "hpipm_split_step": True,
    },
    ("go2_frog", "ssqp-hpipm"): {
        "hpipm_square_root_riccati": False,
    },
    ("pendubot", "ssqp-admm"): {
        "admm_alpha": 1.8,
    },
    ("go2_standing", "ssqp-admm"): {
        "admm_reset_rho": False,
        "admm_reset_y": False,
    },
    ("go2_jump", "ssqp-admm"): {
        "admm_alpha": 1.8,
        "admm_max_iter": 300,
        "admm_reset_rho": False,
        "admm_reset_y": False,
    },
    ("go2_wheelie", "ssqp-admm"): {
        "admm_reset_rho": False,
        "admm_reset_y": False,
        "admm_alpha": 1.0,
    },
    ("go2_frog", "ssqp-admm"): {
        "admm_max_iter": 300,
        "admm_reset_rho": False,
        "admm_reset_y": False,
    },
    ("pendubot", "mim_solvers"): {
        "reset_rho": False,
        "reset_y": False,
        "update_rho_with_heuristic": False,
    },
    ("go2_standing", "mim_solvers"): {
        "reset_rho": True,
        "reset_y": True,
        "update_rho_with_heuristic": False,
    },
    ("panda_circle", "mim_solvers"): {
        "reset_rho": False,
        "reset_y": False,
        "update_rho_with_heuristic": False,
    },
    ("go2_jump", "aligator"): {
        "mu_init": 0.0005,
    },
    ("go2_wheelie", "aligator"): {
        "mu_init": 0.001,
    },
    ("pendubot", "aligator"): {
        "mu_init": 0.002,
    },
    ("go2_jump", "ipopt"): {
        "ipopt": {"hessian_approximation": "limited-memory"},
    },
    ("go2_wheelie", "ipopt"): {
        "ipopt": {"hessian_approximation": "limited-memory"},
    },
    ("go2_standing", "fatrop"): {
        "fatrop": {"mu_init": 0.1},
    },
    ("panda_circle", "fatrop"): {
        "fatrop": {"mu_init": 1.0},
    },
}


def run_solver(problem, solver):
    if problem in DUFFING_PROBLEMS and inverse_dynamics:
        raise ValueError(
            f"--dynamics-mode inverse is not meaningful for {problem} (it "
            "has no robot dynamics to invert)"
        )

    max_iter = MAX_ITER_BY_PROBLEM[problem]
    problem_wrapper_t = None
    constructor_parameters = {}
    solve_parameters = {}
    if problem in DUFFING_PROBLEMS and (
        solver == "ssqp-admm" or solver == "ssqp-hpipm"
    ):
        from ssqp_formulations.SSQPDuffingProblem import SSQPDuffingProblem

        problem_wrapper_t = SSQPDuffingProblem

        if duffing_m_scalar is not None:
            constructor_parameters["m"] = duffing_m_scalar
        if duffing_horizon_scalar is not None:
            constructor_parameters["horizon"] = duffing_horizon_scalar

        solve_parameters["tolerance"] = tolerance
        solve_parameters["max_iter"] = max_iter
        solve_parameters["verbose"] = args.verbose

        if solver == "ssqp-admm":
            solve_parameters["qp_solver"] = "admm"
            solve_parameters["admm_max_iter"] = 100
            solve_parameters["admm_tol"] = tolerance / 10
            solve_parameters["admm_adaptive_rho_tolerance"] = 5
            solve_parameters["admm_rho_update_iter_freq"] = 25
            solve_parameters["admm_warm_start"] = True
        else:
            solve_parameters["qp_solver"] = "hpipm"
            solve_parameters["hpipm_tol"] = tolerance / 10

    elif solver == "ssqp-admm" or solver == "ssqp-hpipm":
        import ssqpy
        from ssqp_formulations.SSQPProblem import SSQPProblem

        problem_wrapper_t = SSQPProblem

        is_floating_base = problem in (
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
        )
        if inverse_dynamics:
            constructor_parameters["solver_mode"] = (
                ssqpy.model.SolverMode.ReducedInverseDynamics
                if is_floating_base
                else ssqpy.model.SolverMode.InverseDynamics
            )
        else:
            constructor_parameters["solver_mode"] = (
                ssqpy.model.SolverMode.ForwardDynamics
            )
        if problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif is_floating_base:
            constructor_parameters["free_flyer"] = True

        solve_parameters["tolerance"] = tolerance
        solve_parameters["max_iter"] = max_iter
        solve_parameters["verbose"] = args.verbose

        if solver == "ssqp-admm":
            solve_parameters["qp_solver"] = "admm"
            # panda_circle's per-stage QP needs a larger ADMM budget than
            # pendubot/go2's to converge within the outer SQP loop
            solve_parameters["admm_max_iter"] = (
                400
                if problem == "panda_circle"
                else (300 if problem == "pendubot" else 100)
            )
            solve_parameters["admm_tol"] = tolerance / 10
            # rho_update_iter_freq also gates ADMM's own termination check (see
            # ssqp's admm_qp.hpp), so it must be lower than admm_max_iter for
            # ADMM to ever detect convergence and exit early
            solve_parameters["admm_adaptive_rho_tolerance"] = 5
            solve_parameters["admm_rho_update_iter_freq"] = 25
            solve_parameters["admm_warm_start"] = True
        else:
            solve_parameters["qp_solver"] = "hpipm"
            solve_parameters["hpipm_tol"] = tolerance / 10

    elif problem in DUFFING_PROBLEMS and solver == "mim_solvers":
        from mim_solvers_formulations.MimSolversDuffingProblem import (
            MimSolversDuffingProblem,
        )

        problem_wrapper_t = MimSolversDuffingProblem

        if duffing_m_scalar is not None:
            constructor_parameters["m"] = duffing_m_scalar
        if duffing_horizon_scalar is not None:
            constructor_parameters["horizon"] = duffing_horizon_scalar

        solve_parameters["tolerance"] = tolerance
        solve_parameters["max_iter"] = max_iter
        solve_parameters["max_qp_iters"] = 300
        solve_parameters["eps_abs"] = tolerance / 10
        solve_parameters["eps_rel"] = tolerance / 10
        solve_parameters["adaptive_rho_tolerance"] = 1
        solve_parameters["rho_update_interval"] = 100
        solve_parameters["callbacks"] = args.verbose

    elif solver == "mim_solvers":
        from mim_solvers_formulations.MimSolversProblem import (
            MimSolversProblem,
            PendubotActuationModel,
        )
        import pinocchio as pin
        import crocoddyl

        problem_wrapper_t = MimSolversProblem

        if problem == "pendubot":
            pin_model = pin.buildModelFromUrdf(
                "urdf/double_pendulum/pendubot.urdf"
            )
            state = crocoddyl.StateMultibody(pin_model)
            custom_actuation = PendubotActuationModel(state, pin_model)

            constructor_parameters["floating_base"] = False
            constructor_parameters["actuation_model"] = custom_actuation
        elif problem in (
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
        ):
            constructor_parameters["floating_base"] = True
        elif problem == "panda_circle":
            # Explicit: adding a free-flyer joint on top of a fixed-base URDF
            # like panda.urdf doesn't raise, so MimSolversProblem's own
            # try/except fallback to fixed-base would never trigger here
            constructor_parameters["floating_base"] = False

        solve_parameters["tolerance"] = tolerance
        solve_parameters["max_iter"] = max_iter
        solve_parameters["max_qp_iters"] = 300
        solve_parameters["eps_abs"] = tolerance / 10
        solve_parameters["eps_rel"] = tolerance / 10
        solve_parameters["adaptive_rho_tolerance"] = 1
        solve_parameters["rho_update_interval"] = 100
        # CallbackVerbose/CallbackLogger add per-iteration logging and
        # bookkeeping to the timed solve, so only enable them when asked
        solve_parameters["callbacks"] = args.verbose

    elif problem in DUFFING_PROBLEMS and solver == "aligator":
        import aligator
        from aligator_formulations.AligatorDuffingProblem import (
            AligatorDuffingProblem,
        )

        problem_wrapper_t = AligatorDuffingProblem

        if duffing_m_scalar is not None:
            constructor_parameters["m"] = duffing_m_scalar
        if duffing_horizon_scalar is not None:
            constructor_parameters["horizon"] = duffing_horizon_scalar

        solve_parameters["tol"] = tolerance
        solve_parameters["mu_init"] = 1e-2
        solve_parameters["verbose"] = (
            aligator.VerboseLevel.VERBOSE
            if args.verbose
            else aligator.VerboseLevel.QUIET
        )
        solve_parameters["max_iters"] = max_iter

    elif solver == "aligator":
        import aligator
        from aligator_formulations.AligatorProblem import AligatorProblem

        problem_wrapper_t = AligatorProblem

        if problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif problem in (
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
        ):
            constructor_parameters["free_flyer"] = True

        solve_parameters["tol"] = tolerance
        solve_parameters["mu_init"] = 1e-2
        solve_parameters["verbose"] = (
            aligator.VerboseLevel.VERBOSE
            if args.verbose
            else aligator.VerboseLevel.QUIET
        )
        solve_parameters["max_iters"] = max_iter

    elif problem in DUFFING_PROBLEMS and (
        solver == "fatrop" or solver == "ipopt"
    ):
        from casadi_formulations.CasadiDuffingProblem import (
            CasadiDuffingProblem,
        )

        problem_wrapper_t = CasadiDuffingProblem

        if duffing_m_scalar is not None:
            constructor_parameters["m"] = duffing_m_scalar
        if duffing_horizon_scalar is not None:
            constructor_parameters["horizon"] = duffing_horizon_scalar
        constructor_parameters["casadi_plugin"] = solver
        constructor_parameters["jit"] = args.jit
        constructor_parameters["jit_opt_level"] = args.jit_opt_level
        constructor_parameters["cold_start"] = args.cold_start

        solve_parameters["print_time"] = args.verbose

        if solver == "fatrop":
            solve_parameters["fatrop"] = {
                "tol": tolerance,
                "max_iter": max_iter,
                "print_level": 5 if args.verbose else 0,
            }
        else:
            solve_parameters["ipopt"] = {
                "max_iter": max_iter,
                "tol": tolerance,
                "acceptable_tol": 1e-3,
                "mu_strategy": "adaptive",
                "print_level": 5 if args.verbose else 0,
                "sb": "yes",
            }

    elif solver == "fatrop" or solver == "ipopt":
        from casadi_formulations.CasadiProblem import CasadiProblem

        problem_wrapper_t = CasadiProblem

        constructor_parameters["casadi_plugin"] = solver
        constructor_parameters["jit"] = args.jit
        constructor_parameters["jit_opt_level"] = args.jit_opt_level
        constructor_parameters["cold_start"] = args.cold_start
        constructor_parameters["inverse_dynamics"] = inverse_dynamics
        if problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif problem in (
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
        ):
            constructor_parameters["free_flyer"] = True

        solve_parameters["print_time"] = args.verbose

        if solver == "fatrop":
            # nx/nu/ng/N + structure_detection="manual" are filled in by
            # CasadiProblem.solve() once it knows the per-stage dimensions
            solve_parameters["fatrop"] = {
                "tol": tolerance,
                "max_iter": max_iter,
                "print_level": 5 if args.verbose else 0,
            }
            if problem in ("go2_standing", "go2_jump", "go2_wheelie"):
                solve_parameters["fatrop"]["mu_init"] = 1.0
        else:
            solve_parameters["ipopt"] = {
                "max_iter": max_iter,
                "tol": tolerance,
                "acceptable_tol": 1e-3,
                "mu_strategy": "adaptive",
                "print_level": 5 if args.verbose else 0,
                "sb": "yes",  # suppresses the startup banner too
            }
            if problem in ("go2_jump", "go2_wheelie"):
                # Same restoration-phase issue as Fatrop above
                solve_parameters["ipopt"]["mu_init"] = 1.0

    problem_wrapper = problems[problem](
        problem_wrapper_t, constructor_parameters
    )

    for key, val in PROBLEM_SOLVER_OVERRIDES.get(
        (problem, solver), {}
    ).items():
        val = val(tolerance) if callable(val) else val
        # A dict value merges into an existing dict-valued default (e.g.
        # fatrop/ipopt's nested options dict) instead of replacing it
        # outright, so an override doesn't have to repeat every option the
        # generic defaults above already set
        if isinstance(val, dict) and isinstance(
            solve_parameters.get(key), dict
        ):
            solve_parameters[key].update(val)
        else:
            solve_parameters[key] = val

    problem_wrapper.solve(solve_parameters)

    save_path = f"data/{solver}/"
    save_path = save_path.replace("-", "_")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_name = problem
    if inverse_dynamics:
        output_name += "_inverse_dynamics"
    if problem in DUFFING_PROBLEMS and (
        duffing_m_scalar is not None or duffing_horizon_scalar is not None
    ):
        # Keeps a scaled run from overwriting the preset's own default-size
        # result (or another scaled run at a different size), without
        # changing the default filename plot_benchmarks.py already expects
        output_name += f"_m{problem_wrapper.problem.m}_N{problem_wrapper.H}"
    output_name += f"_tol_{tolerance_label(tolerance)}"

    problem_wrapper.save(save_path + output_name + ".json")


# Must match the presets in common/problems.py's make_duffing_local/
# global_problem: used below both to predict a duffing run's output
# filename for --skip-existing, and as the (m, horizon) a run without an
# explicit --duffing-m/--duffing-horizon actually solves, for
# estimate_memory_bytes()
DUFFING_DEFAULT_M = 16
DUFFING_DEFAULT_HORIZON = 80


def expected_output_path(
    problem, solver, tolerance, duffing_m, duffing_horizon
):
    """Predict the JSON path run_solver() will save for this combination.

    Mirrors run_solver()'s own output_name construction, so --skip-existing
    can check for the file before paying for a subprocess/solve.
    """
    save_path = f"data/{solver}/".replace("-", "_")
    output_name = problem
    if inverse_dynamics:
        output_name += "_inverse_dynamics"
    if problem in DUFFING_PROBLEMS and (
        duffing_m is not None or duffing_horizon is not None
    ):
        m = duffing_m if duffing_m is not None else DUFFING_DEFAULT_M
        horizon = (
            duffing_horizon
            if duffing_horizon is not None
            else DUFFING_DEFAULT_HORIZON
        )
        output_name += f"_m{m}_N{horizon}"
    output_name += f"_tol_{tolerance_label(tolerance)}"
    return save_path + output_name + ".json"


DUFFING_MEM_BASE_BYTES = 512 * 1024**2  # fixed overhead: Python + imports
DUFFING_MEM_BYTES_PER_ELEMENT = 32
NON_DUFFING_MEMORY_ESTIMATE_BYTES = (
    2 * 1024**3
)  # flat guess, see --memory-budget-fraction

DUFFING_SOLVER_MEM_MULTIPLIER = {
    "ssqp-admm": 1.25,
    "ssqp-hpipm": 1.3,
    "fatrop": 2.0,
    "ipopt": 0.3,
    "aligator": 2.25,
    "mim_solvers": 0.85,
}


def estimate_memory_bytes(problem, solver, duffing_m, duffing_horizon):
    if problem not in DUFFING_PROBLEMS:
        return NON_DUFFING_MEMORY_ESTIMATE_BYTES
    m = duffing_m if duffing_m is not None else DUFFING_DEFAULT_M
    horizon = (
        duffing_horizon
        if duffing_horizon is not None
        else DUFFING_DEFAULT_HORIZON
    )
    nx, nu = 2 * m, m
    base_estimate = (
        DUFFING_MEM_BASE_BYTES
        + DUFFING_MEM_BYTES_PER_ELEMENT * (nx + nu) ** 2 * horizon
    )
    return int(base_estimate * DUFFING_SOLVER_MEM_MULTIPLIER[solver])


def build_run_command(problem, solver, tolerance, duffing_m, duffing_horizon):
    """Re-invoke this script for a single problem/solver/tolerance run.

    Used to run each item of a sweep in its own subprocess, so a leaked fd
    (e.g. CasADi's JIT compiler leaking one per compiled function, see
    --jit) can't accumulate across a long sweep and exhaust the open-file
    limit.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        problem,
        "-s",
        solver,
        "-t",
        str(tolerance),
    ]
    if inverse_dynamics:
        cmd += ["--dynamics-mode", "inverse"]
    if args.verbose:
        cmd.append("--verbose")
    if args.jit:
        cmd.append("--jit")
    if args.jit_opt_level != "O3":
        cmd += ["--jit-opt-level", args.jit_opt_level]
    if args.cold_start:
        cmd.append("--cold-start")
    if duffing_m is not None:
        cmd += ["--duffing-m", str(duffing_m)]
    if duffing_horizon is not None:
        cmd += ["--duffing-horizon", str(duffing_horizon)]
    return cmd


# Tracks how much of --memory-budget-fraction is currently reserved by
# still-running runs (see estimate_memory_bytes()), so a new run's
# admission decision accounts for runs that haven't shown up in the OS's
# own memory accounting yet, not just runs whose usage is already visible
MEMORY_BUDGET_BYTES = int(total_memory_bytes() * args.memory_budget_fraction)
_memory_cv = threading.Condition()
_reserved_bytes = 0


def _reserve_memory(label, amount):
    global _reserved_bytes
    with _memory_cv:
        if (
            _reserved_bytes > 0
            and _reserved_bytes + amount > MEMORY_BUDGET_BYTES
        ):
            print(
                f"{label}: waiting for memory budget "
                f"({_reserved_bytes / 1024**3:.1f} GiB already reserved, "
                f"needs {amount / 1024**3:.1f} GiB, budget is "
                f"{MEMORY_BUDGET_BYTES / 1024**3:.1f} GiB)..."
            )
        # Only block on other runs' reservations, not on this one alone:
        # a single run whose own estimate exceeds the whole budget still
        # has to run eventually, so let it through once nothing else is
        # holding a reservation instead of deadlocking forever
        while (
            _reserved_bytes > 0
            and _reserved_bytes + amount > MEMORY_BUDGET_BYTES
        ):
            _memory_cv.wait()
        _reserved_bytes += amount


def _release_memory(amount):
    global _reserved_bytes
    with _memory_cv:
        _reserved_bytes -= amount
        _memory_cv.notify_all()


def run_in_subprocess(item, attempt):
    label, banner, cmd, estimate = item
    retry_note = f" (retry {attempt})" if attempt else ""
    parallel = args.jobs > 1
    _reserve_memory(label, estimate)
    try:
        if parallel:
            # Captured and printed as one block below instead of streamed
            # live, since interleaved live output from several jobs isn't
            # legible anyway
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        else:
            print(banner + retry_note)
            result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    finally:
        _release_memory(estimate)

    if result.returncode == 0:
        if parallel:
            print(f"{banner}{retry_note}\n{result.stdout}", end="")
        return "ok", attempt

    stderr_lines = result.stderr.strip().splitlines()
    err = (
        stderr_lines[-1] if stderr_lines else f"exit code {result.returncode}"
    )
    if parallel:
        print(f"{banner}{retry_note}\n{result.stdout}", end="")
    # -9 is SIGKILL: our own estimate under-predicted and the OS's OOM
    # killer stepped in
    out_of_memory = result.returncode == -9
    if out_of_memory:
        print(f"{label} ran out of memory{retry_note}: {err}")
        return "memory", attempt
    print(f"{label} failed{retry_note}: {err}")
    return "failed", attempt


def run_items(items):
    """Run every item in its own subprocess, up to --jobs at once.

    An out-of-memory failure (see run_in_subprocess) is re-queued instead
    of being reported right away, up to --max-memory-retries times.
    """
    failures = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_in_subprocess, item, 0): item for item in items
        }
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                status, attempt = future.result()
                if status == "ok":
                    continue
                if status == "memory" and attempt < args.max_memory_retries:
                    futures[
                        pool.submit(run_in_subprocess, item, attempt + 1)
                    ] = item
                else:
                    failures.append(item[0])
    return failures


for tolerance in tolerances_to_run:
    if len(tolerances_to_run) > 1:
        print(f"##### tolerance = {tolerance:g} #####")

    if not grid_search and len(problems_to_run) == 1 and len(solvers) == 1:
        problem, solver = problems_to_run[0], solvers[0]
        if args.skip_existing and os.path.exists(
            expected_output_path(
                problem,
                solver,
                tolerance,
                duffing_m_scalar,
                duffing_horizon_scalar,
            )
        ):
            print(f"Skipping {problem}/{solver}: output file already exists")
        else:
            run_solver(problem, solver)
    else:
        items = []
        skipped = 0
        for problem in problems_to_run:
            if grid_search and problem in DUFFING_PROBLEMS:
                for m in duffing_m_list:
                    for horizon in duffing_horizon_list:
                        # Only the display text needs the resolved default:
                        # expected_output_path/build_run_command/
                        # estimate_memory_bytes already resolve None
                        # themselves
                        display_m = m if m is not None else DUFFING_DEFAULT_M
                        display_horizon = (
                            horizon
                            if horizon is not None
                            else DUFFING_DEFAULT_HORIZON
                        )
                        for solver in solvers:
                            label = (
                                f"{problem}/{solver}/"
                                f"m={display_m}_N={display_horizon}"
                            )
                            if args.skip_existing and os.path.exists(
                                expected_output_path(
                                    problem, solver, tolerance, m, horizon
                                )
                            ):
                                print(
                                    f"Skipping {label}: output file already exists"
                                )
                                skipped += 1
                                continue
                            banner = (
                                f"=== {problem} / {solver} / m={display_m}, "
                                f"horizon={display_horizon} ==="
                            )
                            cmd = build_run_command(
                                problem, solver, tolerance, m, horizon
                            )
                            estimate = estimate_memory_bytes(
                                problem, solver, m, horizon
                            )
                            items.append((label, banner, cmd, estimate))
            else:
                for solver in solvers:
                    label = f"{problem}/{solver}"
                    if args.skip_existing and os.path.exists(
                        expected_output_path(
                            problem,
                            solver,
                            tolerance,
                            duffing_m_scalar,
                            duffing_horizon_scalar,
                        )
                    ):
                        print(f"Skipping {label}: output file already exists")
                        skipped += 1
                        continue
                    banner = f"=== {problem} / {solver} ==="
                    cmd = build_run_command(
                        problem,
                        solver,
                        tolerance,
                        duffing_m_scalar,
                        duffing_horizon_scalar,
                    )
                    estimate = estimate_memory_bytes(
                        problem,
                        solver,
                        duffing_m_scalar,
                        duffing_horizon_scalar,
                    )
                    items.append((label, banner, cmd, estimate))

        failures = run_items(items)

        total = len(items)
        print()
        print(
            f"{total - len(failures)}/{total} runs terminated without crashing."
        )
        if skipped:
            print(f"{skipped} run(s) skipped (output file already exists).")
        if failures:
            print(f"Failed: {', '.join(failures)}")
