import argparse
import os

from benchmark_config import TOLERANCE, tolerance_label
from problems import *


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
        ],
        help="The benchmarking problem to run. Ignored if --all-problems is given.",
    )

    parser.add_argument(
        "-s",
        "--solver",
        nargs="?",
        choices=[
            "ssqp-admm",
            "ssqp-hpipm",
            "fatrop",
            "ipopt",
            "aligator",
            "mim_solvers",
        ],
        default="ssqp-admm",
        help="The solver to benchmark. Ignored if --all-solvers is given.",
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
        default=TOLERANCE,
        help="Convergence tolerance every solver is run with.",
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

    return parser.parse_args()


args = parse_args()
if not args.all_problems and args.problem is None:
    raise SystemExit("Either a problem or --all-problems must be given.")
inverse_dynamics = args.dynamics_mode == "inverse"
tolerance = args.tolerance

ALL_PROBLEMS = [
    "pendubot",
    "go2_standing",
    "go2_jump",
    "go2_wheelie",
    "go2_frog",
    "panda_circle",
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
    if inverse_dynamics and args.solver in INVERSE_UNSUPPORTED:
        raise SystemExit(
            "--dynamics-mode inverse is not supported for "
            f"--solver {args.solver} (see --help for why)"
        )
    solvers = [args.solver]

# Outer-iteration caps, sized per problem from observed convergence
# behavior rather than one flat number for everything
MAX_ITER_BY_PROBLEM = {
    "pendubot": 500,
    "go2_standing": 200,
    "go2_jump": 1000,
    "go2_wheelie": 1000,
    "go2_frog": 1000,
    "panda_circle": 1000,
}
problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
    "go2_jump": make_go2_jump_problem,
    "go2_wheelie": make_go2_wheelie_problem,
    "go2_frog": make_go2_frog_problem,
    "panda_circle": make_panda_circle_problem,
}

# Per-(problem, solver) solve_parameters overrides, applied right before
# solve() on top of that solver's generic defaults above -- the generic
# defaults are shared across problems for a fair comparison, but some
# (problem, solver) pairs need something else, and a value may be a
# callable(tolerance) instead of a constant when it needs to scale with
# -t/--tolerance the way the generic default does. A (problem, solver) pair
# with no entry here behaves exactly as it did before this dict existed
PROBLEM_SOLVER_OVERRIDES = {
    ("pendubot", "ssqp-hpipm"): {
        # The generic hpipm_tol = tolerance / 10 leaves pendubot's inner
        # HPIPM solve tighter than the outer SQP tolerance ever demands,
        # and it runs to max_iter without converging -- matching hpipm_tol
        # to the outer tolerance fixes it, and square_root_riccati=False
        # shaves further time off on top without changing the iteration
        # count
        "hpipm_tol": lambda tolerance: tolerance,
        "hpipm_square_root_riccati": False,
    },
    ("go2_jump", "ssqp-hpipm"): {
        # These four flags together roughly triple go2_jump's convergence
        # speed over the shared defaults, but split_step and warm_start
        # individually break go2_frog's convergence entirely, so the combo
        # is scoped to go2_jump rather than applied generically
        "hpipm_split_step": True,
        "hpipm_prediction_correction": True,
        "hpipm_warm_start": True,
        "hpipm_square_root_riccati": False,
    },
    ("go2_wheelie", "ssqp-hpipm"): {
        # split_step is the one flag that helps go2_wheelie -- unlike
        # go2_jump, pairing it with square_root_riccati=False is markedly
        # slower here, so that one stays at its default
        "hpipm_split_step": True,
    },
    ("pendubot", "ssqp-admm"): {
        # alpha=1.8 converges faster than the default 1.6, with
        # reset_rho/reset_y left at their defaults
        "admm_alpha": 1.8,
    },
    ("go2_jump", "ssqp-admm"): {
        # The default alpha=1.6 never converges for go2_jump within the
        # iteration budget -- alpha=1.8 is what actually makes it
        # converge, not just a speed improvement
        "admm_alpha": 1.8,
    },
    ("go2_wheelie", "ssqp-admm"): {
        # The default alpha=1.6 also never converges for go2_wheelie, and
        # unlike pendubot/go2_jump, alpha=1.8 alone still converges slowly
        # here -- turning off both resets converges faster and more
        # reliably
        "admm_reset_rho": False,
        "admm_reset_y": False,
        "admm_alpha": 1.0,
    },
    ("pendubot", "mim_solvers"): {
        # Turning off all three converges faster here, trading more
        # iterations for less wall-clock time per iteration -- contact-
        # switching problems (go2_jump/go2_wheelie/go2_frog) are excluded
        # since mim_solvers doesn't yet handle those regardless of these
        # settings
        "reset_rho": False,
        "reset_y": False,
        "update_rho_with_heuristic": False,
    },
    ("go2_standing", "mim_solvers"): {
        # Same combo as pendubot above, same reasoning
        "reset_rho": False,
        "reset_y": False,
        "update_rho_with_heuristic": False,
    },
    ("panda_circle", "mim_solvers"): {
        # The defaults never converge at all for panda_circle -- this
        # combo is what fixes it
        "reset_rho": False,
        "reset_y": False,
        "update_rho_with_heuristic": False,
    },
    ("go2_jump", "ipopt"): {
        # Ipopt's default exact Hessian never converges for go2_jump --
        # the BFGS approximation does, and pendubot/go2_standing converge
        # faster with the exact Hessian left alone, so this stays scoped
        # here rather than applied generically
        "ipopt": {"hessian_approximation": "limited-memory"},
    },
    ("go2_wheelie", "ipopt"): {
        # Same as go2_jump above, same reasoning
        "ipopt": {"hessian_approximation": "limited-memory"},
    },
    ("go2_standing", "fatrop"): {
        # mu_init=0.1 converges faster here than the mu_init=1.0 already
        # set above for the restoration-phase issue, without reintroducing
        # it
        "fatrop": {"mu_init": 0.1},
    },
    ("panda_circle", "fatrop"): {
        # mu_init=1.0 roughly halves panda_circle's solve time over
        # fatrop's own default
        "fatrop": {"mu_init": 1.0},
    },
}


def run_solver(problem, solver):
    max_iter = MAX_ITER_BY_PROBLEM[problem]
    problem_wrapper_t = None
    constructor_parameters = {}
    solve_parameters = {}
    if solver == "ssqp-admm" or solver == "ssqp-hpipm":
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
            # Plain InverseDynamics enforces an unactuated floating base via
            # explicit zero-torque equality constraints; ReducedInverseDynamics
            # instead bakes that in implicitly through state propagation, which
            # is the better-conditioned choice whenever a floating base is
            # present (see SolverMode's docstring in include/types.hpp)
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
        # CSQP's convergence check only runs every rho_update_interval QP
        # iterations (csqp.cpp), so it must stay below max_qp_iters for early
        # exit to ever trigger, and a lower rho_update_interval destabilizes
        # the floating-base/contact-rich problems, so max_qp_iters is raised
        # instead of lowering it
        solve_parameters["max_qp_iters"] = 300
        solve_parameters["eps_abs"] = tolerance / 10
        solve_parameters["eps_rel"] = tolerance / 10
        solve_parameters["adaptive_rho_tolerance"] = 1
        solve_parameters["rho_update_interval"] = 100
        # CallbackVerbose/CallbackLogger add per-iteration logging and
        # bookkeeping to the timed solve, so only enable them when asked
        solve_parameters["callbacks"] = args.verbose

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
        # NOTE: go2_jump doesn't converge here, since the solver can't find
        # a feasible direction from the static initial guess against the
        # contact-switching + terminal foot-placement constraints, and
        # likely needs a kinematically-sensible initial guess rather than
        # mu_init tuning (several values tried, none converge)
        solve_parameters["max_iters"] = max_iter

    elif solver == "fatrop" or solver == "ipopt":
        from casadi_formulations.CasadiProblem import CasadiProblem

        problem_wrapper_t = CasadiProblem

        constructor_parameters["casadi_plugin"] = solver
        constructor_parameters["jit"] = args.jit
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

        # Keep backend diagnostics out of the timed benchmark and console
        # output for both plugins by default: print_time is CasADi's own
        # end-of-solve timing breakdown, unused since the wrapper measures
        # the complete opti.solve() call consistently for both, and
        # print_level silences the per-iteration progress log those solvers
        # print by default, which is synchronous console I/O happening
        # *inside* the timed opti.solve() call otherwise, and every other
        # backend is already quiet by default (see
        # MimSolversProblem/AligatorProblem), so leaving these on would only
        # slow down and inflate fatrop/ipopt's measured runtime; --verbose
        # opts back into them for interactive troubleshooting
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
                # Avoids an unrecoverable restoration phase from the
                # zero-velocity/static initial guess against go2's many
                # contact/velocity constraints
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

    for key, val in PROBLEM_SOLVER_OVERRIDES.get((problem, solver), {}).items():
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
    output_name += f"_tol_{tolerance_label(tolerance)}"

    problem_wrapper.save(save_path + output_name + ".json")


if len(problems_to_run) == 1 and len(solvers) == 1:
    run_solver(problems_to_run[0], solvers[0])
else:
    failures = []
    for problem in problems_to_run:
        for solver in solvers:
            print(f"=== {problem} / {solver} ===")
            try:
                run_solver(problem, solver)
            except Exception as exc:
                print(f"{problem}/{solver} failed: {exc}")
                failures.append(f"{problem}/{solver}")

    total = len(problems_to_run) * len(solvers)
    print()
    print(f"{total - len(failures)}/{total} runs terminated without crashing.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
