import argparse
import os

from benchmark_config import TOLERANCE
from problems import *


def parse_args():
    parser = argparse.ArgumentParser(description="Solver benchmark script")

    parser.add_argument(
        "problem",
        choices=[
            "pendubot",
            "go2_standing",
            "go2_jump",
            "go2_frog",
            "panda_circle",
        ],
        help="The benchmarking problem to run",
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
        help="The solver to benchmark. Ignored if --all is given.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run every solver on the given problem, instead of just --solver.",
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
inverse_dynamics = args.dynamics_mode == "inverse"

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

if args.all:
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
    "go2_frog": 1000,
    "panda_circle": 1000,
}
max_iter = MAX_ITER_BY_PROBLEM[args.problem]

problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
    "go2_jump": make_go2_jump_problem,
    "go2_frog": make_go2_frog_problem,
    "panda_circle": make_panda_circle_problem,
}


def run_solver(solver):
    problem_wrapper_t = None
    constructor_parameters = {}
    solve_parameters = {}
    if solver == "ssqp-admm" or solver == "ssqp-hpipm":
        import ssqpy
        from ssqp_formulations.SSQPProblem import SSQPProblem

        problem_wrapper_t = SSQPProblem

        is_floating_base = args.problem in (
            "go2_standing",
            "go2_jump",
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
        if args.problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif is_floating_base:
            constructor_parameters["free_flyer"] = True

        solve_parameters["tolerance"] = TOLERANCE
        solve_parameters["max_iter"] = max_iter

        if solver == "ssqp-admm":
            solve_parameters["qp_solver"] = "admm"
            # panda_circle's per-stage QP needs a larger ADMM budget than
            # pendubot/go2's to converge within the outer SQP loop
            solve_parameters["admm_max_iter"] = (
                400
                if args.problem == "panda_circle"
                else (300 if args.problem == "pendubot" else 100)
            )
            solve_parameters["admm_tol"] = 1e-3
            # rho_update_iter_freq also gates ADMM's own termination check (see
            # ssqp's admm_qp.hpp), so it must be lower than admm_max_iter for
            # ADMM to ever detect convergence and exit early
            solve_parameters["admm_adaptive_rho_tolerance"] = 5
            solve_parameters["admm_rho_update_iter_freq"] = 25
            solve_parameters["admm_warm_start"] = True
        else:
            solve_parameters["qp_solver"] = "hpipm"
            solve_parameters["hpipm_tol"] = (
                1e-2 if args.problem == "pendubot" else 1e-3
            )

    elif solver == "mim_solvers":
        from mim_solvers_formulations.MimSolversProblem import (
            MimSolversProblem,
            PendubotActuationModel,
        )
        import pinocchio as pin
        import crocoddyl

        problem_wrapper_t = MimSolversProblem

        if args.problem == "pendubot":
            pin_model = pin.buildModelFromUrdf(
                "urdf/double_pendulum/pendubot.urdf"
            )
            state = crocoddyl.StateMultibody(pin_model)
            custom_actuation = PendubotActuationModel(state, pin_model)

            constructor_parameters["floating_base"] = False
            constructor_parameters["actuation_model"] = custom_actuation
        elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
            constructor_parameters["floating_base"] = True
        elif args.problem == "panda_circle":
            # Explicit: adding a free-flyer joint on top of a fixed-base URDF
            # like panda.urdf doesn't raise, so MimSolversProblem's own
            # try/except fallback to fixed-base would never trigger here
            constructor_parameters["floating_base"] = False

        solve_parameters["tolerance"] = TOLERANCE
        solve_parameters["max_iter"] = max_iter
        # CSQP's convergence check only runs every rho_update_interval QP
        # iterations (csqp.cpp), so it must stay below max_qp_iters for early
        # exit to ever trigger, and a lower rho_update_interval destabilizes
        # the floating-base/contact-rich problems, so max_qp_iters is raised
        # instead of lowering it
        solve_parameters["max_qp_iters"] = 300
        solve_parameters["eps_abs"] = 1e-3
        solve_parameters["eps_rel"] = 1e-3
        solve_parameters["adaptive_rho_tolerance"] = 1
        solve_parameters["rho_update_interval"] = 100

    elif solver == "aligator":
        from aligator_formulations.AligatorProblem import AligatorProblem

        problem_wrapper_t = AligatorProblem

        if args.problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
            constructor_parameters["free_flyer"] = True

        solve_parameters["tol"] = TOLERANCE
        solve_parameters["mu_init"] = 1e-2
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
        if args.problem == "pendubot":
            constructor_parameters["actuated_joints"] = [0]
        elif args.problem in ("go2_standing", "go2_jump", "go2_frog"):
            constructor_parameters["free_flyer"] = True

        # Keep backend diagnostics out of the timed benchmark and console
        # output for both plugins: print_time is CasADi's own end-of-solve
        # timing breakdown, unused since the wrapper measures the complete
        # opti.solve() call consistently for both, and print_level silences
        # the per-iteration progress log those solvers print by default,
        # which is synchronous console I/O happening *inside* the timed
        # opti.solve() call otherwise, and every other backend is already
        # quiet by default (see MimSolversProblem/AligatorProblem), so
        # leaving these on would only slow down and inflate fatrop/ipopt's
        # measured runtime
        solve_parameters["print_time"] = False

        if solver == "fatrop":
            # nx/nu/ng/N + structure_detection="manual" are filled in by
            # CasadiProblem.solve() once it knows the per-stage dimensions
            solve_parameters["fatrop"] = {
                "tol": TOLERANCE,
                "max_iter": max_iter,
                "print_level": 0,
            }
            if args.problem in ("go2_standing", "go2_jump"):
                # Avoids an unrecoverable restoration phase from the
                # zero-velocity/static initial guess against go2's many
                # contact/velocity constraints
                solve_parameters["fatrop"]["mu_init"] = 1.0
        else:
            solve_parameters["ipopt"] = {
                "max_iter": max_iter,
                "tol": TOLERANCE,
                "acceptable_tol": 1e-3,
                "mu_strategy": "adaptive",
                "print_level": 0,
                "sb": "yes",  # suppresses the startup banner too
            }
            if args.problem == "go2_jump":
                # Same restoration-phase issue as Fatrop above
                solve_parameters["ipopt"]["mu_init"] = 1.0

    problem_wrapper = problems[args.problem](
        problem_wrapper_t, constructor_parameters
    )

    problem_wrapper.solve(solve_parameters)

    save_path = f"data/{solver}/"
    save_path = save_path.replace("-", "_")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    output_name = args.problem
    if inverse_dynamics:
        output_name += "_inverse_dynamics"

    problem_wrapper.save(save_path + output_name + ".json")


if len(solvers) == 1:
    run_solver(solvers[0])
else:
    failures = []
    for solver in solvers:
        print(f"=== {solver} ===")
        try:
            run_solver(solver)
        except Exception as exc:
            print(f"{solver} failed: {exc}")
            failures.append(solver)

    print()
    print(f"{len(solvers) - len(failures)}/{len(solvers)} solvers succeeded.")
    if failures:
        print(f"Failed: {', '.join(failures)}")
