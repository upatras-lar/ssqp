import argparse
import os

from problems import *


def parse_args():
    parser = argparse.ArgumentParser(description="Solver benchmark script")

    parser.add_argument(
        "problem",
        choices=["pendubot", "go2_standing", "go2_jump"],
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
        help="The solver to benchmark",
    )

    return parser.parse_args()


args = parse_args()

problem_wrapper_t = None
constructor_parameters = {}
solve_parameters = {}
if args.solver == "ssqp-admm" or args.solver == "ssqp-hpipm":
    import ssqpy
    from ssqp_formulations.SSQPProblem import SSQPProblem

    problem_wrapper_t = SSQPProblem

    constructor_parameters["solver_mode"] = (
        ssqpy.model.SolverMode.ForwardDynamics
    )
    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump"):
        constructor_parameters["free_flyer"] = True

    solve_parameters["tolerance"] = 1e-2

    if args.solver == "ssqp-admm":
        solve_parameters["qp_solver"] = "admm"
        solve_parameters["admm_max_iter"] = 100
        solve_parameters["admm_tol"] = 1e-3
        solve_parameters["admm_warm_start"] = True
    else:
        solve_parameters["qp_solver"] = "hpipm"
        solve_parameters["hpipm_tol"] = 1e-3

elif args.solver == "mim_solvers":
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
    elif args.problem in ("go2_standing", "go2_jump"):
        constructor_parameters["floating_base"] = True

    solve_parameters["tolerance"] = 1e-2
    solve_parameters["max_qp_iters"] = 100
    solve_parameters["eps_abs"] = 1e-3
    solve_parameters["eps_rel"] = 1e-3
    solve_parameters["adaptive_rho_tolerance"] = 1
    solve_parameters["rho_update_interval"] = 100

elif args.solver == "aligator":
    from aligator_formulations.AligatorProblem import AligatorProblem

    problem_wrapper_t = AligatorProblem

    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump"):
        constructor_parameters["free_flyer"] = True

    solve_parameters["tol"] = 1e-2
    solve_parameters["mu_init"] = 1e-2
    solve_parameters["max_iters"] = 1000

elif args.solver == "fatrop" or args.solver == "ipopt":
    from casadi_formulations.CasadiProblem import CasadiProblem

    problem_wrapper_t = CasadiProblem

    constructor_parameters["casadi_plugin"] = args.solver
    if args.problem == "pendubot":
        constructor_parameters["actuated_joints"] = [0]
    elif args.problem in ("go2_standing", "go2_jump"):
        constructor_parameters["free_flyer"] = True

    if args.solver == "fatrop":
        # nx/nu/ng/N + structure_detection="manual" are filled in
        # automatically by CasadiProblem.solve() (it's the only thing that
        # knows the per-stage dimensions once the problem is built), so we
        # only need to pass Fatrop's own numerical tolerances here.
        solve_parameters["fatrop"] = {"tol": 1e-2}
    else:
        solve_parameters["ipopt"] = {
            "max_iter": 1000,
            "tol": 1e-2,
            "acceptable_tol": 1e-3,
            "mu_strategy": "adaptive",
        }
        solve_parameters["print_time"] = True

problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
    "go2_jump": make_go2_jump_problem,
}


problem_wrapper = problems[args.problem](
    problem_wrapper_t, constructor_parameters
)

problem_wrapper.solve(solve_parameters)

save_path = f"data/{args.solver}/"
save_path = save_path.replace("-", "_")

if not os.path.exists(save_path):
    os.makedirs(save_path)

problem_wrapper.save(save_path + args.problem + ".json")
