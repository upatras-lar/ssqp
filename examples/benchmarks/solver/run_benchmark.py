import argparse
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Solver benchmark script")

    parser.add_argument(
        "problem",
        choices=["pendubot", "go2_standing"],
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


def make_pendubot_problem(problem_wrapper_t, constructor_parameters):
    problem_wrapper = problem_wrapper_t(
        2500,
        0.001,
        "urdf/double_pendulum/pendubot.urdf",
        constructor_parameters,
    )

    V_WGT = 1e-4
    U_WGT = 1e-4
    C_WGT = 1.0

    DESIRED_CONFIG = np.array((np.pi, 0.0))

    for k in range(problem_wrapper.H):
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)

        if k > 4 * problem_wrapper.H // 5:
            problem_wrapper.addSquaredConfigurationErrorCost(
                k, DESIRED_CONFIG, C_WGT
            )

    problem_wrapper.addSquaredConfigurationErrorCost(
        problem_wrapper.H, DESIRED_CONFIG, C_WGT
    )

    return problem_wrapper


def make_go2_standing_problem(problem_wrapper_t, constructor_parameters):
    problem_wrapper = problem_wrapper_t(
        100, 0.01, "urdf/go2/go2.urdf", constructor_parameters
    )

    A_WGT = 10.0
    V_WGT = 1e-2
    U_WGT = 1e-4
    F_WGT = 1e-5

    MAX_NORMAL_FORCE = 50
    FOOT_EQ_TOL = 0.01

    # Initial and final positions
    base_pos = np.array([0.0, 0.0, 0.3879793820122])
    base_config = np.array([0.0, np.pi / 6, -np.pi / 3] * 4)

    # Initial configuration
    init = np.zeros(37)
    init[:3] = base_pos
    init[6] = 1.0
    init[7:19] = base_config

    for k in range(problem_wrapper.H):
        problem_wrapper.addSquaredConfigurationErrorCost(k, base_config, A_WGT)
        problem_wrapper.addSquaredJointVelocityCost(k, V_WGT)
        problem_wrapper.addSquaredControlCost(k, U_WGT)
        problem_wrapper.addSquaredForceCost(k, F_WGT)

        problem_wrapper.addContactConstraint(
            k, "FL_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "FL_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "FR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "FR_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "RL_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "RL_foot", np.zeros(3), np.zeros(3)
        )

        problem_wrapper.addContactConstraint(
            k, "RR_foot", MAX_NORMAL_FORCE, FOOT_EQ_TOL
        )
        problem_wrapper.addFrameLinearVelocityBound(
            k, "RR_foot", np.zeros(3), np.zeros(3)
        )

    problem_wrapper.addSquaredConfigurationErrorCost(
        problem_wrapper.H, base_config, A_WGT
    )
    problem_wrapper.addFrameLinearVelocityBound(
        problem_wrapper.H, "base", np.zeros(3), np.zeros(3)
    )

    problem_wrapper.finalize({"init_states": init})

    return problem_wrapper


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
    elif args.problem == "go2_standing":
        constructor_parameters["free_flyer"] = True

    solve_parameters["tolerance"] = 1e-2

    if args.solver == "ssqp-admm":
        solve_parameters["qp_solver"] = "admm"
        solve_parameters["admm_max_iter"] = 100
        solve_parameters["admm_tol"] = 1e-3
        # solve_parameters["admm_adaptive_rho_tolerance"] = 1
        # solve_parameters["admm_rho_update_iter_freq"] = 25
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
    elif args.problem == "go2_standing":
        constructor_parameters["floating_base"] = True

problems = {
    "pendubot": make_pendubot_problem,
    "go2_standing": make_go2_standing_problem,
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

"""
if args.solver == "fatrop":
    problems[args.problem](
        "fatrop",
        lambda nx, nu, ng, H: {
            "structure_detection": "manual",
            "nx": nx,
            "nu": nu,
            "ng": ng,
            "N": H - 1,
            "expand": True,
        },
        f"{args.problem}_fatrop.json",
    )
else:
    problems[args.problem](
        "ipopt",
        lambda nx, nu, ng, H: {
            "ipopt.max_iter": 1000,
            "ipopt.tol": 1e-2,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.mu_strategy": "adaptive",
            "print_time": True,
        },
        f"{args.problem}_ipopt.json",
    )
"""
