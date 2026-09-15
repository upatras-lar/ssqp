import argparse
import multiprocessing
import json
from os import remove
from os.path import isfile
import ssqpy
import problems

ssqpy.setSilentMode()

MODE_TO_SOLVER_MODE = {
    "forward": ssqpy.model.SolverMode.ForwardDynamics,
    "inverse": ssqpy.model.SolverMode.InverseDynamics,
    "reduced_inverse": ssqpy.model.SolverMode.ReducedInverseDynamics,
}

QP_SOLVER_TYPES = {
    "admm": ssqpy.solvers.QPSolverType.ADMM,
    "hpipm": ssqpy.solvers.QPSolverType.HPIPM,
}

MODEL_MODES = {
    "pendubot": ["forward", "inverse"],
    "panda": ["forward", "inverse"],
    "go2": ["forward", "inverse", "reduced_inverse"],
    "go2_multi_jump": ["forward", "inverse", "reduced_inverse"],
    "g1_jump": ["forward", "inverse", "reduced_inverse"],
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmarking script for trajectory generation solver modes"
    )

    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODEL_MODES.keys()),
        help="The model to benchmark; omit if using --all-problems",
    )

    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Benchmark every available model instead of a single one",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm"],
        default="admm",
        help="The QP backend to use",
    )

    parser.add_argument(
        "-n",
        "--n-problems",
        type=int,
        default=50,
        help="Number of problems to generate (pendubot/panda/go2 only)",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Horizon to use (pendubot only)",
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["forward", "inverse", "reduced_inverse"],
        default=None,
        help=(
            "Subset of dynamics modes to run; defaults to every mode each "
            "benchmarked model supports"
        ),
    )

    args = parser.parse_args()

    if args.all_problems == (args.model is not None):
        parser.error(
            "pass exactly one of a model or --all-problems, not both/neither"
        )

    return args


def get_problems(model, dynamics_mode, n_problems, horizon):
    if model == "pendubot":
        return problems.generate_pendubot_problems(
            n_problems, horizon, dynamics_mode
        )

    if model == "panda":
        return problems.generate_panda_feasible_region_problems(
            n_problems, dynamics_mode
        )

    if model == "go2":
        return problems.generate_go2_walk_problems(n_problems, dynamics_mode)

    # go2_multi_jump/g1_jump are single, fixed choreographed problems, so
    # n_problems/horizon don't apply
    if model == "go2_multi_jump":
        return problems.generate_go2_multi_jump_problems(dynamics_mode)

    return problems.generate_g1_jump_problems(dynamics_mode)


def default_params(qp_solver_type, max_iter=1000):
    """
    Build default solver parameters for the given QP backend, altering
    tolerance and max iterations to get a full run
    """
    sqp_params = ssqpy.solvers.ssqpParams()
    sqp_params.qp_solver_type = qp_solver_type
    sqp_params.tolerance = 1e-3
    sqp_params.max_iter = max_iter

    if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM:
        sqp_params.admmParams.abs_tolerance = 1e-4
        sqp_params.admmParams.rel_tolerance = 1e-4

    return sqp_params


def get_params(model, qp_solver_type):
    """
    Build solver parameters for the given model/backend combination. The
    same params are reused unchanged across every dynamics mode that model
    is benchmarked with
    """
    if model == "pendubot":
        max_iter = (
            2000 if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM else 1000
        )
        return default_params(qp_solver_type, max_iter=max_iter)

    if model == "panda":
        max_iter = (
            200 if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM else 1000
        )
        return default_params(qp_solver_type, max_iter=max_iter)

    if model == "go2":
        params = default_params(qp_solver_type, max_iter=100)
        params.tolerance = 1e-2
        params.accept_worse_update = True

        if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM:
            params.admmParams.abs_tolerance = 1e-3
            params.admmParams.rel_tolerance = 1e-3
            params.admmParams.max_iter = 500
            params.admmParams.adaptive_rho_tolerance = 10.0
        else:
            params.hpipmParams.tol_comp = 1e-3
            params.hpipmParams.tol_stat = 1e-3
            params.hpipmParams.tol_eq = 1e-3
            params.hpipmParams.tol_ineq = 1e-3
            params.hpipmParams.prediction_correction = True
            params.hpipmParams.split_step = True

        return params

    if model == "go2_multi_jump":
        params = default_params(qp_solver_type)
        params.tolerance = 1e-2
        params.loose_success = False

        if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM:
            params.admmParams.abs_tolerance = 1e-3
            params.admmParams.rel_tolerance = 1e-3
            params.admmParams.max_iter = 50
            params.admmParams.reset_rho = False
            params.admmParams.reset_y = False
            params.admmParams.warm_start = False
        else:
            params.hpipmParams.tol_stat = 1e-3
            params.hpipmParams.tol_comp = 1e-3
            params.hpipmParams.tol_eq = 1e-3
            params.hpipmParams.tol_ineq = 1e-3

        return params

    # g1_jump
    params = default_params(qp_solver_type)
    params.tolerance = 1e-2

    if qp_solver_type == ssqpy.solvers.QPSolverType.ADMM:
        params.admmParams.max_iter = 100
        params.admmParams.abs_tolerance = 1e-2
        params.admmParams.rel_tolerance = 1e-2
        params.admmParams.reset_rho = False
        params.admmParams.reset_y = False
        params.admmParams.warm_start = False
    else:
        params.hpipmParams.max_iter = 20
        params.hpipmParams.square_root_riccati = True
        params.hpipmParams.tol_stat = 1e-3
        params.hpipmParams.tol_comp = 1e-3
        params.hpipmParams.tol_eq = 1e-3
        params.hpipmParams.tol_ineq = 1e-3

    return params


def benchmark(probs, sqp_params, save_to):
    """
    Benchmark given problems using the provided parameters and save all
    results to provided filename
    """
    results = []

    pid = multiprocessing.current_process().pid

    tmp_filepath = f"data/tmp{pid}.json"

    for i, problem in enumerate(probs):
        solver = ssqpy.solvers.ssqp(problem, sqp_params)

        try:
            solver.solve()
        except RuntimeError as e:
            results.append({"failed": True, "error": str(e)})
            print(
                f"Process {pid} progress: {(i + 1) / len(probs) * 100}% "
                f"(problem {i} failed: {e})"
            )
            continue

        solver.write(tmp_filepath)

        with open(tmp_filepath, "r") as f:
            data = json.load(f)

        results.append(data)

        print(f"Process {pid} progress: {(i + 1) / len(probs) * 100}%")

    if isfile(tmp_filepath):
        remove(tmp_filepath)

    with open(save_to, "w") as f:
        f.write(json.dumps(results, indent=4))


def run(model, mode, qp_solver, n_problems, horizon):
    dynamics_mode = MODE_TO_SOLVER_MODE[mode]
    qp_solver_type = QP_SOLVER_TYPES[qp_solver]

    probs = get_problems(model, dynamics_mode, n_problems, horizon)
    params = get_params(model, qp_solver_type)

    benchmark(
        probs,
        params,
        f"data/results/{model}_{qp_solver}_{mode}.json",
    )


args = parse_args()

models = list(MODEL_MODES.keys()) if args.all_problems else [args.model]

processes = [
    multiprocessing.Process(
        target=run,
        args=(model, mode, args.qp_solver, args.n_problems, args.horizon),
    )
    for model in models
    for mode in (args.modes or MODEL_MODES[model])
    if mode in MODEL_MODES[model]
]

for p in processes:
    p.start()

for p in processes:
    p.join()
