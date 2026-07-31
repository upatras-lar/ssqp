import argparse
from problems import run_pendubot_problem, run_go2_standing_problem


def parse_args():
    parser = argparse.ArgumentParser(description="Pendubot controller server")

    parser.add_argument("problem", choices=["pendubot", "go2_standing"])

    parser.add_argument(
        "-s",
        "--solver",
        nargs="?",
        choices=["fatrop", "ipopt"],
        default="fatrop",
        help="The solver to use",
    )

    return parser.parse_args()


problems = {
    "pendubot": run_pendubot_problem,
    "go2_standing": run_go2_standing_problem,
}

args = parse_args()

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
