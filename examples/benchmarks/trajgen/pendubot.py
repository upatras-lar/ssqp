import ssqpy
import sys

sys.path.insert(0, "..")

import generate_paramset
import generate_pendubot_problems


def benchmark_pendubot(horizon, n_problems=10):
    problems = generate_pendubot_problems.generate_pendubot_problems(
        n_problems, horizon
    )

    hyperparams_admm = generate_paramset.generate_admm_paramset(
        max_sqp_iter=100,
        adaptive_rho_tols=[5, 10, 25],
        rho_updates=[True, False],
    )
    param_search = ssqpy.solvers.ParamSearch(hyperparams_admm, problems)

    result_admm = param_search.search()

    hyperparams_hpipm = generate_paramset.generate_hpipm_paramset(
        max_sqp_iter=100
    )
    param_search = ssqpy.solvers.ParamSearch(hyperparams_hpipm, problems)

    result_hpipm = param_search.search()

    print("Best ADMM parameters:")

    result_admm.print()

    print("Best HPIPM parameters:")

    result_hpipm.print()

    # Solve one problem with best parameters

    result_admm.max_iter = 1000

    solver = ssqpy.solvers.ssqp(problems[0], result_admm)
    state, x, u = solver.solve()

    solver.write(f"../data/pendubot_admm_H{horizon}.json")

    result_hpipm.max_iter = 1000

    solver = ssqpy.solvers.ssqp(problems[0], result_hpipm)
    state, x, u = solver.solve()

    solver.write(f"../data/pendubot_hpipm_H{horizon}.json")


if __name__ == "__main__":
    # print("Benchmarking pendubot with H=20...")
    # benchmark_pendubot(20)
    print("Benchmarking pendubot with H=200...")
    benchmark_pendubot(200, 5)
