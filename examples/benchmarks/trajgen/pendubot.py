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
        max_sqp_iter=50,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            ssqpy.solvers.LineSearchMethod.Merit,
        ],
        merit_mu_cs=[0.5, 1.0, 2.0],
        merit_mu_gammas=[0.1, 0.5, 1.0],
        init_rhos=[0.1, 0.2, 0.6, 1.0],
        alphas=[1.0, 1.6],
        adaptive_rho_tols=[5, 10, 25],
        rho_updates=[True, False],
    )
    param_search = ssqpy.solvers.ParamSearch(hyperparams_admm, problems)

    result_admm = param_search.search()

    hyperparams_hpipm = generate_paramset.generate_hpipm_paramset(
        max_sqp_iter=50,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            ssqpy.solvers.LineSearchMethod.Merit,
        ],
        merit_mu_cs=[0.5, 1.0, 2.0],
        merit_mu_gammas=[0.1, 0.5, 1.0],
        square_root_riccatis=[True, False],
    )
    param_search = ssqpy.solvers.ParamSearch(hyperparams_hpipm, problems)

    result_hpipm = param_search.search()

    print("Best ADMM parameters:")

    result_admm.print()

    result_admm.write(f"../data/admm_params_H{horizon}.json")

    print("Best HPIPM parameters:")

    result_hpipm.print()

    result_hpipm.write(f"../data/hpipm_params_H{horizon}.json")


if __name__ == "__main__":
    # print("Benchmarking pendubot with H=25...")
    # benchmark_pendubot(25)

    print("Benchmarking pendubot with H=200...")
    benchmark_pendubot(200, 5)
