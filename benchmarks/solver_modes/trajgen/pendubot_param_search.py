import ssqpy
import multiprocessing
import generate_paramset
import generate_pendubot_problems


def paramsearch_pendubot(
    horizon,
    n_problems=10,
    dynamics_mode=ssqpy.model.SolverMode.InverseDynamics,
):
    dynamics_mode_str = (
        "forward"
        if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics
        else "inverse"
    )

    problems = generate_pendubot_problems.generate_pendubot_problems(
        n_problems, horizon, dynamics_mode
    )

    hyperparams_admm = generate_paramset.generate_admm_paramset(
        max_sqp_iter=200,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            # ssqpy.solvers.LineSearchMethod.Merit,
        ],
        filter_sizes=[1, 10, 100],
        # merit_mu_cs=[0.5, 1.0, 2.0],
        # merit_mu_gammas=[0.1, 1.0],
        init_rhos=[0.1, 0.2, 1.0],
        alphas=[1.0, 1.6],
        adaptive_rho_tols=[1, 5, 10, 25],
        rho_update_iter_freqs=[1, 5, 10, 20],
        rho_updates=[True, False],
    )
    param_search = ssqpy.solvers.HyperParamSearch(hyperparams_admm, problems)

    result_admm = param_search.search()

    print("Best ADMM parameters:")

    result_admm.print()

    result_admm.write(
        f"data/parameters/pendubot_admm_params_H{horizon}_{dynamics_mode_str}.json"
    )

    """
    hyperparams_hpipm = generate_paramset.generate_hpipm_paramset(
        max_sqp_iter=200,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            ssqpy.solvers.LineSearchMethod.Merit,
        ],
        filter_sizes=[1, 10, 100],
        merit_mu_cs=[0.5, 1.0, 2.0],
        merit_mu_gammas=[0.1, 1.0],
        reg_prims=[1e-12],
    )
    param_search = ssqpy.solvers.HyperParamSearch(hyperparams_hpipm, problems)

    result_hpipm = param_search.search()

    print("Best HPIPM parameters:")

    result_hpipm.print()

    result_hpipm.write(
        f"data/parameters/pendubot_hpipm_params_H{horizon}_{dynamics_mode_str}.json"
    )
    """


H_small = 25
H_large = 200

if __name__ == "__main__":
    # Small Horizon, Inverse Dynamics
    H_small_inv = multiprocessing.Process(
        target=paramsearch_pendubot,
        args=(H_small, 10, ssqpy.model.SolverMode.InverseDynamics),
    )

    # Large Horizon, Inverse Dynamics
    H_large_inv = multiprocessing.Process(
        target=paramsearch_pendubot,
        args=(H_large, 5, ssqpy.model.SolverMode.InverseDynamics),
    )

    # Small Horizon, Forward Dynamics
    H_small_fwd = multiprocessing.Process(
        target=paramsearch_pendubot,
        args=(H_small, 10, ssqpy.model.SolverMode.ForwardDynamics),
    )

    # Large Horizon, Forward Dynamics
    H_large_fwd = multiprocessing.Process(
        target=paramsearch_pendubot,
        args=(H_large, 5, ssqpy.model.SolverMode.ForwardDynamics),
    )

    # H_small_inv.start()
    H_large_inv.start()
    # H_small_fwd.start()
    H_large_fwd.start()

    # H_small_inv.join()
    H_large_inv.join()
    # H_small_fwd.join()
    H_large_fwd.join()
