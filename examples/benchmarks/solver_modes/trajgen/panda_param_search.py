import ssqpy
import multiprocessing
import generate_paramset
import generate_panda_problems


def paramsearch_panda(dynamics_mode=ssqpy.model.SolverMode.InverseDynamics):
    dynamics_mode_str = (
        "forward"
        if dynamics_mode == ssqpy.model.SolverMode.ForwardDynamics
        else "inverse"
    )

    problems = generate_panda_problems.get_panda_problems(dynamics_mode)

    hyperparams_admm = generate_paramset.generate_admm_paramset(
        max_sqp_iter=50,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            ssqpy.solvers.LineSearchMethod.Merit,
        ],
        filter_sizes=[1, 10, 100],
        merit_mu_cs=[0.5, 1.0, 2.0],
        merit_mu_gammas=[0.1, 0.5, 1.0],
        init_rhos=[0.1, 0.2, 0.6, 1.0],
        alphas=[1.0, 1.6],
        adaptive_rho_tols=[5, 10, 25],
        rho_updates=[True, False],
    )
    param_search = ssqpy.solvers.HyperParamSearch(hyperparams_admm, problems)

    result_admm = param_search.search()

    print("Best ADMM parameters:")

    result_admm.print()

    result_admm.write(
        f"data/parameters/panda_admm_params_{dynamics_mode_str}.json"
    )

    """
    hyperparams_hpipm = generate_paramset.generate_hpipm_paramset(
        max_sqp_iter=50,
        line_search_methods=[
            ssqpy.solvers.LineSearchMethod.Filter,
            ssqpy.solvers.LineSearchMethod.Merit,
        ],
        filter_sizes=[1, 10, 100],
        merit_mu_cs=[0.5, 1.0, 2.0],
        merit_mu_gammas=[0.1, 0.5, 1.0],
        square_root_riccatis=[True, False],
    )
    param_search = ssqpy.solvers.HyperParamSearch(hyperparams_hpipm, problems)

    result_hpipm = param_search.search()

    print("Best HPIPM parameters:")

    result_hpipm.print()

    result_hpipm.write(
        f"data/parameters/panda_hpipm_params_{dynamics_mode_str}.json"
    )
    """


if __name__ == "__main__":
    # Inverse Dynamics
    H_inv = multiprocessing.Process(
        target=paramsearch_panda,
        args=(ssqpy.model.SolverMode.InverseDynamics,),
    )

    # Forward Dynamics
    H_fwd = multiprocessing.Process(
        target=paramsearch_panda,
        args=(ssqpy.model.SolverMode.ForwardDynamics,),
    )

    H_inv.start()
    H_fwd.start()

    H_inv.join()
    H_fwd.join()
