import ssqpy
from itertools import product


def generate_admm_paramset(
    max_sqp_iter=500,
    line_search_methods=[ssqpy.solvers.LineSearchMethod.Filter],
    filter_sizes=[1, 100],
    merit_mu_cs=[1.0],
    merit_mu_gammas=[1.0],
    init_rhos=[0.1],
    alphas=[1.6],
    sigmas=[1e-6],
    rho_updates=[True, False],
    adaptive_rho_tols=[10],
    rho_update_iter_freqs=[25],
    warm_start=[True, False],
    reset_rho_toggle=[False],
    reset_y_toggle=[False],
):
    paramset = []

    for lsm in line_search_methods:
        if lsm == ssqpy.solvers.LineSearchMethod.Filter:
            paramset.extend(
                _generate_admm_filter_paramset(
                    max_sqp_iter,
                    filter_sizes,
                    init_rhos,
                    alphas,
                    sigmas,
                    rho_updates,
                    adaptive_rho_tols,
                    rho_update_iter_freqs,
                    warm_start,
                    reset_rho_toggle,
                    reset_y_toggle,
                )
            )
        elif lsm == ssqpy.solvers.LineSearchMethod.Merit:
            paramset.extend(
                _generate_admm_merit_paramset(
                    max_sqp_iter,
                    merit_mu_cs,
                    merit_mu_gammas,
                    init_rhos,
                    alphas,
                    sigmas,
                    rho_updates,
                    adaptive_rho_tols,
                    rho_update_iter_freqs,
                    warm_start,
                    reset_rho_toggle,
                    reset_y_toggle,
                )
            )

    return paramset


def _generate_admm_filter_paramset(
    max_sqp_iter,
    filter_sizes,
    init_rhos,
    alphas,
    sigmas,
    rho_updates,
    adaptive_rho_tols,
    rho_update_iter_freqs,
    warm_starts,
    reset_rhos,
    reset_ys,
):
    paramset = []

    no_rho_update_set = set()

    for (
        filter_size,
        init_rho,
        alpha,
        sigma,
        rho_update,
        adaptive_rho_tol,
        rho_update_iter_freq,
        warm_start,
        reset_rho,
        reset_y,
    ) in product(
        filter_sizes,
        init_rhos,
        alphas,
        sigmas,
        rho_updates,
        adaptive_rho_tols,
        rho_update_iter_freqs,
        warm_starts,
        reset_rhos,
        reset_ys,
    ):
        # if not updating rho, skip this set if it only differs in rho update
        # specific parameters (would be the same as existing)
        if not rho_update:
            tpl = (filter_size, init_rho, alpha, sigma, warm_start, reset_y)

            if tpl in no_rho_update_set:
                continue

            no_rho_update_set.add(tpl)

        params = ssqpy.solvers.ssqpParams()
        params.accept_worse_update = True
        params.max_iter = max_sqp_iter
        params.line_search_method = ssqpy.solvers.LineSearchMethod.Filter
        params.filter_size = filter_size
        params.qp_solver_type = ssqpy.solvers.QPSolverType.ADMM
        params.admm_params.init_rho = init_rho
        params.admm_params.alpha = alpha
        params.admm_params.sigma = sigma
        params.admm_params.rho_update = rho_update
        params.admm_params.adaptive_rho_tolerance = adaptive_rho_tol
        params.admm_params.rho_update_iter_freq = rho_update_iter_freq
        params.admm_params.warm_start = warm_start
        params.admm_params.reset_rho = reset_rho
        params.admm_params.reset_y = reset_y

        paramset.append(params)

    return paramset


def _generate_admm_merit_paramset(
    max_sqp_iter,
    merit_mu_cs,
    merit_mu_gammas,
    init_rhos,
    alphas,
    sigmas,
    rho_updates,
    adaptive_rho_tols,
    rho_update_iter_freqs,
    warm_starts,
    reset_rhos,
    reset_ys,
):
    paramset = []

    no_rho_update_set = set()

    for (
        merit_mu_c,
        merit_mu_gamma,
        init_rho,
        alpha,
        sigma,
        rho_update,
        adaptive_rho_tol,
        rho_update_iter_freq,
        warm_start,
        reset_rho,
        reset_y,
    ) in product(
        merit_mu_cs,
        merit_mu_gammas,
        init_rhos,
        alphas,
        sigmas,
        rho_updates,
        adaptive_rho_tols,
        rho_update_iter_freqs,
        warm_starts,
        reset_rhos,
        reset_ys,
    ):
        # if not updating rho, skip this set if it only differs in rho update
        # specific parameters (would be the same as existing)
        if not rho_update:
            tpl = (
                merit_mu_c,
                merit_mu_gamma,
                init_rho,
                alpha,
                sigma,
                warm_start,
                reset_y,
            )

            if tpl in no_rho_update_set:
                continue

            no_rho_update_set.add(tpl)

        params = ssqpy.solvers.ssqpParams()
        params.accept_worse_update = True
        params.max_iter = max_sqp_iter
        params.line_search_method = ssqpy.solvers.LineSearchMethod.Merit
        params.merit_mu_c = merit_mu_c
        params.merit_mu_gamma = merit_mu_gamma
        params.qp_solver_type = ssqpy.solvers.QPSolverType.ADMM
        params.admm_params.init_rho = init_rho
        params.admm_params.alpha = alpha
        params.admm_params.sigma = sigma
        params.admm_params.rho_update = rho_update
        params.admm_params.adaptive_rho_tolerance = adaptive_rho_tol
        params.admm_params.rho_update_iter_freq = rho_update_iter_freq
        params.admm_params.warm_start = warm_start
        params.admm_params.reset_rho = reset_rho
        params.admm_params.reset_y = reset_y

        paramset.append(params)

    return paramset


def generate_hpipm_paramset(
    max_sqp_iter=500,
    line_search_methods=[ssqpy.solvers.LineSearchMethod.Filter],
    filter_sizes=[1, 100],
    merit_mu_cs=[1.0],
    merit_mu_gammas=[1.0],
    modes=[ssqpy.solvers.HPIPMMode.Robust],
    alpha_mins=[1e-8],
    mu0s=[100],
    reg_prims=[1e-12, 1e-5],
    warm_starts=[True, False],
    prediction_corrections=[True, False],
    square_root_riccatis=[True],
    split_steps=[True, False],
):
    paramset = []

    for lsm in line_search_methods:
        if lsm == ssqpy.solvers.LineSearchMethod.Filter:
            paramset.extend(
                _generate_hpipm_filter_paramset(
                    max_sqp_iter,
                    filter_sizes,
                    modes,
                    alpha_mins,
                    mu0s,
                    reg_prims,
                    warm_starts,
                    prediction_corrections,
                    square_root_riccatis,
                    split_steps,
                )
            )
        elif lsm == ssqpy.solvers.LineSearchMethod.Merit:
            paramset.extend(
                _generate_hpipm_merit_paramset(
                    max_sqp_iter,
                    merit_mu_cs,
                    merit_mu_gammas,
                    modes,
                    alpha_mins,
                    mu0s,
                    reg_prims,
                    warm_starts,
                    prediction_corrections,
                    square_root_riccatis,
                    split_steps,
                )
            )

    return paramset


def _generate_hpipm_filter_paramset(
    max_sqp_iter,
    filter_sizes,
    modes,
    alpha_mins,
    mu0s,
    reg_prims,
    warm_starts,
    prediction_corrections,
    square_root_riccatis,
    split_steps,
):
    paramset = []

    for (
        filter_size,
        mode,
        alpha_min,
        mu0,
        reg_prim,
        warm_start,
        prediction_correction,
        square_root_riccati,
        split_step,
    ) in product(
        filter_sizes,
        modes,
        alpha_mins,
        mu0s,
        reg_prims,
        warm_starts,
        prediction_corrections,
        square_root_riccatis,
        split_steps,
    ):
        params = ssqpy.solvers.ssqpParams()
        params.accept_worse_update = True
        params.max_iter = max_sqp_iter
        params.line_search_method = ssqpy.solvers.LineSearchMethod.Filter
        params.filter_size = filter_size
        params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
        params.hpipm_params.mode = mode
        params.hpipm_params.alpha_min = alpha_min
        params.hpipm_params.mu0 = mu0
        params.hpipm_params.reg_prim = reg_prim
        params.hpipm_params.warm_start = warm_start
        params.hpipm_params.prediction_correction = prediction_correction
        params.hpipm_params.square_root_riccati = square_root_riccati
        params.hpipm_params.split_step = split_step

        paramset.append(params)

    return paramset


def _generate_hpipm_merit_paramset(
    max_sqp_iter,
    merit_mu_cs,
    merit_mu_gammas,
    modes,
    alpha_mins,
    mu0s,
    reg_prims,
    warm_starts,
    prediction_corrections,
    square_root_riccatis,
    split_steps,
):
    paramset = []

    for (
        merit_mu_c,
        merit_mu_gamma,
        mode,
        alpha_min,
        mu0,
        reg_prim,
        warm_start,
        prediction_correction,
        square_root_riccati,
        split_step,
    ) in product(
        merit_mu_cs,
        merit_mu_gammas,
        modes,
        alpha_mins,
        mu0s,
        reg_prims,
        warm_starts,
        prediction_corrections,
        square_root_riccatis,
        split_steps,
    ):
        params = ssqpy.solvers.ssqpParams()
        params.accept_worse_update = True
        params.max_iter = max_sqp_iter
        params.line_search_method = ssqpy.solvers.LineSearchMethod.Merit
        params.merit_mu_c = merit_mu_c
        params.merit_mu_gamma = merit_mu_gamma
        params.qp_solver_type = ssqpy.solvers.QPSolverType.HPIPM
        params.hpipm_params.mode = mode
        params.hpipm_params.alpha_min = alpha_min
        params.hpipm_params.mu0 = mu0
        params.hpipm_params.reg_prim = reg_prim
        params.hpipm_params.warm_start = warm_start
        params.hpipm_params.prediction_correction = prediction_correction
        params.hpipm_params.square_root_riccati = square_root_riccati
        params.hpipm_params.split_step = split_step

        paramset.append(params)

    return paramset
