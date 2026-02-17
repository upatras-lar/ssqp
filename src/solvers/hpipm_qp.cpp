#include "solvers/hpipm_qp.hpp"

namespace ssqp
{
namespace solvers
{
hpipm_qp::hpipm_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _solution(qp.dims)
{
    d_ocp_qp_ipm_arg_set_iter_max(&_params.max_iter, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_stat(&_params.tol_stat, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&_params.tol_eq, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&_params.tol_ineq, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&_params.tol_comp, &_hpipm_qp_arg);
    int ws_int = _params.warm_start ? 1 : 0;
    d_ocp_qp_ipm_arg_set_warm_start(&ws_int, &_hpipm_qp_arg);

    int dim_size   = d_ocp_qp_dim_memsize(_qp.dims.horizon);
    _hpipm_dim_mem = malloc(dim_size);
    d_ocp_qp_dim_create(_qp.dims.horizon, &_hpipm_qp_dim, _hpipm_dim_mem);

    // Allocate QP problem structure
    int qp_size   = d_ocp_qp_memsize(&_hpipm_qp_dim);
    _hpipm_qp_mem = malloc(qp_size);
    d_ocp_qp_create(&_hpipm_qp_dim, &_hpipm_qp, _hpipm_qp_mem);

    // Allocate solution structure
    int sol_size   = d_ocp_qp_sol_memsize(&_hpipm_qp_dim);
    _hpipm_sol_mem = malloc(sol_size);
    d_ocp_qp_sol_create(&_hpipm_qp_dim, &_hpipm_qp_sol, _hpipm_sol_mem);

    // Allocate IPM arguments
    int arg_size   = d_ocp_qp_ipm_arg_memsize(&_hpipm_qp_dim);
    _hpipm_arg_mem = malloc(arg_size);
    d_ocp_qp_ipm_arg_create(&_hpipm_qp_dim, &_hpipm_qp_arg, _hpipm_arg_mem);

    // Set default IPM mode (SPEED, BALANCE, or ROBUST)
    d_ocp_qp_ipm_arg_set_default(ROBUST, &_hpipm_qp_arg);

    // Allocate workspace (size depends on arg settings)
    int ws_size   = d_ocp_qp_ipm_ws_memsize(&_hpipm_qp_dim, &_hpipm_qp_arg);
    _hpipm_ws_mem = malloc(ws_size);
    d_ocp_qp_ipm_ws_create(&_hpipm_qp_dim,
                           &_hpipm_qp_arg,
                           &_hpipm_qp_ws,
                           _hpipm_ws_mem);

    reset();
}

hpipm_qp::~hpipm_qp()
{
    if (_hpipm_ws_mem) free(_hpipm_ws_mem);
    if (_hpipm_arg_mem) free(_hpipm_arg_mem);
    if (_hpipm_sol_mem) free(_hpipm_sol_mem);
    if (_hpipm_qp_mem) free(_hpipm_qp_mem);
    if (_hpipm_dim_mem) free(_hpipm_dim_mem);
}

void hpipm_qp::reset()
{
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_dim_set_nx(k, _qp.dims.stage_state[k], &_hpipm_qp_dim);
        d_ocp_qp_dim_set_nu(k, _qp.dims.stage_control[k], &_hpipm_qp_dim);

        d_ocp_qp_set_A(k, (double *)(_qp.stages[k].A.data()), &_hpipm_qp);
        d_ocp_qp_set_B(k, (double *)(_qp.stages[k].B.data()), &_hpipm_qp);
        d_ocp_qp_set_b(k, (double *)(_qp.stages[k].gamma.data()), &_hpipm_qp);
        d_ocp_qp_set_Q(k, (double *)(_qp.stages[k].Qxx.data()), &_hpipm_qp);
        d_ocp_qp_set_S(k, (double *)(_qp.stages[k].Qxu.data()), &_hpipm_qp);
        d_ocp_qp_set_R(k, (double *)(_qp.stages[k].Quu.data()), &_hpipm_qp);
        d_ocp_qp_set_q(k, (double *)(_qp.stages[k].qx.data()), &_hpipm_qp);
        d_ocp_qp_set_r(k, (double *)(_qp.stages[k].qu.data()), &_hpipm_qp);

        if (_qp.dims.stage_constraint[k] > 0)
        {
            d_ocp_qp_dim_set_ng(k,
                                _qp.dims.stage_constraint[k],
                                &_hpipm_qp_dim);
            d_ocp_qp_set_C(k, (double *)(_qp.stages[k].D.data()), &_hpipm_qp);
            d_ocp_qp_set_D(k, (double *)(_qp.stages[k].E.data()), &_hpipm_qp);
            d_ocp_qp_set_lg(k, (double *)(_qp.stages[k].cl.data()), &_hpipm_qp);
            d_ocp_qp_set_ug(k, (double *)(_qp.stages[k].cu.data()), &_hpipm_qp);
        }
    }

    d_ocp_qp_dim_set_nx(_qp.dims.horizon,
                        _qp.dims.stage_state[_qp.dims.horizon],
                        &_hpipm_qp_dim);

    d_ocp_qp_set_Q(_qp.dims.horizon, (double *)(_qp.Qxx_t.data()), &_hpipm_qp);
    d_ocp_qp_set_q(_qp.dims.horizon, (double *)(_qp.qx_t.data()), &_hpipm_qp);

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        d_ocp_qp_dim_set_ng(_qp.dims.horizon,
                            _qp.dims.stage_constraint[_qp.dims.horizon],
                            &_hpipm_qp_dim);
        d_ocp_qp_set_C(_qp.dims.horizon, (double *)(_qp.Dt.data()), &_hpipm_qp);
        d_ocp_qp_set_lg(_qp.dims.horizon,
                        (double *)(_qp.cl_t.data()),
                        &_hpipm_qp);
        d_ocp_qp_set_ug(_qp.dims.horizon,
                        (double *)(_qp.cu_t.data()),
                        &_hpipm_qp);
    }
}

void hpipm_qp::increaseRegularization() {}

SolverState hpipm_qp::solve(hpipm_qp::Solution &sol)
{
    d_ocp_qp_ipm_solve(&_hpipm_qp,
                       &_hpipm_qp_sol,
                       &_hpipm_qp_arg,
                       &_hpipm_qp_ws);

    double *lg, *ug;

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_sol_get_x(k, &_hpipm_qp_sol, sol.dx.stage(k).data());
        d_ocp_qp_sol_get_u(k, &_hpipm_qp_sol, sol.du.stage(k).data());
        d_ocp_qp_sol_get_pi(k, &_hpipm_qp_sol, sol.lambda.stage(k).data());

        d_ocp_qp_sol_get_lam_lg(k, &_hpipm_qp_sol, lg);
        d_ocp_qp_sol_get_lam_ug(k, &_hpipm_qp_sol, ug);

        for (uint i = 0; i < sol.mu.stage(k).size(); ++i)
        {
            sol.mu.stage(k)[i] = ug[i] - lg[i];
        }
    }

    d_ocp_qp_sol_get_x(_qp.dims.horizon,
                       &_hpipm_qp_sol,
                       sol.dx.stage(_qp.dims.horizon).data());
    d_ocp_qp_sol_get_pi(_qp.dims.horizon,
                        &_hpipm_qp_sol,
                        sol.lambda.stage(_qp.dims.horizon).data());
    d_ocp_qp_sol_get_lam_lg(_qp.dims.horizon, &_hpipm_qp_sol, lg);
    d_ocp_qp_sol_get_lam_ug(_qp.dims.horizon, &_hpipm_qp_sol, ug);

    for (uint i = 0; i < sol.mu.stage(_qp.dims.horizon).size(); ++i)
    {
        sol.mu.stage(_qp.dims.horizon)[i] = ug[i] - lg[i];
    }

    return SUCCESS;
}

uint hpipm_qp::iterations() const {}

}
}