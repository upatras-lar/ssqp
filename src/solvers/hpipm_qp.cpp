#include "solvers/hpipm_qp.hpp"

namespace ssqp
{
namespace solvers
{
void hpipm_qp::Params::print() const
{
    LOGLN("HPIPM Parameters");
    LOGLN("================");
    LOGLN("Maximum iterations: " << max_iter);
    LOG("Mode: ");

    switch (mode)
    {
    case ROBUST:
        LOGLN("Robust");
        break;
    case SPEED:
        LOGLN("Speed");
        break;
    case BALANCE:
        LOGLN("Balance");
        break;
    default:
        break;
    }

    LOGLN("----------------");
    LOGLN("Stationarity tolerance: " << tol_stat);
    LOGLN("Complementarity tolerance: " << tol_comp);
    LOGLN("Equality tolerance: " << tol_eq);
    LOGLN("Inequality tolerance: " << tol_ineq);

    LOGLN("----------------");
    LOGLN("Minimum alpha: " << alpha_min);
    LOGLN("Initial barrier (mu0): " << mu0);
    LOGLN("Regularization weight: " << reg_prim);

    LOGLN("----------------");
    LOGLN("Warm start enabled: " << warm_start);
    LOGLN("Prediction correction enabled: " << prediction_correction);
    LOGLN("Use square root riccati: " << square_root_riccati);
    LOGLN("Split step for primal and dual variables enabled: " << split_step);
}

hpipm_qp::hpipm_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _solution(qp.dims)
  , _reg(_params.reg_prim)
  , _cl(_qp.dims.horizon + 1)
  , _cu(_qp.dims.horizon + 1)
  , _cl_masks(_qp.dims.horizon + 1)
  , _cu_masks(_qp.dims.horizon + 1)
{
    for (uint k = 0; k <= _qp.dims.horizon; ++k)
    {
        _cl[k].resize(_qp.dims.stage_constraint[k]);
        _cu[k].resize(_qp.dims.stage_constraint[k]);
        _cl_masks[k].resize(_qp.dims.stage_constraint[k]);
        _cu_masks[k].resize(_qp.dims.stage_constraint[k]);
    }

    int dim_size   = d_ocp_qp_dim_memsize(_qp.dims.horizon);
    _hpipm_dim_mem = malloc(dim_size);
    d_ocp_qp_dim_create(_qp.dims.horizon, &_hpipm_qp_dim, _hpipm_dim_mem);

    set_stage_sizes();

    // Allocate QP problem structure
    int qp_size   = d_ocp_qp_memsize(&_hpipm_qp_dim);
    _hpipm_qp_mem = malloc(qp_size);
    d_ocp_qp_create(&_hpipm_qp_dim, &_hpipm_qp, _hpipm_qp_mem);

    reset();

    // Allocate solution structure
    int sol_size   = d_ocp_qp_sol_memsize(&_hpipm_qp_dim);
    _hpipm_sol_mem = malloc(sol_size);
    d_ocp_qp_sol_create(&_hpipm_qp_dim, &_hpipm_qp_sol, _hpipm_sol_mem);

    // Allocate IPM arguments
    int arg_size   = d_ocp_qp_ipm_arg_memsize(&_hpipm_qp_dim);
    _hpipm_arg_mem = malloc(arg_size);
    d_ocp_qp_ipm_arg_create(&_hpipm_qp_dim, &_hpipm_qp_arg, _hpipm_arg_mem);

    // Set default IPM mode (SPEED, BALANCE, or ROBUST)
    d_ocp_qp_ipm_arg_set_default(_params.mode, &_hpipm_qp_arg);

    d_ocp_qp_ipm_arg_set_iter_max(&_params.max_iter, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_alpha_min(&_params.alpha_min, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_mu0(&_params.mu0, &_hpipm_qp_arg);

    d_ocp_qp_ipm_arg_set_tol_stat(&_params.tol_stat, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&_params.tol_eq, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&_params.tol_ineq, &_hpipm_qp_arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&_params.tol_comp, &_hpipm_qp_arg);

    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_qp_arg);

    int warm_start_int = _params.warm_start ? 1 : 0;
    d_ocp_qp_ipm_arg_set_warm_start(&warm_start_int, &_hpipm_qp_arg);

    int pred_corr_int = _params.prediction_correction ? 1 : 0;
    d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr_int, &_hpipm_qp_arg);

    int square_root_riccati_int = _params.square_root_riccati ? 1 : 0;
    d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr_int, &_hpipm_qp_arg);

    int split_step_int = _params.split_step ? 1 : 0;
    d_ocp_qp_ipm_arg_set_split_step(&split_step_int, &_hpipm_qp_arg);

    // Allocate workspace (size depends on arg settings)
    int ws_size   = d_ocp_qp_ipm_ws_memsize(&_hpipm_qp_dim, &_hpipm_qp_arg);
    _hpipm_ws_mem = malloc(ws_size);
    d_ocp_qp_ipm_ws_create(&_hpipm_qp_dim,
                           &_hpipm_qp_arg,
                           &_hpipm_qp_ws,
                           _hpipm_ws_mem);
}

hpipm_qp::~hpipm_qp()
{
    if (_hpipm_ws_mem) free(_hpipm_ws_mem);
    if (_hpipm_arg_mem) free(_hpipm_arg_mem);
    if (_hpipm_sol_mem) free(_hpipm_sol_mem);
    if (_hpipm_qp_mem) free(_hpipm_qp_mem);
    if (_hpipm_dim_mem) free(_hpipm_dim_mem);
}

void hpipm_qp::set_stage_sizes()
{
    d_ocp_qp_dim_set_nbx(0, _qp.dims.stage_state[0], &_hpipm_qp_dim);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_dim_set_nx(k, _qp.dims.stage_state[k], &_hpipm_qp_dim);
        d_ocp_qp_dim_set_nu(k, _qp.dims.stage_control[k], &_hpipm_qp_dim);

        if (_qp.dims.stage_constraint[k] > 0)
        {
            d_ocp_qp_dim_set_ng(k,
                                _qp.dims.stage_constraint[k],
                                &_hpipm_qp_dim);
        }
    }

    d_ocp_qp_dim_set_nx(_qp.dims.horizon,
                        _qp.dims.stage_state[_qp.dims.horizon],
                        &_hpipm_qp_dim);

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        d_ocp_qp_dim_set_ng(_qp.dims.horizon,
                            _qp.dims.stage_constraint[_qp.dims.horizon],
                            &_hpipm_qp_dim);
    }
}

void hpipm_qp::reset()
{
    _reg = _params.reg_prim;
    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_qp_arg);

    std::vector<int> idxbx(_qp.dims.stage_state[0]);
    for (uint i = 0; i < idxbx.size(); ++i)
    {
        idxbx[i] = i;
    }

    VecX lbx = VecX::Zero(_qp.dims.stage_state[0]);
    VecX ubx = VecX::Zero(_qp.dims.stage_state[0]);

    d_ocp_qp_set_idxbx(0, idxbx.data(), &_hpipm_qp);
    d_ocp_qp_set_lbx(0, lbx.data(), &_hpipm_qp);
    d_ocp_qp_set_ubx(0, ubx.data(), &_hpipm_qp);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
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
            d_ocp_qp_set_C(k, (double *)(_qp.stages[k].D.data()), &_hpipm_qp);
            d_ocp_qp_set_D(k, (double *)(_qp.stages[k].E.data()), &_hpipm_qp);

            _cl[k] = _qp.stages[k].cl;
            _cu[k] = _qp.stages[k].cu;

            for (uint c = 0; c < _qp.dims.stage_constraint[k]; ++c)
            {
                if (std::isinf(_qp.stages[k].cl[c]))
                {
                    _cl[k][c]       = 0.0;
                    _cl_masks[k][c] = 0.0;
                }
                else
                {
                    _cl_masks[k][c] = 1.0;
                }

                if (std::isinf(_qp.stages[k].cu[c]))
                {
                    _cu[k][c]       = 0.0;
                    _cu_masks[k][c] = 0.0;
                }
                else
                {
                    _cu_masks[k][c] = 1.0;
                }
            }

            d_ocp_qp_set_lg(k, (double *)(_cl[k].data()), &_hpipm_qp);
            d_ocp_qp_set_ug(k, (double *)(_cu[k].data()), &_hpipm_qp);
            d_ocp_qp_set_lg_mask(k, _cl_masks[k].data(), &_hpipm_qp);
            d_ocp_qp_set_ug_mask(k, _cu_masks[k].data(), &_hpipm_qp);
        }
    }

    d_ocp_qp_set_Q(_qp.dims.horizon, (double *)(_qp.Qxx_t.data()), &_hpipm_qp);
    d_ocp_qp_set_q(_qp.dims.horizon, (double *)(_qp.qx_t.data()), &_hpipm_qp);

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        d_ocp_qp_set_C(_qp.dims.horizon, (double *)(_qp.Dt.data()), &_hpipm_qp);

        _cl[_qp.dims.horizon] = _qp.cl_t;
        _cu[_qp.dims.horizon] = _qp.cu_t;

        for (uint c = 0; c < _qp.dims.stage_constraint[_qp.dims.horizon]; ++c)
        {
            if (std::isinf(_qp.cl_t[c]))
            {
                _cl[_qp.dims.horizon][c]       = 0.0;
                _cl_masks[_qp.dims.horizon][c] = 0.0;
            }
            else
            {
                _cl_masks[_qp.dims.horizon][c] = 1.0;
            }

            if (std::isinf(_qp.cu_t[c]))
            {
                _cu[_qp.dims.horizon][c]       = 0.0;
                _cu_masks[_qp.dims.horizon][c] = 0.0;
            }
            else
            {
                _cu_masks[_qp.dims.horizon][c] = 1.0;
            }
        }

        d_ocp_qp_set_lg(_qp.dims.horizon,
                        (double *)(_cl[_qp.dims.horizon].data()),
                        &_hpipm_qp);
        d_ocp_qp_set_ug(_qp.dims.horizon,
                        (double *)(_cu[_qp.dims.horizon].data()),
                        &_hpipm_qp);
        d_ocp_qp_set_lg_mask(_qp.dims.horizon,
                             _cl_masks[_qp.dims.horizon].data(),
                             &_hpipm_qp);
        d_ocp_qp_set_ug_mask(_qp.dims.horizon,
                             _cu_masks[_qp.dims.horizon].data(),
                             &_hpipm_qp);
    }
}

void hpipm_qp::increaseRegularization()
{
    _reg *= 10;

    if (_reg > _reg_max)
    {
        throw std::runtime_error("HPIPM: Regularization failed");
    }

    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_qp_arg);
}

SolverState hpipm_qp::solve(hpipm_qp::Solution &sol)
{
    d_ocp_qp_ipm_solve(&_hpipm_qp,
                       &_hpipm_qp_sol,
                       &_hpipm_qp_arg,
                       &_hpipm_qp_ws);

    d_ocp_qp_ipm_get_status(&_hpipm_qp_ws, &_status);
    d_ocp_qp_ipm_get_iter(&_hpipm_qp_ws, &_iter);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_sol_get_x(k, &_hpipm_qp_sol, sol.dx.stage(k).data());
        d_ocp_qp_sol_get_u(k, &_hpipm_qp_sol, sol.du.stage(k).data());
        d_ocp_qp_sol_get_pi(k, &_hpipm_qp_sol, sol.lambda.stage(k).data());

        VecX lg(_qp.dims.stage_constraint[k]), ug(_qp.dims.stage_constraint[k]);

        d_ocp_qp_sol_get_lam_lg(k, &_hpipm_qp_sol, lg.data());
        d_ocp_qp_sol_get_lam_ug(k, &_hpipm_qp_sol, ug.data());

        sol.mu.stage(k) = lg - ug;
    }

    d_ocp_qp_sol_get_x(_qp.dims.horizon,
                       &_hpipm_qp_sol,
                       sol.dx.stage(_qp.dims.horizon).data());

    VecX lg(_qp.dims.stage_constraint[_qp.dims.horizon]),
      ug(_qp.dims.stage_constraint[_qp.dims.horizon]);

    d_ocp_qp_sol_get_lam_lg(_qp.dims.horizon, &_hpipm_qp_sol, lg.data());
    d_ocp_qp_sol_get_lam_ug(_qp.dims.horizon, &_hpipm_qp_sol, ug.data());

    sol.mu.stage(_qp.dims.horizon) = lg - ug;

    return SUCCESS;
}

uint hpipm_qp::iterations() const
{
    return _iter;
}

}
}