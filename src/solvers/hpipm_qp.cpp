#ifdef WITH_HPIPM
#include "solvers/hpipm_qp.hpp"

extern "C"
{
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_utils.h>
}

namespace ssqp
{
namespace solvers
{
struct hpipm_data
{
    d_ocp_qp_dim     dim;
    d_ocp_qp         qp;
    d_ocp_qp_sol     sol;
    d_ocp_qp_ipm_arg arg;
    d_ocp_qp_ipm_ws  ws;
};

hpipm_qp::Params::Params(const nlohmann::ordered_json &hpipmj)
{
    using namespace nlohmann;

    max_iter = hpipmj["max_iter"];

    if (hpipmj["mode"] == "Speed")
    {
        mode = HPIPMMode::SPEED;
    }
    else if (hpipmj["mode"] == "Balance")
    {
        mode = HPIPMMode::BALANCE;
    }
    else
    {
        mode = HPIPMMode::ROBUST;
    }

    tol_stat = hpipmj["tol_stat"];
    tol_eq   = hpipmj["tol_eq"];
    tol_ineq = hpipmj["tol_ineq"];
    tol_comp = hpipmj["tol_comp"];

    alpha_min = hpipmj["alpha_min"];
    mu0       = hpipmj["mu0"];
    reg_prim  = hpipmj["reg_prim"];

    warm_start            = hpipmj["warm_start"];
    prediction_correction = hpipmj["prediction_correction"];
    square_root_riccati   = hpipmj["square_root_riccati"];
    split_step            = hpipmj["split_step"];
}

void hpipm_qp::Params::print() const
{
    std::cout << "HPIPM Parameters" << std::endl;
    std::cout << "================" << std::endl;
    std::cout << "Maximum iterations: " << max_iter << std::endl;
    std::cout << "Mode: ";

    switch (mode)
    {
    case ROBUST:
        std::cout << "Robust" << std::endl;
        break;
    case SPEED:
        std::cout << "Speed" << std::endl;
        break;
    case BALANCE:
        std::cout << "Balance" << std::endl;
        break;
    default:
        break;
    }

    std::cout << "----------------" << std::endl;
    std::cout << "Stationarity tolerance: " << tol_stat << std::endl;
    std::cout << "Complementarity tolerance: " << tol_comp << std::endl;
    std::cout << "Equality tolerance: " << tol_eq << std::endl;
    std::cout << "Inequality tolerance: " << tol_ineq << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "Minimum alpha: " << alpha_min << std::endl;
    std::cout << "Initial barrier (mu0): " << mu0 << std::endl;
    std::cout << "Regularization weight: " << reg_prim << std::endl;

    std::cout << "----------------" << std::endl;
    std::cout << "Warm start enabled: " << warm_start << std::endl;
    std::cout << "Prediction correction enabled: " << prediction_correction
              << std::endl;
    std::cout << "Use square root riccati: " << square_root_riccati
              << std::endl;
    std::cout << "Split step for primal and dual variables enabled: "
              << split_step << std::endl;
    std::cout << "---------------" << std::endl;
}

void hpipm_qp::Params::write(nlohmann::ordered_json &j) const
{
    using namespace nlohmann;

    ordered_json hpipmj;

    hpipmj["max_iter"] = max_iter;
    switch (mode)
    {
    case ROBUST:
        hpipmj["mode"] = "Robust";
        break;
    case SPEED:
        hpipmj["mode"] = "Speed";
        break;
    case BALANCE:
        hpipmj["mode"] = "Balance";
        break;
    default:
        break;
    }

    hpipmj["tol_stat"] = tol_stat;
    hpipmj["tol_eq"]   = tol_eq;
    hpipmj["tol_ineq"] = tol_ineq;
    hpipmj["tol_comp"] = tol_comp;

    hpipmj["alpha_min"] = alpha_min;
    hpipmj["mu0"]       = mu0;
    hpipmj["reg_prim"]  = reg_prim;

    hpipmj["warm_start"]            = warm_start;
    hpipmj["prediction_correction"] = prediction_correction;
    hpipmj["square_root_riccati"]   = square_root_riccati;
    hpipmj["split_step"]            = split_step;

    j["hpipm"] = hpipmj;
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
  , _hpipm_data(std::make_shared<hpipm_data>())
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
    d_ocp_qp_dim_create(_qp.dims.horizon, &_hpipm_data->dim, _hpipm_dim_mem);

    set_stage_sizes();

    // Allocate QP problem structure
    int qp_size   = d_ocp_qp_memsize(&_hpipm_data->dim);
    _hpipm_qp_mem = malloc(qp_size);
    d_ocp_qp_create(&_hpipm_data->dim, &_hpipm_data->qp, _hpipm_qp_mem);

    reset();

    // Allocate solution structure
    int sol_size   = d_ocp_qp_sol_memsize(&_hpipm_data->dim);
    _hpipm_sol_mem = malloc(sol_size);
    d_ocp_qp_sol_create(&_hpipm_data->dim, &_hpipm_data->sol, _hpipm_sol_mem);

    // Allocate IPM arguments
    int arg_size   = d_ocp_qp_ipm_arg_memsize(&_hpipm_data->dim);
    _hpipm_arg_mem = malloc(arg_size);
    d_ocp_qp_ipm_arg_create(&_hpipm_data->dim,
                            &_hpipm_data->arg,
                            _hpipm_arg_mem);

    // Set default IPM mode (SPEED, BALANCE, or ROBUST)
    hpipm_mode mode;
    switch (_params.mode)
    {
    case HPIPMMode::ROBUST:
    default:
        mode = hpipm_mode::ROBUST;
        break;
    case HPIPMMode::SPEED:
        mode = hpipm_mode::SPEED;
        break;
    case HPIPMMode::BALANCE:
        mode = hpipm_mode::BALANCE;
        break;
    }

    d_ocp_qp_ipm_arg_set_default(mode, &_hpipm_data->arg);

    d_ocp_qp_ipm_arg_set_iter_max(&_params.max_iter, &_hpipm_data->arg);
    d_ocp_qp_ipm_arg_set_alpha_min(&_params.alpha_min, &_hpipm_data->arg);
    d_ocp_qp_ipm_arg_set_mu0(&_params.mu0, &_hpipm_data->arg);

    d_ocp_qp_ipm_arg_set_tol_stat(&_params.tol_stat, &_hpipm_data->arg);
    d_ocp_qp_ipm_arg_set_tol_eq(&_params.tol_eq, &_hpipm_data->arg);
    d_ocp_qp_ipm_arg_set_tol_ineq(&_params.tol_ineq, &_hpipm_data->arg);
    d_ocp_qp_ipm_arg_set_tol_comp(&_params.tol_comp, &_hpipm_data->arg);

    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_data->arg);

    int warm_start_int = _params.warm_start ? 1 : 0;
    d_ocp_qp_ipm_arg_set_warm_start(&warm_start_int, &_hpipm_data->arg);

    int pred_corr_int = _params.prediction_correction ? 1 : 0;
    d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr_int, &_hpipm_data->arg);

    int square_root_riccati_int = _params.square_root_riccati ? 1 : 0;
    d_ocp_qp_ipm_arg_set_pred_corr(&pred_corr_int, &_hpipm_data->arg);

    int split_step_int = _params.split_step ? 1 : 0;
    d_ocp_qp_ipm_arg_set_split_step(&split_step_int, &_hpipm_data->arg);

    // Allocate _hpipm_data->ws (size depends on arg settings)
    int ws_size = d_ocp_qp_ipm_ws_memsize(&_hpipm_data->dim, &_hpipm_data->arg);
    _hpipm_ws_mem = malloc(ws_size);
    d_ocp_qp_ipm_ws_create(&_hpipm_data->dim,
                           &_hpipm_data->arg,
                           &_hpipm_data->ws,
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
    d_ocp_qp_dim_set_nbx(0, _qp.dims.stage_state[0], &_hpipm_data->dim);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_dim_set_nx(k, _qp.dims.stage_state[k], &_hpipm_data->dim);
        d_ocp_qp_dim_set_nu(k, _qp.dims.stage_control[k], &_hpipm_data->dim);

        if (_qp.dims.stage_constraint[k] > 0)
        {
            d_ocp_qp_dim_set_ng(k,
                                _qp.dims.stage_constraint[k],
                                &_hpipm_data->dim);
        }
    }

    d_ocp_qp_dim_set_nx(_qp.dims.horizon,
                        _qp.dims.stage_state[_qp.dims.horizon],
                        &_hpipm_data->dim);

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        d_ocp_qp_dim_set_ng(_qp.dims.horizon,
                            _qp.dims.stage_constraint[_qp.dims.horizon],
                            &_hpipm_data->dim);
    }
}

void hpipm_qp::reset()
{
    _reg = _params.reg_prim;
    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_data->arg);

    std::vector<int> idxbx(_qp.dims.stage_state[0]);
    for (uint i = 0; i < idxbx.size(); ++i)
    {
        idxbx[i] = i;
    }

    VecX lbx = VecX::Zero(_qp.dims.stage_state[0]);
    VecX ubx = VecX::Zero(_qp.dims.stage_state[0]);

    d_ocp_qp_set_idxbx(0, idxbx.data(), &_hpipm_data->qp);
    d_ocp_qp_set_lbx(0, lbx.data(), &_hpipm_data->qp);
    d_ocp_qp_set_ubx(0, ubx.data(), &_hpipm_data->qp);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_set_A(k, (double *)(_qp.stages[k].A.data()), &_hpipm_data->qp);
        d_ocp_qp_set_B(k, (double *)(_qp.stages[k].B.data()), &_hpipm_data->qp);
        d_ocp_qp_set_b(k,
                       (double *)(_qp.stages[k].gamma.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_Q(k,
                       (double *)(_qp.stages[k].Qxx.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_S(k,
                       (double *)(_qp.stages[k].Qxu.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_R(k,
                       (double *)(_qp.stages[k].Quu.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_q(k,
                       (double *)(_qp.stages[k].qx.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_r(k,
                       (double *)(_qp.stages[k].qu.data()),
                       &_hpipm_data->qp);

        if (_qp.dims.stage_constraint[k] > 0)
        {
            d_ocp_qp_set_C(k,
                           (double *)(_qp.stages[k].D.data()),
                           &_hpipm_data->qp);
            d_ocp_qp_set_D(k,
                           (double *)(_qp.stages[k].E.data()),
                           &_hpipm_data->qp);

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

            d_ocp_qp_set_lg(k, (double *)(_cl[k].data()), &_hpipm_data->qp);
            d_ocp_qp_set_ug(k, (double *)(_cu[k].data()), &_hpipm_data->qp);
            d_ocp_qp_set_lg_mask(k, _cl_masks[k].data(), &_hpipm_data->qp);
            d_ocp_qp_set_ug_mask(k, _cu_masks[k].data(), &_hpipm_data->qp);
        }
    }

    d_ocp_qp_set_Q(_qp.dims.horizon,
                   (double *)(_qp.Qxx_t.data()),
                   &_hpipm_data->qp);
    d_ocp_qp_set_q(_qp.dims.horizon,
                   (double *)(_qp.qx_t.data()),
                   &_hpipm_data->qp);

    if (_qp.dims.stage_constraint[_qp.dims.horizon] > 0)
    {
        d_ocp_qp_set_C(_qp.dims.horizon,
                       (double *)(_qp.Dt.data()),
                       &_hpipm_data->qp);

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
                        &_hpipm_data->qp);
        d_ocp_qp_set_ug(_qp.dims.horizon,
                        (double *)(_cu[_qp.dims.horizon].data()),
                        &_hpipm_data->qp);
        d_ocp_qp_set_lg_mask(_qp.dims.horizon,
                             _cl_masks[_qp.dims.horizon].data(),
                             &_hpipm_data->qp);
        d_ocp_qp_set_ug_mask(_qp.dims.horizon,
                             _cu_masks[_qp.dims.horizon].data(),
                             &_hpipm_data->qp);
    }
}

void hpipm_qp::increaseRegularization()
{
    _reg *= 10;

    if (_reg > _reg_max)
    {
        throw std::runtime_error("HPIPM: Regularization failed");
    }

    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_data->arg);
}

SolverState hpipm_qp::solve(hpipm_qp::Solution &sol)
{
    d_ocp_qp_ipm_solve(&_hpipm_data->qp,
                       &_hpipm_data->sol,
                       &_hpipm_data->arg,
                       &_hpipm_data->ws);

    d_ocp_qp_ipm_get_status(&_hpipm_data->ws, &_status);
    d_ocp_qp_ipm_get_iter(&_hpipm_data->ws, &_iter);

    if (GlobalLogMask & LogLevel::Warning && _status != 0)
    {
        logln_at(LogLevel::Warning,
                 "Warning: HPIPM returned status " << _status);

        // double res_stat;
        // double res_eq;
        // double res_ineq;
        // double res_comp;

        // d_ocp_qp_ipm_get_max_res_stat(&_hpipm_data->ws, &res_stat);
        // d_ocp_qp_ipm_get_max_res_eq(&_hpipm_data->ws, &res_eq);
        // d_ocp_qp_ipm_get_max_res_ineq(&_hpipm_data->ws, &res_ineq);
        // d_ocp_qp_ipm_get_max_res_comp(&_hpipm_data->ws, &res_comp);

        // logln(" -> Stationarity Residual:   " << res_stat);
        // logln(" -> Dynamics (Eq) Residual:  " << res_eq);
        // logln(" -> Bounds (Ineq) Residual:  " << res_ineq);
        // logln(" -> Complementarity Resid:   " << res_comp);
    }

    logln_at(LogLevel::Debug, "HPIPM status: " << _status);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_sol_get_x(k, &_hpipm_data->sol, sol.dx.stage(k).data());
        d_ocp_qp_sol_get_u(k, &_hpipm_data->sol, sol.du.stage(k).data());
        d_ocp_qp_sol_get_pi(k, &_hpipm_data->sol, sol.lambda.stage(k).data());

        VecX lg(_qp.dims.stage_constraint[k]), ug(_qp.dims.stage_constraint[k]);

        d_ocp_qp_sol_get_lam_lg(k, &_hpipm_data->sol, lg.data());
        d_ocp_qp_sol_get_lam_ug(k, &_hpipm_data->sol, ug.data());

        sol.mu.stage(k) = lg - ug;
    }

    d_ocp_qp_sol_get_x(_qp.dims.horizon,
                       &_hpipm_data->sol,
                       sol.dx.stage(_qp.dims.horizon).data());

    VecX lg(_qp.dims.stage_constraint[_qp.dims.horizon]),
      ug(_qp.dims.stage_constraint[_qp.dims.horizon]);

    d_ocp_qp_sol_get_lam_lg(_qp.dims.horizon, &_hpipm_data->sol, lg.data());
    d_ocp_qp_sol_get_lam_ug(_qp.dims.horizon, &_hpipm_data->sol, ug.data());

    sol.mu.stage(_qp.dims.horizon) = lg - ug;

    return SUCCESS;
}

uint hpipm_qp::iterations() const
{
    return _iter;
}

}
}
#endif