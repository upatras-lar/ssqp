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

    // Missing parameters keep their defaults
    max_iter = hpipmj.value("max_iter", max_iter);

    const std::string mode_name = hpipmj.value("mode", "");
    if (mode_name == "Speed")
    {
        mode = HPIPMMode::SPEED;
    }
    else if (mode_name == "Balance")
    {
        mode = HPIPMMode::BALANCE;
    }
    else
    {
        mode = HPIPMMode::ROBUST;
    }

    tol_stat = hpipmj.value("tol_stat", tol_stat);
    tol_eq   = hpipmj.value("tol_eq", tol_eq);
    tol_ineq = hpipmj.value("tol_ineq", tol_ineq);
    tol_comp = hpipmj.value("tol_comp", tol_comp);

    alpha_min = hpipmj.value("alpha_min", alpha_min);
    mu0       = hpipmj.value("mu0", mu0);
    reg_prim  = hpipmj.value("reg_prim", reg_prim);

    warm_start = hpipmj.value("warm_start", warm_start);
    prediction_correction =
      hpipmj.value("prediction_correction", prediction_correction);
    square_root_riccati =
      hpipmj.value("square_root_riccati", square_root_riccati);
    split_step = hpipmj.value("split_step", split_step);
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
  , _S(_qp.dims.horizon)
  , _hpipm_data(std::make_shared<hpipm_data>())
{
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _S[k].resize(_qp.dims.stage_control[k], _qp.dims.stage_state[k]);
    }

    // reset() creates the HPIPM structures and uploads numeric QP data. The QP
    // and its box layout are assembled by SSQP after this constructor
    // returns, so defer both until the first solve.
}

hpipm_qp::~hpipm_qp()
{
    _free();
}

void hpipm_qp::_split_rows(const uint k, const uint nu, const bool fixed_state)
{
    const uint nx = _qp.dims.stage_state[k];
    const uint nc = _qp.dims.stage_constraint[k];

    const std::vector<model::BoxRow> &box_rows = _box_rows[k];

    Stage &stage  = _stages[k];
    stage.x.fixed = fixed_state;
    stage.x.layout(box_rows, false, nx);
    stage.u.layout(box_rows, true, nu);

    std::vector<bool> is_box(nc, false);
    for (const model::BoxRow &box : box_rows)
    {
        is_box[box.row] = true;
    }

    uint ng = 0;
    stage.general_rows.resize(nc);
    for (uint c = 0; c < nc; ++c)
    {
        if (!is_box[c])
        {
            stage.general_rows[ng++] = c;
        }
    }
    stage.general_rows.resize(ng);

    stage.C.resize(ng, nx);
    stage.D.resize(ng, nu);
    stage.lg.resize(ng);
    stage.ug.resize(ng);
    stage.lg_mask.resize(ng);
    stage.ug_mask.resize(ng);
}

void hpipm_qp::_build()
{
    const uint horizon = _qp.dims.horizon;
    _stages.assign(horizon + 1, Stage());

    // x_0 is fixed by a box on every entry, which its state rows join
    _split_rows(0, horizon > 0 ? _qp.dims.stage_control[0] : 0, true);
    for (uint k = 1; k < horizon; ++k)
    {
        _split_rows(k, _qp.dims.stage_control[k], false);
    }
    if (horizon > 0)
    {
        _split_rows(horizon, 0, false);
    }

    size_t max_rows = 0;
    for (const Stage &stage : _stages)
    {
        max_rows = std::max({ max_rows,
                              stage.general_rows.size(),
                              stage.x.idx.size(),
                              stage.u.idx.size() });
    }
    _lam.resize(2 * max_rows);

    hpipm_size_t dim_size = d_ocp_qp_dim_memsize(horizon);
    _hpipm_dim_mem        = malloc(dim_size);
    d_ocp_qp_dim_create(horizon, &_hpipm_data->dim, _hpipm_dim_mem);

    set_stage_sizes();

    // Allocate QP problem structure
    hpipm_size_t qp_size = d_ocp_qp_memsize(&_hpipm_data->dim);
    _hpipm_qp_mem        = malloc(qp_size);
    d_ocp_qp_create(&_hpipm_data->dim, &_hpipm_data->qp, _hpipm_qp_mem);

    // The box entries only change with the layout
    for (uint k = 0; k <= horizon; ++k)
    {
        if (!_stages[k].x.idx.empty())
        {
            d_ocp_qp_set_idxbx(k, _stages[k].x.idx.data(), &_hpipm_data->qp);
        }
        if (!_stages[k].u.idx.empty())
        {
            d_ocp_qp_set_idxbu(k, _stages[k].u.idx.data(), &_hpipm_data->qp);
        }
    }

    // Allocate solution structure
    hpipm_size_t sol_size = d_ocp_qp_sol_memsize(&_hpipm_data->dim);
    _hpipm_sol_mem        = malloc(sol_size);
    d_ocp_qp_sol_create(&_hpipm_data->dim, &_hpipm_data->sol, _hpipm_sol_mem);

    // Allocate IPM arguments
    hpipm_size_t arg_size = d_ocp_qp_ipm_arg_memsize(&_hpipm_data->dim);
    _hpipm_arg_mem        = malloc(arg_size);
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
    d_ocp_qp_ipm_arg_set_ric_alg(&square_root_riccati_int, &_hpipm_data->arg);

    int split_step_int = _params.split_step ? 1 : 0;
    d_ocp_qp_ipm_arg_set_split_step(&split_step_int, &_hpipm_data->arg);

    // Allocate _hpipm_data->ws (size depends on arg settings)
    hpipm_size_t ws_size =
      d_ocp_qp_ipm_ws_memsize(&_hpipm_data->dim, &_hpipm_data->arg);
    _hpipm_ws_mem = malloc(ws_size);
    d_ocp_qp_ipm_ws_create(&_hpipm_data->dim,
                           &_hpipm_data->arg,
                           &_hpipm_data->ws,
                           _hpipm_ws_mem);
}

void hpipm_qp::_free()
{
    if (_hpipm_ws_mem) free(_hpipm_ws_mem);
    if (_hpipm_arg_mem) free(_hpipm_arg_mem);
    if (_hpipm_sol_mem) free(_hpipm_sol_mem);
    if (_hpipm_qp_mem) free(_hpipm_qp_mem);
    if (_hpipm_dim_mem) free(_hpipm_dim_mem);

    _hpipm_ws_mem  = nullptr;
    _hpipm_arg_mem = nullptr;
    _hpipm_sol_mem = nullptr;
    _hpipm_qp_mem  = nullptr;
    _hpipm_dim_mem = nullptr;
}

void hpipm_qp::set_stage_sizes()
{
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_dim_set_nx(k, _qp.dims.stage_state[k], &_hpipm_data->dim);
        d_ocp_qp_dim_set_nu(k, _qp.dims.stage_control[k], &_hpipm_data->dim);
        d_ocp_qp_dim_set_nbx(k, _stages[k].x.idx.size(), &_hpipm_data->dim);
        d_ocp_qp_dim_set_nbu(k, _stages[k].u.idx.size(), &_hpipm_data->dim);
        d_ocp_qp_dim_set_ng(k,
                            _stages[k].general_rows.size(),
                            &_hpipm_data->dim);
    }

    const uint horizon = _qp.dims.horizon;
    d_ocp_qp_dim_set_nx(horizon,
                        _qp.dims.stage_state[horizon],
                        &_hpipm_data->dim);
    d_ocp_qp_dim_set_nbx(horizon,
                         _stages[horizon].x.idx.size(),
                         &_hpipm_data->dim);
    d_ocp_qp_dim_set_ng(horizon,
                        _stages[horizon].general_rows.size(),
                        &_hpipm_data->dim);
}

void hpipm_qp::updateDimensions()
{
    // The HPIPM structures are sized by the dimensions, so the next reset
    // creates new ones
    _free();
    _solved      = false;
    _qp_uploaded = false;

    _solution.resize(_qp.dims);

    _S.resize(_qp.dims.horizon);
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _S[k].resize(_qp.dims.stage_control[k], _qp.dims.stage_state[k]);
    }
}

void hpipm_qp::reset()
{
    // The HPIPM structures are sized by the box layout, so a new layout needs
    // new ones
    if (_hpipm_dim_mem == nullptr || _qp.box_rows != _box_rows)
    {
        _free();
        _box_rows = _qp.box_rows;
        _build();
        _solved = false;
    }

    _reg = _params.reg_prim;
    d_ocp_qp_ipm_arg_set_reg_prim(&_reg, &_hpipm_data->arg);

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
        // QPStage stores Qxu as nx-by-nu, while HPIPM stores S as nu-by-nx.
        // The transpose is required once a nonzero state/control cross term
        // is present at a stage with a nonzero state.
        _S[k].noalias() = _qp.stages[k].Qxu.transpose();
        d_ocp_qp_set_S(k, (double *)(_S[k].data()), &_hpipm_data->qp);
        d_ocp_qp_set_R(k,
                       (double *)(_qp.stages[k].Quu.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_q(k,
                       (double *)(_qp.stages[k].qx.data()),
                       &_hpipm_data->qp);
        d_ocp_qp_set_r(k,
                       (double *)(_qp.stages[k].qu.data()),
                       &_hpipm_data->qp);

        _upload_constraints(k,
                            _qp.stages[k].D,
                            &_qp.stages[k].E,
                            _qp.stages[k].cl,
                            _qp.stages[k].cu);
    }

    d_ocp_qp_set_Q(_qp.dims.horizon,
                   (double *)(_qp.Qxx_t.data()),
                   &_hpipm_data->qp);
    d_ocp_qp_set_q(_qp.dims.horizon,
                   (double *)(_qp.qx_t.data()),
                   &_hpipm_data->qp);

    _upload_constraints(_qp.dims.horizon, _qp.Dt, nullptr, _qp.cl_t, _qp.cu_t);

    _qp_uploaded = true;
}

void hpipm_qp::Boxes::layout(const std::vector<model::BoxRow> &box_rows,
                             const bool control, const uint n)
{
    // The number of rows on each entry, later the position of its next row
    std::vector<uint> next(n, 0);
    for (const model::BoxRow &box : box_rows)
    {
        if (box.is_control == control)
        {
            ++next[box.index];
        }
    }

    uint nb = 0;
    for (uint i = 0; i < n; ++i)
    {
        if (fixed || next[i] > 0)
        {
            ++nb;
        }
    }

    idx.reserve(nb);
    start.reserve(nb + 1);
    start.push_back(0);
    for (uint i = 0; i < n; ++i)
    {
        if (fixed || next[i] > 0)
        {
            const uint first = start.back();
            idx.push_back(i);
            start.push_back(first + next[i]);
            next[i] = first;
        }
    }

    // Each entry's rows keep their declared order, which decides ties
    rows.resize(start.back());
    for (const model::BoxRow &box : box_rows)
    {
        if (box.is_control == control)
        {
            rows[next[box.index]++] = box.row;
        }
    }

    lb_row.resize(nb);
    ub_row.resize(nb);
    lb.resize(nb);
    ub.resize(nb);
    lb_mask.resize(nb);
    ub_mask.resize(nb);
}

void hpipm_qp::Boxes::merge(const VecX &cl, const VecX &cu)
{
    const double inf   = std::numeric_limits<double>::infinity();
    const double bound = fixed ? 0.0 : inf;

    for (uint j = 0; j < idx.size(); ++j)
    {
        // The first of equally tight rows gives the side, and fixed entries
        // keep it unless a row is tighter
        double lower = -bound, upper = bound;
        lb_row[j] = -1;
        ub_row[j] = -1;
        for (uint r = start[j]; r < start[j + 1]; ++r)
        {
            const uint row = rows[r];
            if (cl[row] > lower)
            {
                lower     = cl[row];
                lb_row[j] = row;
            }
            if (cu[row] < upper)
            {
                upper     = cu[row];
                ub_row[j] = row;
            }
        }

        if (std::isinf(lower))
        {
            lb[j]      = 0.0;
            lb_mask[j] = 0.0;
        }
        else
        {
            lb[j]      = lower;
            lb_mask[j] = 1.0;
        }

        if (std::isinf(upper))
        {
            ub[j]      = 0.0;
            ub_mask[j] = 0.0;
        }
        else
        {
            ub[j]      = upper;
            ub_mask[j] = 1.0;
        }
    }
}

void hpipm_qp::Boxes::multipliers(const double *lam_lb, const double *lam_ub,
                                  VecXRef mu) const
{
    // Rows that give neither side of their box are inactive or tied with the
    // ones that do, so their multipliers are zero
    for (uint j = 0; j < idx.size(); ++j)
    {
        for (uint r = start[j]; r < start[j + 1]; ++r)
        {
            mu[rows[r]] = 0.0;
        }

        if (lb_row[j] >= 0)
        {
            mu[lb_row[j]] += lam_lb[j];
        }
        if (ub_row[j] >= 0)
        {
            mu[ub_row[j]] -= lam_ub[j];
        }
    }
}

void hpipm_qp::_upload_constraints(const uint k, const MatX &D, const MatX *E,
                                   const VecX &cl, const VecX &cu)
{
    Stage &stage = _stages[k];

#ifndef NDEBUG
    for (uint j = 0; j < stage.x.idx.size(); ++j)
    {
        for (uint r = stage.x.start[j]; r < stage.x.start[j + 1]; ++r)
        {
            const uint row = stage.x.rows[r];
            assert(D.row(row).transpose() ==
                     VecX::Unit(D.cols(), stage.x.idx[j]) &&
                   (E == nullptr || E->row(row).isZero(0)) &&
                   "State box row is not +1 on its entry and zero elsewhere");
        }
    }
    for (uint j = 0; j < stage.u.idx.size(); ++j)
    {
        for (uint r = stage.u.start[j]; r < stage.u.start[j + 1]; ++r)
        {
            const uint row = stage.u.rows[r];
            assert(D.row(row).isZero(0) &&
                   E->row(row).transpose() ==
                     VecX::Unit(E->cols(), stage.u.idx[j]) &&
                   "Control box row is not +1 on its entry and zero elsewhere");
        }
    }
#endif

    stage.x.merge(cl, cu);
    if (!stage.x.idx.empty())
    {
        d_ocp_qp_set_lbx(k, stage.x.lb.data(), &_hpipm_data->qp);
        d_ocp_qp_set_ubx(k, stage.x.ub.data(), &_hpipm_data->qp);
        d_ocp_qp_set_lbx_mask(k, stage.x.lb_mask.data(), &_hpipm_data->qp);
        d_ocp_qp_set_ubx_mask(k, stage.x.ub_mask.data(), &_hpipm_data->qp);
    }

    stage.u.merge(cl, cu);
    if (!stage.u.idx.empty())
    {
        d_ocp_qp_set_lbu(k, stage.u.lb.data(), &_hpipm_data->qp);
        d_ocp_qp_set_ubu(k, stage.u.ub.data(), &_hpipm_data->qp);
        d_ocp_qp_set_lbu_mask(k, stage.u.lb_mask.data(), &_hpipm_data->qp);
        d_ocp_qp_set_ubu_mask(k, stage.u.ub_mask.data(), &_hpipm_data->qp);
    }

    const uint ng = stage.general_rows.size();
    if (ng == 0)
    {
        return;
    }

    for (uint g = 0; g < ng; ++g)
    {
        const uint row = stage.general_rows[g];
        stage.C.row(g) = D.row(row);

        if (std::isinf(cl[row]))
        {
            stage.lg[g]      = 0.0;
            stage.lg_mask[g] = 0.0;
        }
        else
        {
            stage.lg[g]      = cl[row];
            stage.lg_mask[g] = 1.0;
        }

        if (std::isinf(cu[row]))
        {
            stage.ug[g]      = 0.0;
            stage.ug_mask[g] = 0.0;
        }
        else
        {
            stage.ug[g]      = cu[row];
            stage.ug_mask[g] = 1.0;
        }
    }

    d_ocp_qp_set_C(k, stage.C.data(), &_hpipm_data->qp);
    if (E != nullptr)
    {
        for (uint g = 0; g < ng; ++g)
        {
            stage.D.row(g) = E->row(stage.general_rows[g]);
        }
        d_ocp_qp_set_D(k, stage.D.data(), &_hpipm_data->qp);
    }

    d_ocp_qp_set_lg(k, stage.lg.data(), &_hpipm_data->qp);
    d_ocp_qp_set_ug(k, stage.ug.data(), &_hpipm_data->qp);
    d_ocp_qp_set_lg_mask(k, stage.lg_mask.data(), &_hpipm_data->qp);
    d_ocp_qp_set_ug_mask(k, stage.ug_mask.data(), &_hpipm_data->qp);
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
    if (!_qp_uploaded)
    {
        reset();
    }

    d_ocp_qp_ipm_solve(&_hpipm_data->qp,
                       &_hpipm_data->sol,
                       &_hpipm_data->arg,
                       &_hpipm_data->ws);
    _solved = true;

    d_ocp_qp_ipm_get_status(&_hpipm_data->ws, &_status);
    d_ocp_qp_ipm_get_iter(&_hpipm_data->ws, &_iter);

    if (GlobalLogMask & LogLevel::Warning && _status != 0)
    {
        logln_at(LogLevel::Warning,
                 "Warning: HPIPM returned status " << _status);
    }

    logln_at(LogLevel::Debug, "HPIPM status: " << _status);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        d_ocp_qp_sol_get_x(k, &_hpipm_data->sol, sol.dx.stage(k).data());
        d_ocp_qp_sol_get_u(k, &_hpipm_data->sol, sol.du.stage(k).data());
        d_ocp_qp_sol_get_pi(k, &_hpipm_data->sol, sol.lambda.stage(k).data());

        _get_multipliers(k, sol);
    }

    d_ocp_qp_sol_get_x(_qp.dims.horizon,
                       &_hpipm_data->sol,
                       sol.dx.stage(_qp.dims.horizon).data());

    _get_multipliers(_qp.dims.horizon, sol);

    return SUCCESS;
}

void hpipm_qp::_get_multipliers(const uint k, Solution &sol) const
{
    const Stage &stage = _stages[k];
    VecXRef      mu    = sol.mu.stage(k);

    const uint ng    = stage.general_rows.size();
    double    *lower = _lam.data();
    double    *upper = lower + _lam.size() / 2;

    d_ocp_qp_sol_get_lam_lg(k, &_hpipm_data->sol, lower);
    d_ocp_qp_sol_get_lam_ug(k, &_hpipm_data->sol, upper);
    for (uint g = 0; g < ng; ++g)
    {
        mu[stage.general_rows[g]] = lower[g] - upper[g];
    }

    d_ocp_qp_sol_get_lam_lbx(k, &_hpipm_data->sol, lower);
    d_ocp_qp_sol_get_lam_ubx(k, &_hpipm_data->sol, upper);
    stage.x.multipliers(lower, upper, mu);

    d_ocp_qp_sol_get_lam_lbu(k, &_hpipm_data->sol, lower);
    d_ocp_qp_sol_get_lam_ubu(k, &_hpipm_data->sol, upper);
    stage.u.multipliers(lower, upper, mu);
}

uint hpipm_qp::iterations() const
{
    return _iter;
}

MatX hpipm_qp::feedbackGain(const uint k) const
{
    if (k >= _qp.dims.horizon)
    {
        throw std::out_of_range("HPIPM: No feedback gain for stage " +
                                std::to_string(k));
    }

    if (!_solved)
    {
        throw std::runtime_error("HPIPM: No feedback gain before solving");
    }

    // HPIPM writes the gain column-major, as Eigen stores it
    MatX K(_qp.dims.stage_control[k], _qp.dims.stage_state[k]);
    d_ocp_qp_ipm_get_ric_K(&_hpipm_data->qp,
                           &_hpipm_data->arg,
                           &_hpipm_data->ws,
                           k,
                           K.data());

    return K;
}

}
}
#endif
