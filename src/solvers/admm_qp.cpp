#include "solvers/admm_qp.hpp"
#include "solvers/lqr.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace solvers
{
admm_qp::Params::Params(const nlohmann::ordered_json &j)
{
    using namespace nlohmann;

    max_iter               = j["max_iter"];
    abs_tolerance          = j["abs_tolerance"];
    abs_tolerance_final    = j["abs_tolerance_final"];
    rel_tolerance          = j["rel_tolerance"];
    rel_tolerance_final    = j["rel_tolerance_final"];
    tolerance_update_steps = j["tolerance_update_steps"];

    init_rho = j["init_rho"];
    alpha    = j["alpha"];
    sigma    = j["sigma"];

    rho_update             = j["rho_update"];
    rho_min                = j["rho_min"];
    rho_max                = j["rho_max"];
    adaptive_rho_tolerance = j["adaptive_rho_tolerance"];
    rho_update_iter_freq   = j["rho_update_iter_freq"];

    warm_start = j["warm_start"];
    reset_rho  = j["reset_rho"];
    reset_y    = j["reset_y"];
}

void admm_qp::Params::print() const
{
    std::cout << "ADMM Parameters" << std::endl;
    std::cout << "===============" << std::endl;
    std::cout << "Maximum iterations: " << max_iter << std::endl;
    std::cout << "Absolute tolerance: " << abs_tolerance << std::endl;

    if (abs_tolerance_final > 0.0)
    {
        std::cout << "Final absolute tolerance: " << abs_tolerance_final
                  << std::endl;
    }

    std::cout << "Relative tolerance: " << rel_tolerance << std::endl;

    if (rel_tolerance_final > 0.0)
    {
        std::cout << "Final relative tolerance: " << rel_tolerance_final
                  << std::endl;
    }

    if (abs_tolerance_final > 0.0 || rel_tolerance_final > 0.0)
    {
        std::cout << "Update steps: " << tolerance_update_steps << std::endl;
    }

    std::cout << "---------------" << std::endl;
    std::cout << "Init rho: " << init_rho << std::endl;
    std::cout << "Alpha: " << alpha << std::endl;
    std::cout << "Sigma: " << sigma << std::endl;

    std::cout << "---------------" << std::endl;
    std::cout << "Rho update enabled: " << rho_update << std::endl;
    std::cout << "Minimum rho: " << rho_min << std::endl;
    std::cout << "Maximum rho: " << rho_max << std::endl;
    std::cout << "Adaptive rho tolerance: " << adaptive_rho_tolerance
              << std::endl;
    std::cout << "Rho update iteration frequency: " << rho_update_iter_freq
              << std::endl;

    std::cout << "---------------" << std::endl;
    std::cout << "Warm start enabled: " << warm_start << std::endl;
    std::cout << "Reset rho enabled: " << reset_rho << std::endl;
    std::cout << "Reset y enabled: " << reset_y << std::endl;
    std::cout << "---------------" << std::endl;
}

void admm_qp::Params::write(nlohmann::ordered_json &j) const
{
    using namespace nlohmann;

    ordered_json admmj;

    admmj["max_iter"]               = max_iter;
    admmj["abs_tolerance"]          = abs_tolerance;
    admmj["abs_tolerance_final"]    = abs_tolerance_final;
    admmj["rel_tolerance"]          = rel_tolerance;
    admmj["rel_tolerance_final"]    = rel_tolerance_final;
    admmj["tolerance_update_steps"] = tolerance_update_steps;

    admmj["init_rho"] = init_rho;
    admmj["alpha"]    = alpha;
    admmj["sigma"]    = sigma;

    admmj["rho_update"]             = rho_update;
    admmj["rho_min"]                = rho_min;
    admmj["rho_max"]                = rho_max;
    admmj["adaptive_rho_tolerance"] = adaptive_rho_tolerance;
    admmj["rho_update_iter_freq"]   = rho_update_iter_freq;

    admmj["warm_start"] = warm_start;
    admmj["reset_rho"]  = reset_rho;
    admmj["reset_y"]    = reset_y;

    j["admm"] = admmj;
}

admm_qp::admm_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _solution(qp.dims)
  , _z(_solution.mu, false)
  , _y(_solution.mu, false)
  , _rho_common(params.init_rho)
  , _rho(_z, false)
  , _rho_inv(_z, false)
  , _reg_incr(0)
  , _abs_tolerance_init(_params.abs_tolerance)
  , _rel_tolerance_init(_params.rel_tolerance)
{
    _rho.vec.setConstant(_params.init_rho);
    _rho_inv.vec.setConstant(1 / _params.init_rho);

    uint max_x =
      *std::max_element(qp.dims.stage_state.begin(), qp.dims.stage_state.end());
    uint max_u = *std::max_element(qp.dims.stage_control.begin(),
                                   qp.dims.stage_control.end());
    uint max_c = *std::max_element(qp.dims.stage_constraint.begin(),
                                   qp.dims.stage_constraint.end());

    _tmp.x1.resize(max_x);
    _tmp.u1.resize(max_u);
    _tmp.c1.resize(max_c);
    _tmp.c2.resize(max_c);
}

void admm_qp::reset()
{
    if (_params.reset_rho)
    {
        _rho_common = _params.init_rho;
        _rho.vec.setConstant(_params.init_rho);
        _rho_inv.vec.setConstant(1 / _params.init_rho);
    }

    if (_params.reset_y)
    {
        _y.vec.setZero();
    }

    _z.vec.setZero();

    _reg_incr = 0;
}

void admm_qp::increaseRegularization()
{
    ++_reg_incr;
}

SolverState admm_qp::solve(admm_qp::Solution &sol)
{
    ZoneScopedN("ADMM: solve");

    logln_at(LogLevel::Debug, "Solving with ADMM");
    logln_at(LogLevel::Debug, "=============");

    SolverState admm_state;

    // Warm start with unconstrained LQR
    lqr::Params lqr_params;
    lqr_params.sigma     = _params.sigma;
    lqr_params.reg_init *= std::pow(10, _reg_incr);
    if (lqr_params.reg_init > lqr_params.reg_max)
    {
        logln_at(LogLevel::Error, "ADMM: Initial LQR regularization failed");
        throw std::runtime_error("LQR: Regularization failed");
    }

    lqr lqr_solver(_qp, _solution, _rho, lqr_params);

    lqr::Output &lqr_out = lqr_solver.output_structure();

    if (_params.warm_start)
    {
        lqr_out      = lqr_solver.unconstrained();
        _solution.dx = lqr_out.x;
        _solution.du = lqr_out.u;
        logln_at(LogLevel::Debug, "LQR unconstrained solution");
        logln_at(LogLevel::Debug, "=============");
        logln_at(LogLevel::Debug, "x:");
        logln_at(LogLevel::Debug, lqr_out.x);
        logln_at(LogLevel::Debug, "u:");
        logln_at(LogLevel::Debug, lqr_out.u);
        logln_at(LogLevel::Debug, "=============");
    }

    _params.abs_tolerance = _abs_tolerance_init;
    _params.rel_tolerance = _rel_tolerance_init;

    // With no inequality constraints the LQR solve is already the complete
    // QP solution. Applying the ADMM over-relaxation step would otherwise
    // scale the unconstrained solution by alpha before the residual checks,
    // which are intentionally inactive in this case.
    if (_qp.dims.constraint == 0)
    {
        lqr_out      = lqr_solver.unconstrained();
        _solution.dx = lqr_out.x;
        _solution.du = lqr_out.u;
        _solution.mu.vec.setZero();
        for (uint k = 0; k < _qp.dims.horizon; ++k)
        {
            _solution.lambda.stage(k).noalias() =
              lqr_out.P[k + 1] * _solution.dx.stage(k + 1);
            _solution.lambda.stage(k) += lqr_out.p[k + 1];
        }
        _iter = 1;
        sol   = _solution;
        return SUCCESS;
    }

    _rho_changed = true;
    for (_iter = 0;; ++_iter)
    {
        logln_at(LogLevel::Debug, "ADMM Iteration: " << _iter);
        if (_rho_changed)
        {
            lqr_out      = lqr_solver.constrained(_solution, _z, _y, _rho);
            _rho_changed = false;
        }
        else
        {
            lqr_out = lqr_solver.constrained(_solution, _z, _y);
        }

        logln_at(LogLevel::Debug, "LQR solution, iteration " << _iter + 1);
        logln_at(LogLevel::Debug, "=============");
        logln_at(LogLevel::Debug, "x:");
        logln_at(LogLevel::Debug, lqr_out.x);
        logln_at(LogLevel::Debug, "u:");
        logln_at(LogLevel::Debug, lqr_out.u);
        logln_at(LogLevel::Debug, "=============");

        FlatVec x_prev = _solution.dx, u_prev = _solution.du, z_prev = _z,
                y_prev = _y;

        _step(lqr_out.x, lqr_out.u);

        logln_at(LogLevel::Debug, "Update step, iteration " << _iter + 1);
        logln_at(LogLevel::Debug, "=============");
        logln_at(LogLevel::Debug, "x:");
        logln_at(LogLevel::Debug, _solution.dx);
        logln_at(LogLevel::Debug, "u:");
        logln_at(LogLevel::Debug, _solution.du);
        logln_at(LogLevel::Debug, "z:");
        logln_at(LogLevel::Debug, _z);
        logln_at(LogLevel::Debug, "y:");
        logln_at(LogLevel::Debug, _y);
        logln_at(LogLevel::Debug, "=============");
        log_wait_at(LogLevel::Debug);

        _calculate_residuals(x_prev, u_prev, z_prev, y_prev);

        // Termination
        if ((admm_state = _check_termination()) != ONGOING)
        {
            break;
        }

        if (_params.rho_update && !((_iter) % _params.rho_update_iter_freq))
        {
            _update_rho();
        }

        if (_params.abs_tolerance_final > 0.0 &&
            _params.abs_tolerance > _params.abs_tolerance_final)
        {
            _params.abs_tolerance +=
              (_params.abs_tolerance_final - _abs_tolerance_init) /
              (_params.tolerance_update_steps == 0
                 ? _params.max_iter
                 : _params.tolerance_update_steps);
        }

        if (_params.rel_tolerance_final > 0.0 &&
            _params.rel_tolerance > _params.rel_tolerance_final)
        {
            _params.rel_tolerance +=
              (_params.rel_tolerance_final - _rel_tolerance_init) /
              (_params.tolerance_update_steps == 0
                 ? _params.max_iter
                 : _params.tolerance_update_steps);
        }

        FrameMarkNamed("ADMM Iteration");
    }

    sol = _solution;

    // Lagrange multiplier update
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        sol.lambda.stage(k).noalias()  = lqr_out.P[k + 1] * sol.dx.stage(k + 1);
        sol.lambda.stage(k)           += lqr_out.p[k + 1];
    }
    // Need a minus because the QP (ADMM) assumes the opposite sign in the
    // Lagrangian
    sol.mu = -_y;

    return admm_state;
}

uint admm_qp::iterations() const
{
    return _iter;
}

void admm_qp::_calculate_residuals(const FlatVec &x_prev, const FlatVec &u_prev,
                                   const FlatVec &z_prev, const FlatVec &y_prev)
{
    ZoneScopedN("ADMM: _calculate_residuals");

    const uint H = _qp.dims.horizon;

    _norm_primal     = -std::numeric_limits<double>::infinity();
    _norm_dual       = -std::numeric_limits<double>::infinity();
    _norm_primal_rel = -std::numeric_limits<double>::infinity();
    _norm_dual_rel   = -std::numeric_limits<double>::infinity();

    _primal_infs_norm       = -std::numeric_limits<double>::infinity();
    _primal_infs_support    = 0;
    _primal_infs_tol_scaled = _params.primal_infs_tolerance;

    for (int k = 0; k < H; ++k)
    {
        const auto &stage = _qp.stages[k];

        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];
        const uint nc = stage.D.rows();

        if (nc == 0) continue;

        const auto &D = stage.D;
        const auto &E = stage.E;

        auto c1 = _tmp.c1.head(nc); // zdiff_scaled / workspace
        auto c2 = _tmp.c2.head(nc); // CdxCdu

        auto x1 = _tmp.x1.head(nx);
        auto u1 = _tmp.u1.head(nu);

        // =========================
        // zdiff_scaled = rho ∘ (z - z_prev)
        // =========================
        c1          = _z.stage(k);
        c1         -= z_prev.stage(k);
        c1.array() *= _rho.stage(k).array();

        // dual residual
        x1.noalias() = D.transpose() * c1;
        _norm_dual   = std::max(_norm_dual, x1.lpNorm<Eigen::Infinity>());

        u1.noalias() = E.transpose() * c1;
        _norm_dual   = std::max(_norm_dual, u1.lpNorm<Eigen::Infinity>());

        // =========================
        // CdxCdu = D dx + E du
        // =========================
        c2.noalias()  = D * _solution.dx.stage(k);
        c2.noalias() += E * _solution.du.stage(k);

        // primal residual
        c1  = c2;
        c1 -= _z.stage(k);

        _norm_primal = std::max(_norm_primal, c1.lpNorm<Eigen::Infinity>());

        // primal relative
        _norm_primal_rel =
          std::max(_norm_primal_rel, c2.lpNorm<Eigen::Infinity>());

        _norm_primal_rel =
          std::max(_norm_primal_rel, _z.stage(k).lpNorm<Eigen::Infinity>());

        // =========================
        // dual relative
        // =========================
        x1.noalias()   = D.transpose() * _y.stage(k);
        _norm_dual_rel = std::max(_norm_dual_rel, x1.lpNorm<Eigen::Infinity>());

        u1.noalias()   = E.transpose() * _y.stage(k);
        _norm_dual_rel = std::max(_norm_dual_rel, u1.lpNorm<Eigen::Infinity>());
    }

    // =========================
    // Terminal stage
    // =========================
    if (_qp.Dt.rows() == 0) return;

    const uint nx = _qp.dims.stage_state[H];
    const uint nc = _qp.Dt.rows();

    const auto &Dt = _qp.Dt;

    auto c1 = _tmp.c1.head(nc);
    auto c2 = _tmp.c2.head(nc);
    auto x1 = _tmp.x1.head(nx);

    // zdiff_scaled
    c1          = _z.stage(H);
    c1         -= z_prev.stage(H);
    c1.array() *= _rho.stage(H).array();

    x1.noalias() = Dt.transpose() * c1;
    _norm_dual   = std::max(_norm_dual, x1.lpNorm<Eigen::Infinity>());

    // Cdx
    c2.noalias() = Dt * _solution.dx.stage(H);

    // primal residual
    c1  = c2;
    c1 -= _z.stage(H);

    _norm_primal = std::max(_norm_primal, c1.lpNorm<Eigen::Infinity>());

    // primal relative
    _norm_primal_rel = std::max(_norm_primal_rel, c2.lpNorm<Eigen::Infinity>());

    _norm_primal_rel =
      std::max(_norm_primal_rel, _z.stage(H).lpNorm<Eigen::Infinity>());

    // dual relative
    x1.noalias()   = Dt.transpose() * _y.stage(H);
    _norm_dual_rel = std::max(_norm_dual_rel, x1.lpNorm<Eigen::Infinity>());
}

SolverState admm_qp::_check_termination() const
{
    ZoneScopedN("ADMM: _check_termination");

    if (_iter >= _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    // logln_at(LogLevel::Info, "Primal infeasibility norm vs tolerance: "
    //       << _primal_infs_norm << " vs " << _primal_infs_tol_scaled);
    // logln_at(LogLevel::Info, "Primal infeasibility support vs tolerance: "
    //       << _primal_infs_support << " vs " << _primal_infs_tol_scaled);

    // TODO: Primal infeasibility detection doesn't work yet
    // if (_primal_infs_norm < _primal_infs_tol_scaled &&
    //     _primal_infs_support < _primal_infs_tol_scaled)
    // {
    //     return PRIMAL_INFEASIBLE;
    // }

    logln_at(LogLevel::Debug, "rho:" << std::endl << _rho);

    double tolerance_prim =
      _params.abs_tolerance + _params.rel_tolerance * _norm_primal_rel;

    double tolerance_dual =
      _params.abs_tolerance + _params.rel_tolerance * _norm_dual_rel;

    logln_at(LogLevel::Debug, "Tolerances");
    logln_at(LogLevel::Debug, "=============");
    logln_at(LogLevel::Debug,
             "Primal (tol vs norm): " << tolerance_prim << " vs "
                                      << _norm_primal);
    logln_at(LogLevel::Debug,
             "Dual (tol vs norm): " << tolerance_dual << " vs " << _norm_dual);
    logln_at(LogLevel::Debug, "=============");

    if (_norm_primal > tolerance_prim || _norm_dual > tolerance_dual)
    {
        return ONGOING;
    }

    return SUCCESS;
}

void admm_qp::_update_rho()
{
    ZoneScopedN("ADMM: _update_rho");

    const double scale = std::sqrt((_norm_primal * _norm_dual_rel) /
                                   (_norm_dual * _norm_primal_rel));

    double rho_estimate_sparse =
      std::min(std::max(scale * _rho_common, _params.rho_min), _params.rho_max);

    if (rho_estimate_sparse > _rho_common * _params.adaptive_rho_tolerance ||
        rho_estimate_sparse < _rho_common / _params.adaptive_rho_tolerance)
    {
        _rho_common = rho_estimate_sparse;
        for (uint k = 0; k < _qp.dims.horizon; ++k)
        {
            for (uint c = 0; c < _qp.dims.stage_constraint[k]; ++c)
            {
                if (_qp.stages[k].cl(c) ==
                      -std::numeric_limits<double>::infinity() &&
                    _qp.stages[k].cu(c) ==
                      std::numeric_limits<double>::infinity())
                {
                    _rho.stage(k)(c)     = _rho_min;
                    _rho_inv.stage(k)(c) = 1.0 / _rho_min;
                }
                else if (std::abs(_qp.stages[k].cl(c) - _qp.stages[k].cu(c)) <
                         std::numeric_limits<double>::epsilon())
                {
                    _rho.stage(k)(c)     = 1e3 * _rho_common;
                    _rho_inv.stage(k)(c) = 1.0 / (1e3 * _rho_common);
                }
                else
                {
                    _rho.stage(k)(c)     = _rho_common;
                    _rho_inv.stage(k)(c) = 1.0 / _rho_common;
                }
            }
        }

        for (uint c = 0; c < _qp.dims.stage_constraint[_qp.dims.horizon]; ++c)
        {
            if (_qp.cl_t(c) == -std::numeric_limits<double>::infinity() &&
                _qp.cu_t(c) == std::numeric_limits<double>::infinity())
            {
                _rho.stage(_qp.dims.horizon)(c)     = _rho_min;
                _rho_inv.stage(_qp.dims.horizon)(c) = 1.0 / _rho_min;
            }
            else if (std::abs(_qp.cl_t(c) - _qp.cu_t(c)) <
                     std::numeric_limits<double>::epsilon())
            {
                _rho.stage(_qp.dims.horizon)(c)     = 1e3 * _rho_common;
                _rho_inv.stage(_qp.dims.horizon)(c) = 1.0 / (1e3 * _rho_common);
            }
            else
            {
                _rho.stage(_qp.dims.horizon)(c)     = _rho_common;
                _rho_inv.stage(_qp.dims.horizon)(c) = 1.0 / _rho_common;
            }
        }

        _rho_changed = true;
    }
}

void admm_qp::_step(const FlatVec &dx_hat, const FlatVec &du_hat)
{
    ZoneScopedN("ADMM: _step");

    const uint   H     = _qp.dims.horizon;
    const double alpha = _params.alpha;

    for (int k = 0; k < H; ++k)
    {
        const auto &stage = _qp.stages[k];

        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];
        const uint nc = stage.cl.size();

        // dx update
        _solution.dx.stage(k)           *= (1.0 - alpha);
        _solution.dx.stage(k).noalias() += alpha * dx_hat.stage(k);

        // du update
        _solution.du.stage(k)           *= (1.0 - alpha);
        _solution.du.stage(k).noalias() += alpha * du_hat.stage(k);

        if (nc == 0) continue;

        const auto &D = stage.D; // nc x nx
        const auto &E = stage.E; // nc x nu

        auto c1 = _tmp.c1.head(nc); // z_hat
        auto c2 = _tmp.c2.head(nc); // workspace

        // =========================
        // z_hat = alpha*(D dx + E du) + (1-alpha)*z
        // =========================

        // c1 = D * dx_hat
        c1.noalias() = D * dx_hat.stage(k);

        // c1 += E * du_hat
        c1.noalias() += E * du_hat.stage(k);

        // scale by alpha
        c1 *= alpha;

        // add (1-alpha) * z
        c1.noalias() += (1.0 - alpha) * _z.stage(k);

        // =========================
        // projection:
        // z = clamp(z_hat + rho_inv ∘ y, cl, cu)
        // =========================

        // c2 = rho_inv ∘ y
        c2          = _rho_inv.stage(k);
        c2.array() *= _y.stage(k).array();

        // c2 = z_hat + rho_inv ∘ y
        c2 += c1;

        // c2 = max(cl, c2)
        c2 = c2.cwiseMax(stage.cl);

        // z = min(cu, c2)
        _z.stage(k) = c2.cwiseMin(stage.cu);

        // =========================
        // dual update:
        // y += rho ∘ (z_hat - z)
        // =========================

        // c2 = z_hat - z
        c2  = c1;
        c2 -= _z.stage(k);

        // c2 = rho ∘ c2
        c2.array() *= _rho.stage(k).array();

        _y.stage(k) += c2;
    }

    // =========================
    // Terminal dx update
    // =========================
    _solution.dx.stage(H)           *= (1.0 - alpha);
    _solution.dx.stage(H).noalias() += alpha * dx_hat.stage(H);

    // =========================
    // Terminal constraints
    // =========================
    if (_qp.cl_t.size() == 0) return;

    const uint nc = _qp.cl_t.size();

    auto c1 = _tmp.c1.head(nc); // z_hat
    auto c2 = _tmp.c2.head(nc);

    // z_hat = alpha * Dt * dx + (1-alpha)*z
    c1.noalias()  = _qp.Dt * dx_hat.stage(H);
    c1           *= alpha;
    c1.noalias() += (1.0 - alpha) * _z.stage(H);

    // projection

    // c2 = rho_inv ∘ y
    c2          = _rho_inv.stage(H);
    c2.array() *= _y.stage(H).array();

    // c2 = z_hat + ...
    c2 += c1;

    // clamp
    c2          = c2.cwiseMax(_qp.cl_t);
    _z.stage(H) = c2.cwiseMin(_qp.cu_t);

    // dual update

    c2          = c1;
    c2         -= _z.stage(H);
    c2.array() *= _rho.stage(H).array();

    _y.stage(H) += c2;
}
}
}
