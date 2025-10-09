#include "solvers/admm_qp.hpp"
#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{
admm_qp::admm_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _rho_common(params.init_rho)
{
    _solution.dx = FlatVec(qp.dims.state, qp.dims.stage_state);
    _solution.du = FlatVec(qp.dims.control, qp.dims.stage_control);
    _solution.z  = FlatVec(qp.dims.constraint, qp.dims.stage_constraint);
    _solution.y  = FlatVec(_solution.z, false);

    _rho = FlatVec(_solution.z, false);
    _rho.vec.setConstant(_params.init_rho);
    _rho_inv = FlatVec(_solution.z, false);
    _rho_inv.vec.setConstant(1 / _params.init_rho);
}

SolverState admm_qp::solve(admm_qp::Solution &sol)
{
#ifdef DEBUG
    std::cout << "Solving with ADMM" << std::endl;
    std::cout << "=============" << std::endl;
#endif

    SolverState admm_state;

    // Warm start with unconstrained LQR
    if (_params.warm_start)
    {
        solvers::lqr::Output out =
          solvers::lqr::lqr(_qp, _solution, _rho, _params.sigma, false);
        _solution.dx = out.x;
        _solution.du = out.u;
    }

    for (_iter = 0;; ++_iter)
    {
        solvers::lqr::Output out =
          solvers::lqr::lqr(_qp, _solution, _rho, _params.sigma);
        _solution.P = out.P;
        _solution.p = out.p;

#ifdef DEBUG
        std::cout << "LQR solution, iteration " << _iter + 1 << std::endl;
        std::cout << "=============" << std::endl;
        std::cout << "x:" << std::endl;
        std::cout << out.x << std::endl;
        std::cout << "u:" << std::endl;
        std::cout << out.u << std::endl;
        std::cout << "=============" << std::endl;
#endif

        FlatVec z_prev = _solution.z;
        _step(out.x, out.u);
        _calculate_residuals(z_prev);

#ifdef DEBUG
        double tolerance_prim =
          _params.abs_tolerance + _params.rel_tolerance * _norm_primal_rel;

        double tolerance_dual =
          _params.abs_tolerance + _params.rel_tolerance * _norm_dual_rel;

        std::cout << "Update step, iteration " << _iter + 1 << std::endl;
        std::cout << "=============" << std::endl;
        std::cout << "x:" << std::endl;
        std::cout << _solution.dx << std::endl;
        std::cout << "u:" << std::endl;
        std::cout << _solution.du << std::endl;
        std::cout << "z:" << std::endl;
        std::cout << _solution.z << std::endl;
        std::cout << "y:" << std::endl;
        std::cout << _solution.y << std::endl;
        std::cout << "=============" << std::endl;
        std::cout << "Residuals" << std::endl;
        std::cout << "=============" << std::endl;
        std::cout << "Primal" << std::endl
                  << tolerance_prim << ", " << _norm_primal << std::endl;
        std::cout << "Dual" << std::endl
                  << tolerance_dual << ", " << _norm_dual << std::endl;
        std::cout << "=============" << std::endl;
        std::cin.get();
#endif

        if (!((_iter) % _params.rho_update_iter_freq))
        {
            _update_rho();
        }

        // Termination
        if ((admm_state = _check_termination()) != ONGOING)
        {
            break;
        }
    }

    sol = _solution;
    return admm_state;
}

void admm_qp::_calculate_residuals(const FlatVec &z_prev)
{
    _norm_primal     = -std::numeric_limits<double>::infinity();
    _norm_dual       = -std::numeric_limits<double>::infinity();
    _norm_primal_rel = -std::numeric_limits<double>::infinity();
    _norm_dual_rel   = -std::numeric_limits<double>::infinity();

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        VecX zdiff_scaled =
          _rho.stage(k).cwiseProduct(_solution.z.stage(k) - z_prev.stage(k));
        _norm_dual = std::max(_norm_dual,
                              (_qp.stages[k].D.transpose() * zdiff_scaled)
                                .lpNorm<Eigen::Infinity>());
        _norm_dual = std::max(_norm_dual,
                              (_qp.stages[k].E.transpose() * zdiff_scaled)
                                .lpNorm<Eigen::Infinity>());

        VecX CdxCdu = _qp.stages[k].D * _solution.dx.stage(k) +
                      _qp.stages[k].E * _solution.du.stage(k);
        _norm_primal =
          std::max(_norm_primal,
                   (CdxCdu - _solution.z.stage(k)).lpNorm<Eigen::Infinity>());

        _norm_primal_rel =
          std::max(_norm_primal_rel, CdxCdu.lpNorm<Eigen::Infinity>());
        _norm_primal_rel =
          std::max(_norm_primal_rel,
                   _solution.z.stage(k).lpNorm<Eigen::Infinity>());

        _norm_dual_rel =
          std::max(_norm_dual_rel,
                   (_qp.stages[k].D.transpose() * _solution.y.stage(k))
                     .lpNorm<Eigen::Infinity>());
        _norm_dual_rel =
          std::max(_norm_dual_rel,
                   (_qp.stages[k].E.transpose() * _solution.y.stage(k))
                     .lpNorm<Eigen::Infinity>());
    }

    VecX zdiff_scaled = _rho.stage(_qp.dims.horizon)
                          .cwiseProduct(_solution.z.stage(_qp.dims.horizon) -
                                        z_prev.stage(_qp.dims.horizon));
    _norm_dual =
      std::max(_norm_dual,
               (_qp.Dt.transpose() * zdiff_scaled).lpNorm<Eigen::Infinity>());

    VecX Cdx     = _qp.Dt * _solution.dx.stage(_qp.dims.horizon);
    _norm_primal = std::max(
      _norm_primal,
      (Cdx - _solution.z.stage(_qp.dims.horizon)).lpNorm<Eigen::Infinity>());

    _norm_primal_rel =
      std::max(_norm_primal_rel, Cdx.lpNorm<Eigen::Infinity>());
    _norm_primal_rel =
      std::max(_norm_primal_rel,
               _solution.z.stage(_qp.dims.horizon).lpNorm<Eigen::Infinity>());

    _norm_dual_rel =
      std::max(_norm_dual_rel,
               (_qp.Dt.transpose() * _solution.y.stage(_qp.dims.horizon))
                 .lpNorm<Eigen::Infinity>());
}

SolverState admm_qp::_check_termination() const
{
    if (_iter == _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    double tolerance_prim =
      _params.abs_tolerance + _params.rel_tolerance * _norm_primal_rel;

    double tolerance_dual =
      _params.abs_tolerance + _params.rel_tolerance * _norm_dual_rel;

#ifdef DEBUG
    std::cout << "Tolerances" << std::endl;
    std::cout << "=============" << std::endl;
    std::cout << "Primal: " << tolerance_prim << " vs " << _norm_primal
              << std::endl;
    std::cout << "Dual: " << tolerance_dual << " vs " << _norm_dual
              << std::endl;
    std::cout << "=============" << std::endl;
#endif

    if (_norm_primal > tolerance_prim)
    {
        if (_norm_dual < _params.primal_infeasibility_tolerance)
        {
            return PRIMAL_INFEASIBLE;
        }

        return ONGOING;
    }

    if (_norm_dual > tolerance_dual)
    {
        return ONGOING;
    }

    return SUCCESS;
}

void admm_qp::_update_rho()
{
    const double scale = std::sqrt((_norm_primal * _norm_dual_rel) /
                                   (_norm_dual * _norm_primal_rel));

    double rho_estimate_sparse_ =
      std::min(std::max(scale * _rho_common, _params.rho_min_),
               _params.rho_max_);

    if (rho_estimate_sparse_ > _rho_common * _params.adaptive_rho_tolerance_ ||
        rho_estimate_sparse_ < _rho_common / _params.adaptive_rho_tolerance_)
    {
        _rho_common = rho_estimate_sparse_;
        for (uint k = 0; k < _qp.dims.horizon; ++k)
        {
            for (uint c = 0; c < _qp.dims.stage_constraint[k]; ++c)
            {
                if (std::abs(_qp.stages[k].cl(c) - _qp.stages[k].cu(c)) <
                    std::numeric_limits<double>::epsilon())
                {
                    _rho.stage(k)(c)     = 1e3 * _rho_common;
                    _rho_inv.stage(k)(c) = 1. / (1e3 * _rho_common);
                }
                else
                {
                    _rho.stage(k)(c)     = _rho_common;
                    _rho_inv.stage(k)(c) = 1. / _rho_common;
                }
            }
        }

        for (uint c = 0; c < _qp.dims.stage_constraint[_qp.dims.horizon]; ++c)
        {
            if (std::abs(_qp.cl_t(c) - _qp.cu_t(c)) <
                std::numeric_limits<double>::epsilon())
            {
                _rho.stage(_qp.dims.horizon)(c)     = 1e3 * _rho_common;
                _rho_inv.stage(_qp.dims.horizon)(c) = 1. / (1e3 * _rho_common);
            }
            else
            {
                _rho.stage(_qp.dims.horizon)(c)     = _rho_common;
                _rho_inv.stage(_qp.dims.horizon)(c) = 1. / _rho_common;
            }
        }
    }
}

void admm_qp::_step(const FlatVec &dx_hat, const FlatVec &du_hat)
{
    FlatVec z_hat(_solution.z, false);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        z_hat.stage(k) = _params.alpha * (_qp.stages[k].D * dx_hat.stage(k) +
                                          _qp.stages[k].E * du_hat.stage(k)) +
                         (1 - _params.alpha) * _solution.z.stage(k);
        _solution.dx.stage(k) = _params.alpha * dx_hat.stage(k) +
                                (1 - _params.alpha) * _solution.dx.stage(k);
        _solution.du.stage(k) = _params.alpha * du_hat.stage(k) +
                                (1 - _params.alpha) * _solution.du.stage(k);

        _solution.z.stage(k) =
          (-_qp.stages[k].cu)
            .cwiseMin(
              (-_qp.stages[k].cl)
                .cwiseMax(z_hat.stage(k) + _rho_inv.stage(k).cwiseProduct(
                                             _solution.y.stage(k))));
        _solution.y.stage(k) +=
          _rho.stage(k).cwiseProduct(z_hat.stage(k) - _solution.z.stage(k));
    }
    z_hat.stage(_qp.dims.horizon) =
      _params.alpha * _qp.Dt * dx_hat.stage(_qp.dims.horizon) +
      (1 - _params.alpha) * _solution.z.stage(_qp.dims.horizon);
    _solution.dx.stage(_qp.dims.horizon) =
      _params.alpha * dx_hat.stage(_qp.dims.horizon) +
      (1 - _params.alpha) * _solution.dx.stage(_qp.dims.horizon);

    _solution.z.stage(_qp.dims.horizon) =
      (-_qp.cu_t).cwiseMin((-_qp.cl_t).cwiseMax(
        z_hat.stage(_qp.dims.horizon) +
        _rho_inv.stage(_qp.dims.horizon)
          .cwiseProduct(_solution.y.stage(_qp.dims.horizon))));
    _solution.y.stage(_qp.dims.horizon) +=
      _rho.stage(_qp.dims.horizon)
        .cwiseProduct(z_hat.stage(_qp.dims.horizon) -
                      _solution.z.stage(_qp.dims.horizon));
}
}
}