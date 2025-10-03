#include "solvers/admm_qp.hpp"
#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{
admm_qp::admm_qp(const QPData &qp, const csqp::Guess &guess,
                 const Params &params)
  : _qp(qp)
  , _guess(guess)
  , _params(params)
  , _rho(qp.horizon + 1, VecX(qp.nc))
  , _rho_inv(qp.horizon + 1, VecX(qp.nc))
  , _rho_common(params.init_rho)
{
    assert(guess.x.cols() == guess.u.cols() + 1 &&
           "Inconsistent time horizon"); // ensure consistent time horizon

    _solution.dX = MatX::Zero(qp.nx, qp.horizon + 1);
    _solution.dU = MatX::Zero(qp.nu, qp.horizon);
    _solution.z  = MatX::Zero(qp.nc, qp.horizon + 1);
    _solution.y  = MatX::Zero(qp.nc, qp.horizon + 1);

    for (uint k = 0; k <= qp.horizon; ++k)
    {
        _rho[k].setConstant(_params.init_rho);
        _rho_inv[k].setConstant(1. / _params.init_rho);

        // for (uint c = 0; c < _qp.nc; ++c)
        // {
        //     if (k == _qp.horizon && c >= 2)
        //     {
        //         _rho[k][c]     = 1e2 * _params.init_rho;
        //         _rho_inv[k][c] = 1. / (1e2 * _params.init_rho);
        //     }
        // }
    }
}

SolverState admm_qp::solve(admm_qp::Solution &sol)
{
#ifdef DEBUG
    std::cout << "Solving with ADMM" << std::endl;
    std::cout << "=============" << std::endl;
#endif

    SolverState admm_state;

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

        MatX z_prev = _solution.z;
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
        std::cout << _solution.dX << std::endl;
        std::cout << "u:" << std::endl;
        std::cout << _solution.dU << std::endl;
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

void admm_qp::_calculate_residuals(const MatX &z_prev)
{
    _norm_primal     = -std::numeric_limits<double>::infinity();
    _norm_dual       = -std::numeric_limits<double>::infinity();
    _norm_primal_rel = -std::numeric_limits<double>::infinity();
    _norm_dual_rel   = -std::numeric_limits<double>::infinity();

    for (uint k = 0; k < _qp.horizon; ++k)
    {
        // std::cout << "ρ[" << k << "]: " << _rho[k].transpose() << std::endl;
        VecX zdiff_scaled =
          _rho[k].cwiseProduct(_solution.z.col(k) - z_prev.col(k));
        _norm_dual = std::max(_norm_dual,
                              (_qp.stages[k].D.transpose() * zdiff_scaled)
                                .lpNorm<Eigen::Infinity>());
        _norm_dual = std::max(_norm_dual,
                              (_qp.stages[k].E.transpose() * zdiff_scaled)
                                .lpNorm<Eigen::Infinity>());

        VecX CdxCdu = _qp.stages[k].D * _solution.dX.col(k) +
                      _qp.stages[k].E * _solution.dU.col(k);
        _norm_primal =
          std::max(_norm_primal,
                   (CdxCdu - _solution.z.col(k)).lpNorm<Eigen::Infinity>());

        _norm_primal_rel =
          std::max(_norm_primal_rel, CdxCdu.lpNorm<Eigen::Infinity>());
        _norm_primal_rel =
          std::max(_norm_primal_rel,
                   _solution.z.col(k).lpNorm<Eigen::Infinity>());

        _norm_dual_rel =
          std::max(_norm_dual_rel,
                   (_qp.stages[k].D.transpose() * _solution.y.col(k))
                     .lpNorm<Eigen::Infinity>());
        _norm_dual_rel =
          std::max(_norm_dual_rel,
                   (_qp.stages[k].E.transpose() * _solution.y.col(k))
                     .lpNorm<Eigen::Infinity>());
    }
    // std::cout << "ρ[" << _qp.horizon << "]: " <<
    // _rho[_qp.horizon].transpose()
    //   << std::endl;

    VecX zdiff_scaled = _rho[_qp.horizon].cwiseProduct(
      _solution.z.col(_qp.horizon) - z_prev.col(_qp.horizon));

    _norm_dual =
      std::max(_norm_dual,
               (_qp.Dt.transpose() * zdiff_scaled).lpNorm<Eigen::Infinity>());

    VecX Cdx = _qp.Dt * _solution.dX.col(_qp.horizon);
    _norm_primal =
      std::max(_norm_primal,
               (Cdx - _solution.z.col(_qp.horizon)).lpNorm<Eigen::Infinity>());

    _norm_primal_rel =
      std::max(_norm_primal_rel, Cdx.lpNorm<Eigen::Infinity>());
    _norm_primal_rel =
      std::max(_norm_primal_rel,
               _solution.z.col(_qp.horizon).lpNorm<Eigen::Infinity>());

    _norm_dual_rel =
      std::max(_norm_dual_rel,
               (_qp.Dt.transpose() * _solution.y.col(_qp.horizon))
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

    // double r_prim_infnorm = _norm_primal;
    // double r_dual_infnorm = _norm_dual;

    // if (r_prim_infnorm > tolerance_prim)
    // {
    //     if (r_dual_infnorm < _params.primal_infeasibility_tolerance)
    //     {
    //         return PRIMAL_INFEASIBLE;
    //     }

    //     return ONGOING;
    // }

    // if (_norm_dual > tolerance_dual)
    // {
    //     return ONGOING;
    // }
    if (_norm_primal > tolerance_prim || _norm_dual > tolerance_dual)
        return ONGOING;

    return SUCCESS;
}

// TODO: this results in primal infeasibility
// causes r_dual to become very small even as r_prim is still converging
void admm_qp::_update_rho()
{
    const double rho_min_                = 1e-6;
    const double rho_max_                = 1e3;
    const double adaptive_rho_tolerance_ = 5;

    const double scale = std::sqrt((_norm_primal * _norm_dual_rel) /
                                   (_norm_dual * _norm_primal_rel));

    double rho_estimate_sparse_ =
      std::min(std::max(scale * _rho_common, rho_min_), rho_max_);

    if (rho_estimate_sparse_ > _rho_common * adaptive_rho_tolerance_ ||
        rho_estimate_sparse_ < _rho_common / adaptive_rho_tolerance_)
    {
        _rho_common = rho_estimate_sparse_;
        // Let's apply it
        for (uint k = 0; k <= _qp.horizon; ++k)
        {
            VecX &rho     = _rho[k];
            VecX &rho_inv = _rho_inv[k];

            for (uint c = 0; c < _qp.nc; ++c)
            {
                // if (k == _qp.horizon && c >= 2)
                // {
                //     rho(c)     = 1e3 * _rho_common;
                //     rho_inv(c) = 1. / (1e3 * _rho_common);
                // }
                // else
                {
                    rho(c)     = _rho_common;
                    rho_inv(c) = 1. / _rho_common;
                }
            }
        }
    }
}

void admm_qp::_step(const MatX &dX_hat, const MatX &dU_hat)
{
    MatX z_hat(_qp.nc, _qp.horizon + 1);

    for (uint k = 0; k < _qp.horizon; ++k)
    {
        z_hat.col(k) = _params.alpha * (_qp.stages[k].D * dX_hat.col(k) +
                                        _qp.stages[k].E * dU_hat.col(k)) +
                       (1 - _params.alpha) * _solution.z.col(k);
        _solution.dX.col(k) = _params.alpha * dX_hat.col(k) +
                              (1 - _params.alpha) * _solution.dX.col(k);
        _solution.dU.col(k) = _params.alpha * dU_hat.col(k) +
                              (1 - _params.alpha) * _solution.dU.col(k);

        _solution.z.col(k) =
          (-_qp.stages[k].c)
            .cwiseMax(z_hat.col(k) +
                      _rho_inv[k].cwiseProduct(_solution.y.col(k)));
        _solution.y.col(k) +=
          _rho[k].cwiseProduct(z_hat.col(k) - _solution.z.col(k));
    }
    z_hat.col(_qp.horizon) = _params.alpha * _qp.Dt * dX_hat.col(_qp.horizon) +
                             (1 - _params.alpha) * _solution.z.col(_qp.horizon);
    _solution.dX.col(_qp.horizon) =
      _params.alpha * dX_hat.col(_qp.horizon) +
      (1 - _params.alpha) * _solution.dX.col(_qp.horizon);

    _solution.z.col(_qp.horizon) = (-_qp.ct).cwiseMax(
      z_hat.col(_qp.horizon) +
      _rho_inv[_qp.horizon].cwiseProduct(_solution.y.col(_qp.horizon)));
    _solution.y.col(_qp.horizon) += _rho[_qp.horizon].cwiseProduct(
      z_hat.col(_qp.horizon) - _solution.z.col(_qp.horizon));
}
}
}