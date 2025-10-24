#include "solvers/admm_qp.hpp"
#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{
admm_qp::Solution::Solution(const model::Dimensions &dims)
  : dx(dims.state, dims.stage_state)
  , du(dims.control, dims.stage_control)
  , z(dims.constraint, dims.stage_constraint)
  , y(z, false)
{
}

admm_qp::admm_qp(const QPData &qp, const Params &params)
  : _qp(qp)
  , _params(params)
  , _solution(qp.dims)
  , _rho_common(params.init_rho)
  , _rho(_solution.z, false)
  , _rho_inv(_solution.z, false)
{
    _rho.vec.setConstant(_params.init_rho);
    _rho_inv.vec.setConstant(1 / _params.init_rho);
}

SolverState admm_qp::solve(admm_qp::Solution &sol)
{
    DEBUG_LOG("Solving with ADMM" << std::endl);
    DEBUG_LOG("=============" << std::endl);

    SolverState admm_state;

    // Warm start with unconstrained LQR
    solvers::lqr lqr_solver(_qp, _solution, _rho, _params.sigma);

    lqr::Output lqr_out = lqr_solver.output_structure();

    if (_params.warm_start)
    {
        lqr_out      = lqr_solver.unconstrained();
        _solution.dx = lqr_out.x;
        _solution.du = lqr_out.u;
    }

    _rho_changed = true;
    for (_iter = 0;; ++_iter)
    {
        if (_rho_changed)
        {
            lqr_out      = lqr_solver.constrained(_solution, _rho);
            _rho_changed = false;
        }
        else
        {
            lqr_out = lqr_solver.constrained(_solution);
        }
        _solution.P = lqr_out.P;
        _solution.p = lqr_out.p;

        DEBUG_LOG("LQR solution, iteration " << _iter + 1 << std::endl);
        DEBUG_LOG("=============" << std::endl);
        DEBUG_LOG("x:" << std::endl);
        DEBUG_LOG(lqr_out.x << std::endl);
        DEBUG_LOG("u:" << std::endl);
        DEBUG_LOG(lqr_out.u << std::endl);
        DEBUG_LOG("=============" << std::endl);

        FlatVec x_prev = _solution.dx, u_prev = _solution.du,
                z_prev = _solution.z, y_prev = _solution.y;
        _step(lqr_out.x, lqr_out.u);
        _calculate_residuals(x_prev, u_prev, z_prev, y_prev);

        DEBUG_LOG("Update step, iteration " << _iter + 1 << std::endl);
        DEBUG_LOG("=============" << std::endl);
        DEBUG_LOG("x:" << std::endl);
        DEBUG_LOG(_solution.dx << std::endl);
        DEBUG_LOG("u:" << std::endl);
        DEBUG_LOG(_solution.du << std::endl);
        DEBUG_LOG("z:" << std::endl);
        DEBUG_LOG(_solution.z << std::endl);
        DEBUG_LOG("y:" << std::endl);
        DEBUG_LOG(_solution.y << std::endl);
        DEBUG_LOG("=============" << std::endl);
        DEBUG_WAIT();

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

void admm_qp::_calculate_residuals(const FlatVec &x_prev, const FlatVec &u_prev,
                                   const FlatVec &z_prev, const FlatVec &y_prev)
{
    _norm_primal     = -std::numeric_limits<double>::infinity();
    _norm_dual       = -std::numeric_limits<double>::infinity();
    _norm_primal_rel = -std::numeric_limits<double>::infinity();
    _norm_dual_rel   = -std::numeric_limits<double>::infinity();

    /*
    _primal_infs_norm       = -std::numeric_limits<double>::infinity();
    _primal_infs_support    = 0;
    _primal_infs_tol_scaled = _params.primal_infs_tolerance;
    */

    // TODO: Infeasibility
    FlatVec dy = _solution.y - y_prev;
    // dy.vec     /= dy.vec.lpNorm<Eigen::Infinity>();

    // DEBUG_LOG("dx: " << _solution.dx - x_prev);
    // DEBUG_LOG("du: " << _solution.du - u_prev);
    // DEBUG_LOG("dy: " << dy);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        // DEBUG_LOG("Stage " << k << ": " << std::endl);
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

        /*
        _primal_infs_norm = std::max(_primal_infs_norm,
                                     (_qp.stages[k].D.transpose() * dy.stage(k))
                                       .lpNorm<Eigen::Infinity>());
        DEBUG_LOG("|| D*dy || = " << (_qp.stages[k].D.transpose() * dy.stage(k))
                                       .lpNorm<Eigen::Infinity>()
                                  << ", ");
        _primal_infs_norm = std::max(_primal_infs_norm,
                                     (_qp.stages[k].E.transpose() * dy.stage(k))
                                       .lpNorm<Eigen::Infinity>());
        DEBUG_LOG("|| E*dy || = " << (_qp.stages[k].E.transpose() * dy.stage(k))
                                       .lpNorm<Eigen::Infinity>()
                                  << std::endl);

        _primal_infs_support +=
          (_qp.stages[k].cu.transpose() * dy.stage(k).cwiseMax(0) +
           _qp.stages[k].cl.transpose() * dy.stage(k).cwiseMin(0))(0);
        */
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

    /*
    _primal_infs_norm =
      std::max(_primal_infs_norm,
               (_qp.Dt.transpose() * dy.stage(_qp.dims.horizon))
                 .lpNorm<Eigen::Infinity>());


    DEBUG_LOG("Horizon:" << std::endl);
    DEBUG_LOG("D = " << std::endl << _qp.Dt << std::endl);
    DEBUG_LOG("|| D*dy || = "
              << (_qp.Dt.transpose() * dy.stage(_qp.dims.horizon))
                   .lpNorm<Eigen::Infinity>()
              << std::endl);

    _primal_infs_support +=
      (_qp.cu_t.transpose() * dy.stage(_qp.dims.horizon).cwiseMax(0) +
       _qp.cl_t.transpose() * dy.stage(_qp.dims.horizon).cwiseMin(0))(0);

    _primal_infs_tol_scaled *= dy.vec.lpNorm<Eigen::Infinity>();
    DEBUG_LOG("dy norm: " << dy.vec.lpNorm<Eigen::Infinity>() << std::endl);
    */
}

SolverState admm_qp::_check_termination() const
{
    if (_iter == _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    /*
    DEBUG_LOG("Primal infeasiblity norm: " << _primal_infs_norm << std::endl);
    DEBUG_LOG("Primal infeasiblity support: " << _primal_infs_support
                                              << std::endl);
    DEBUG_LOG("Primal infeasiblity tolerance: " << _primal_infs_tol_scaled
                                                << std::endl);

    if (_primal_infs_norm < _primal_infs_tol_scaled &&
        _primal_infs_support < _primal_infs_tol_scaled)
    {
        return PRIMAL_INFEASIBLE;
    }
    */

    DEBUG_LOG("rho:" << std::endl << _rho << std::endl);

    double tolerance_prim =
      _params.abs_tolerance + _params.rel_tolerance * _norm_primal_rel;

    double tolerance_dual =
      _params.abs_tolerance + _params.rel_tolerance * _norm_dual_rel;

    DEBUG_LOG("Tolerances" << std::endl);
    DEBUG_LOG("=============" << std::endl);
    DEBUG_LOG("Primal (tol vs norm): " << tolerance_prim << " vs "
                                       << _norm_primal << std::endl);
    DEBUG_LOG("Dual (tol vs norm): " << tolerance_dual << " vs " << _norm_dual
                                     << std::endl);
    DEBUG_LOG("=============" << std::endl);

    if (_norm_primal > tolerance_prim || _norm_dual > tolerance_dual)
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
                else if (_qp.stages[k].cl(c) < _qp.stages[k].cu(c))
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
            else if (_qp.cl_t(c) < _qp.cu_t(c))
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
    FlatVec z_hat(_solution.z, false);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        z_hat.stage(k).noalias() =
          _params.alpha * (_qp.stages[k].D * dx_hat.stage(k) +
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
    z_hat.stage(_qp.dims.horizon).noalias() =
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
    DEBUG_LOG("Terminal z diff: " << z_hat.stage(_qp.dims.horizon) -
                                       _solution.z.stage(_qp.dims.horizon)
                                  << std::endl);
    DEBUG_LOG("Terminal y increase: "
              << _rho.stage(_qp.dims.horizon)
                   .cwiseProduct(z_hat.stage(_qp.dims.horizon) -
                                 _solution.z.stage(_qp.dims.horizon))
              << std::endl);
}
}
}