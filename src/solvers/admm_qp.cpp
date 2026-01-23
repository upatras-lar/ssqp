#include "solvers/admm_qp.hpp"
#include "solvers/lqr.hpp"

#include "tracy/Tracy.hpp"

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
  , _reg_incr(0)
  , _abs_tolerance_init(_params.abs_tolerance)
  , _rel_tolerance_init(_params.rel_tolerance)
{
    _rho.vec.setConstant(_params.init_rho);
    _rho_inv.vec.setConstant(1 / _params.init_rho);
}

void admm_qp::reset(const QPData &qp)
{
    _qp = qp;

    if (_params.reset_rho)
    {
        _rho_common = _params.init_rho;
        _rho.vec.setConstant(_params.init_rho);
        _rho_inv.vec.setConstant(1 / _params.init_rho);
    }

    if (_params.reset_y)
    {
        _solution.y.vec.setZero();
    }

    _solution.z.vec.setZero();

    _reg_incr = 0;
}

void admm_qp::increaseRegularization()
{
    ++_reg_incr;
}

SolverState admm_qp::solve(admm_qp::Solution &sol)
{
    ZoneScopedN("ADMM: solve");

    DEBUG_LOGLN("Solving with ADMM");
    DEBUG_LOGLN("=============");

    SolverState admm_state;

    // Warm start with unconstrained LQR
    lqr::Params lqr_params;
    lqr_params.reg_init *= std::pow(10, _reg_incr);

    lqr lqr_solver(_qp, _solution, _rho, lqr_params);

    lqr::Output &lqr_out = lqr_solver.output_structure();

    if (_params.warm_start)
    {
        lqr_out      = lqr_solver.unconstrained();
        _solution.dx = lqr_out.x;
        _solution.du = lqr_out.u;
        DEBUG_LOGLN("LQR unconstrained solution");
        DEBUG_LOGLN("=============");
        DEBUG_LOGLN("x:");
        DEBUG_LOGLN(lqr_out.x);
        DEBUG_LOGLN("u:");
        DEBUG_LOGLN(lqr_out.u);
        DEBUG_LOGLN("=============");
    }

    _params.abs_tolerance = _abs_tolerance_init;
    _params.rel_tolerance = _rel_tolerance_init;

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

        DEBUG_LOGLN("LQR solution, iteration " << _iter + 1);
        DEBUG_LOGLN("=============");
        DEBUG_LOGLN("x:");
        DEBUG_LOGLN(lqr_out.x);
        DEBUG_LOGLN("u:");
        DEBUG_LOGLN(lqr_out.u);
        DEBUG_LOGLN("=============");

        FlatVec x_prev = _solution.dx, u_prev = _solution.du,
                z_prev = _solution.z, y_prev = _solution.y;

        _step(lqr_out.x, lqr_out.u);

        DEBUG_LOGLN("Update step, iteration " << _iter + 1);
        DEBUG_LOGLN("=============");
        DEBUG_LOGLN("x:");
        DEBUG_LOGLN(_solution.dx);
        DEBUG_LOGLN("u:");
        DEBUG_LOGLN(_solution.du);
        DEBUG_LOGLN("z:");
        DEBUG_LOGLN(_solution.z);
        DEBUG_LOGLN("y:");
        DEBUG_LOGLN(_solution.y);
        DEBUG_LOGLN("=============");
        //DEBUG_WAIT();

        if (!((_iter) % _params.rho_update_iter_freq))
        {
            _calculate_residuals(x_prev, u_prev, z_prev, y_prev);

            // Termination
            if ((admm_state = _check_termination()) != ONGOING)
            {
                break;
            }

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

    if (admm_state == SUCCESS)
    {
        LOGLN("ADMM: Converged in " << _iter + 1 << " iterations");
    }

    _solution.P = lqr_out.P;
    _solution.p = lqr_out.p;

    sol = _solution;
    return admm_state;
}

void admm_qp::_calculate_residuals(const FlatVec &x_prev, const FlatVec &u_prev,
                                   const FlatVec &z_prev, const FlatVec &y_prev)
{
    ZoneScopedN("ADMM: _calculate_residuals");
    _norm_primal     = -std::numeric_limits<double>::infinity();
    _norm_dual       = -std::numeric_limits<double>::infinity();
    _norm_primal_rel = -std::numeric_limits<double>::infinity();
    _norm_dual_rel   = -std::numeric_limits<double>::infinity();

    _primal_infs_norm       = -std::numeric_limits<double>::infinity();
    _primal_infs_support    = 0;
    _primal_infs_tol_scaled = _params.primal_infs_tolerance;

    FlatVec dy = _solution.y - y_prev;

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        if (_qp.stages[k].D.rows() == 0)
        {
            continue;
        }

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
        _primal_infs_norm = std::max(_primal_infs_norm,
                                     (_qp.stages[k].E.transpose() * dy.stage(k))
                                       .lpNorm<Eigen::Infinity>());

        for (uint i = 0; i < _qp.stages[k].cu.size(); ++i)
        {
            if (std::isfinite(_qp.stages[k].cu[i]))
            {
                _primal_infs_support +=
                  _qp.stages[k].cu[i] * std::max(dy.stage(k)[i], 0.0);
            }

            if (std::isfinite(_qp.stages[k].cl[i]))
            {
                _primal_infs_support +=
                  _qp.stages[k].cl[i] * std::min(dy.stage(k)[i], 0.0);
            }
        }
        */
    }

    if (_qp.Dt.rows() == 0)
    {
        return;
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

    for (uint i = 0; i < _qp.cu_t.size(); ++i)
    {
        if (std::isfinite(_qp.cu_t[i]))
        {
            _primal_infs_support +=
              _qp.cu_t[i] * std::max(dy.stage(_qp.dims.horizon)[i], 0.0);
        }

        if (std::isfinite(_qp.cl_t[i]))
        {
            _primal_infs_support +=
              _qp.cl_t[i] * std::min(dy.stage(_qp.dims.horizon)[i], 0.0);
        }
    }

    _primal_infs_tol_scaled *= dy.vec.lpNorm<Eigen::Infinity>();
    */
}

SolverState admm_qp::_check_termination() const
{
    ZoneScopedN("ADMM: _check_termination");

    if (_iter >= _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    // LOGLN("Primal infeasibility norm vs tolerance: "
    //       << _primal_infs_norm << " vs " << _primal_infs_tol_scaled);
    // LOGLN("Primal infeasibility support vs tolerance: "
    //       << _primal_infs_support << " vs " << _primal_infs_tol_scaled);

    // TODO: Primal infeasibility detection doesn't work yet
    // if (_primal_infs_norm < _primal_infs_tol_scaled &&
    //     _primal_infs_support < _primal_infs_tol_scaled)
    // {
    //     return PRIMAL_INFEASIBLE;
    // }

    DEBUG_LOGLN("rho:" << std::endl << _rho);

    double tolerance_prim =
      _params.abs_tolerance + _params.rel_tolerance * _norm_primal_rel;

    double tolerance_dual =
      _params.abs_tolerance + _params.rel_tolerance * _norm_dual_rel;

    DEBUG_LOGLN("Tolerances");
    DEBUG_LOGLN("=============");
    DEBUG_LOGLN("Primal (tol vs norm): " << tolerance_prim << " vs "
                                         << _norm_primal);
    DEBUG_LOGLN("Dual (tol vs norm): " << tolerance_dual << " vs "
                                       << _norm_dual);
    DEBUG_LOGLN("=============");

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

    FlatVec z_hat(_solution.z, false);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _solution.dx.stage(k) = _params.alpha * dx_hat.stage(k) +
                                (1 - _params.alpha) * _solution.dx.stage(k);
        _solution.du.stage(k) = _params.alpha * du_hat.stage(k) +
                                (1 - _params.alpha) * _solution.du.stage(k);

        if (_qp.stages[k].cl.size() == 0)
        {
            continue;
        }

        z_hat.stage(k).noalias() =
          _params.alpha * (_qp.stages[k].D * dx_hat.stage(k) +
                           _qp.stages[k].E * du_hat.stage(k)) +
          (1 - _params.alpha) * _solution.z.stage(k);

        _solution.z.stage(k) =
          _qp.stages[k].cu.cwiseMin(_qp.stages[k].cl.cwiseMax(
            z_hat.stage(k) +
            _rho_inv.stage(k).cwiseProduct(_solution.y.stage(k))));

        _solution.y.stage(k) +=
          _rho.stage(k).cwiseProduct(z_hat.stage(k) - _solution.z.stage(k));
    }
    _solution.dx.stage(_qp.dims.horizon) =
      _params.alpha * dx_hat.stage(_qp.dims.horizon) +
      (1 - _params.alpha) * _solution.dx.stage(_qp.dims.horizon);

    if (_qp.cl_t.size() == 0)
    {
        return;
    }

    z_hat.stage(_qp.dims.horizon).noalias() =
      _params.alpha * _qp.Dt * dx_hat.stage(_qp.dims.horizon) +
      (1 - _params.alpha) * _solution.z.stage(_qp.dims.horizon);

    _solution.z.stage(_qp.dims.horizon) = _qp.cu_t.cwiseMin(
      _qp.cl_t.cwiseMax(z_hat.stage(_qp.dims.horizon) +
                        _rho_inv.stage(_qp.dims.horizon)
                          .cwiseProduct(_solution.y.stage(_qp.dims.horizon))));
    _solution.y.stage(_qp.dims.horizon) +=
      _rho.stage(_qp.dims.horizon)
        .cwiseProduct(z_hat.stage(_qp.dims.horizon) -
                      _solution.z.stage(_qp.dims.horizon));
}
}
}