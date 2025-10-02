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
  , _rho(qp.horizon + 1, DMatX(qp.nc))
  , _rho_inv(qp.horizon + 1, DMatX(qp.nc))
  , _Dx(MatX(qp.nc, qp.horizon + 1))
  , _rQTz(MatX(qp.nx + qp.nu, qp.horizon + 1))
  , _rQTzp(MatX(qp.nx + qp.nu, qp.horizon + 1))
  , _r_prim(MatX(qp.nc, qp.horizon + 1))
  , _r_dual(MatX(qp.nx + qp.nu, qp.horizon + 1))
{
    assert(guess.x.cols() == guess.u.cols() + 1 &&
           "Inconsistent time horizon"); // ensure consistent time horizon

    _solution.dX = MatX::Zero(qp.nx, qp.horizon + 1);
    _solution.dU = MatX::Zero(qp.nu, qp.horizon);
    _solution.z  = MatX::Zero(qp.nc, qp.horizon + 1);
    _solution.y  = MatX::Zero(qp.nc, qp.horizon + 1);

    for (uint k = 0; k <= qp.horizon; ++k)
    {
        _rho[k].diagonal().setConstant(_params.init_rho);
        _rho_inv[k].diagonal().setConstant(1 / _params.init_rho);
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
        std::cout << "Primal" << std::endl << _r_prim << std::endl;
        std::cout << "Dual" << std::endl << _r_dual << std::endl;
        std::cout << "=============" << std::endl;
        std::cin.get();
#endif

        if (!((_iter) % _params.rho_update_iter_freq))
        {
            //_update_rho();
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
    for (uint k = 0; k < _qp.horizon; ++k)
    {
        _Dx.col(k) = _qp.stages[k].D * _solution.dX.col(k) +
                     _qp.stages[k].E * _solution.dU.col(k);

        VecX z_scaled  = _rho[k] * _solution.z.col(k);
        VecX zp_scaled = _rho[k] * z_prev.col(k);

        _rQTz.col(k).head(_qp.nx) = _qp.stages[k].D.transpose() * z_scaled;
        _rQTz.col(k).tail(_qp.nu) = _qp.stages[k].E.transpose() * z_scaled;

        _rQTzp.col(k).head(_qp.nx) = _qp.stages[k].D.transpose() * zp_scaled;
        _rQTzp.col(k).tail(_qp.nu) = _qp.stages[k].E.transpose() * zp_scaled;
    }

    _Dx.col(_qp.horizon) = _qp.Dt * _solution.dX.col(_qp.horizon);

    VecX z_scaled  = _rho[_qp.horizon] * _solution.z.col(_qp.horizon);
    VecX zp_scaled = _rho[_qp.horizon] * z_prev.col(_qp.horizon);

    _rQTz.col(_qp.horizon).head(_qp.nx) = _qp.Dt.transpose() * z_scaled;
    _rQTz.col(_qp.horizon).tail(_qp.nu).setZero();

    _rQTzp.col(_qp.horizon).head(_qp.nx) = _qp.Dt.transpose() * zp_scaled;
    _rQTzp.col(_qp.horizon).tail(_qp.nu).setZero();

    _r_prim = _Dx - _solution.z;
    _r_dual = _rQTz - _rQTzp;
}

SolverState admm_qp::_check_termination() const
{
    if (_iter == _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    double tolerance_prim =
      _params.abs_tolerance +
      _params.rel_tolerance * std::max(_Dx.norm(), _solution.z.norm());

    double tolerance_dual =
      _params.abs_tolerance +
      _params.rel_tolerance * std::max(_rQTz.lpNorm<Eigen::Infinity>(),
                                       _rQTzp.lpNorm<Eigen::Infinity>());

#ifdef DEBUG
    std::cout << "Tolerances" << std::endl;
    std::cout << "=============" << std::endl;
    std::cout << "Primal: " << tolerance_prim << std::endl;
    std::cout << "Dual: " << tolerance_dual << std::endl;
    std::cout << "=============" << std::endl;
#endif

    double r_prim_infnorm = _r_prim.lpNorm<Eigen::Infinity>();
    double r_dual_infnorm = _r_dual.lpNorm<Eigen::Infinity>();

    if (r_prim_infnorm >= tolerance_prim)
    {
        if (r_dual_infnorm < _params.primal_infeasibility_tolerance)
        {
            return PRIMAL_INFEASIBLE;
        }

        return ONGOING;
    }

    if (_r_dual.lpNorm<Eigen::Infinity>() >= tolerance_dual)
    {
        return ONGOING;
    }

    return SUCCESS;
}

// TODO: this results in primal infeasibility
// causes r_dual to become very small even as r_prim is still converging
void admm_qp::_update_rho()
{
    for (uint k = 0; k < _qp.horizon; ++k)
    {
        VecX &rho     = _rho[k].diagonal();
        VecX &rho_inv = _rho_inv[k].diagonal();

        for (uint c = 0; c < _qp.nc; ++c)
        {
            // Active constraint
            if (std::abs(_solution.z(c, k) - _qp.stages[k].c(c)) < 1e-5)
            {
                rho(c)     = _params.init_rho;
                rho_inv(c) = 1 / _params.init_rho;
            }
            // Inactive constraint
            else
            {
                rho(c)     = _rho_min;
                rho_inv(c) = 1 / _rho_min;
            }
            // TODO: Equality constraint
        }
    }

    // TODO: make horizon generally part of other terms
    // Active constraint
    for (uint c = 0; c < _qp.nc; ++c)
    {
        VecX &rho     = _rho[_qp.horizon].diagonal();
        VecX &rho_inv = _rho_inv[_qp.horizon].diagonal();

        // Active constraint
        if (std::abs(_solution.z(c, _qp.horizon) - _qp.ct(c)) < 1e-5)
        {
            rho(c)     = _params.init_rho;
            rho_inv(c) = 1 / _params.init_rho;
        }
        // Inactive constraint
        else
        {
            rho(c)     = _rho_min;
            rho_inv(c) = 1 / _rho_min;
        }
        // TODO: Equality constraint
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
            .cwiseMax(z_hat.col(k) + _rho_inv[k] * _solution.y.col(k));
        _solution.y.col(k) += _rho[k] * (z_hat.col(k) - _solution.z.col(k));
    }
    z_hat.col(_qp.horizon) = _params.alpha * _qp.Dt * dX_hat.col(_qp.horizon) +
                             (1 - _params.alpha) * _solution.z.col(_qp.horizon);
    _solution.dX.col(_qp.horizon) =
      _params.alpha * dX_hat.col(_qp.horizon) +
      (1 - _params.alpha) * _solution.dX.col(_qp.horizon);

    _solution.z.col(_qp.horizon) =
      (-_qp.ct).cwiseMax(z_hat.col(_qp.horizon) +
                         _rho_inv[_qp.horizon] * _solution.y.col(_qp.horizon));
    _solution.y.col(_qp.horizon) +=
      _rho[_qp.horizon] *
      (z_hat.col(_qp.horizon) - _solution.z.col(_qp.horizon));
}
}
}