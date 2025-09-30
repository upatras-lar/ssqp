#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "solvers/lqr.hpp"
#include "model/constraint/AbstractConstraint.hpp"
#include "model/cost/AbstractCost.hpp"
#include "model/dynamics/AbstractDynamics.hpp"

#ifndef SSQP_ADMM_QP_HPP
#define SSQP_ADMM_QP_HPP

namespace ssqp
{
namespace solvers
{
struct admmParams
{
    double       rho                  = 0.1;
    const double alpha                = 1.6;
    const double sigma                = 10e-6;
    const uint   rho_scale_factor     = 2;
    const uint   rho_incr_factor      = 5;
    const uint   rho_decr_factor      = 5;
    const uint   rho_update_iter_freq = 25;
    const double tolerance_abs        = 10e-4;
    const double tolerance_rel        = 10e-3;
};

class admm_qp
{
  public:
    admm_qp(const QPData &qp, const Guess &guess,
            const admmParams &params = admmParams {})
      : _qp(qp)
      , _guess(guess)
      , _params(params)
      , _r_prim(MatX(qp.nc, qp.horizon + 1))
      , _r_dual(MatX(qp.nx + qp.nu, qp.horizon + 1))
    {
        assert(guess.x.cols() == guess.u.cols() + 1 &&
               "Inconsistent time horizon"); // ensure consistent time horizon

        _solution.dX = MatX::Zero(qp.nx, qp.horizon + 1);
        _solution.dU = MatX::Zero(qp.nu, qp.horizon);
        _solution.z  = MatX::Zero(qp.nc, qp.horizon + 1);
        _solution.y  = MatX::Zero(qp.nc, qp.horizon + 1);
    }

    admmQPSolution solve()
    {
#ifdef DEBUG
        std::cout << "Solving with ADMM" << std::endl;
        std::cout << "=============" << std::endl;
#endif

        for (_iter = 0;; ++_iter)
        {
            solvers::lqr::Output out =
              solvers::lqr::lqr(_qp, _solution, _params.rho, _params.sigma);
            _solution.dX_hat = out.x;
            _solution.P      = out.P;
            _solution.p      = out.p;

#ifdef DEBUG
            std::cout << "LQR solution, iteration " << _iter + 1 << std::endl;
            std::cout << "=============" << std::endl;
            std::cout << "x:" << std::endl;
            std::cout << out.x << std::endl;
            std::cout << "u:" << std::endl;
            std::cout << out.u << std::endl;
            std::cout << "=============" << std::endl;
#endif

            MatX z_hat(_qp.nc, _qp.horizon + 1);

            _step(_solution.dX_hat,
                  out.u,
                  _solution.dX,
                  _solution.dU,
                  _solution.y,
                  _solution.z,
                  z_hat);

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

            if ((_iter + 1) % _params.rho_update_iter_freq)
            {
                double r_prim_eucn = _r_prim.norm();
                double r_dual_eucn = _r_dual.norm();

                if (r_prim_eucn > _params.rho_incr_factor * r_dual_eucn)
                {
                    _params.rho *= _params.rho_scale_factor;
                }
                else if (r_dual_eucn > _params.rho_decr_factor * r_prim_eucn)
                {
                    _params.rho /= _params.rho_scale_factor;
                }
            }

#ifdef DEBUG
            std::cout << "rho: " << _params.rho << std::endl;
#endif
            // Termination
            double tolerance_prim =
              _params.tolerance_abs * std::sqrt(_qp.nc) +
              _params.tolerance_rel *
                std::max(z_hat.norm(), _solution.z.norm());

            MatX PTy(_qp.nx + _qp.nu, _qp.horizon + 1);
            for (uint k = 0; k < _qp.horizon; ++k)
            {
                PTy.col(k).segment(0, _qp.nx) =
                  _qp.stages[k].D.transpose() * _solution.y.col(k);
                PTy.col(k).segment(_qp.nx, _qp.nu) =
                  _qp.stages[k].E.transpose() * _solution.y.col(k);
            }

            PTy.col(_qp.horizon).segment(0, _qp.nx) =
              _qp.Dt.transpose() * _solution.y.col(_qp.horizon);
            PTy.col(_qp.horizon).segment(_qp.nx, _qp.nu) = VecX::Zero(_qp.nu);

            double tolerance_dual =
              _params.tolerance_abs * std::sqrt(_qp.nx + _qp.nu) +
              _params.tolerance_rel * (_params.rho * PTy).norm();

#ifdef DEBUG
            std::cout << "Tolerances" << std::endl;
            std::cout << "=============" << std::endl;
            std::cout << "Primal: " << tolerance_prim << std::endl;
            std::cout << "Dual: " << tolerance_dual << std::endl;
            std::cout << "=============" << std::endl;
#endif

            if (_r_prim.norm() < tolerance_prim &&
                _r_dual.norm() < tolerance_dual)
            {
                break;
            }
        }

        return _solution;
    }

  protected:
    QPData     _qp;
    Guess      _guess;
    admmParams _params;

    MatX _r_prim, _r_dual;

    uint _iter;

    admmQPSolution _solution;

    void _step(const MatX &dX_hat, const MatX &dU_hat, MatX &dX, MatX &dU,
               MatX &y, MatX &z, MatX &z_hat)
    {
        for (uint k = 0; k < _qp.horizon; ++k)
        {
            z_hat.col(k) = _params.alpha * (_qp.stages[k].D * dX_hat.col(k) +
                                            _qp.stages[k].E * dU_hat.col(k)) +
                           (1 - _params.alpha) * z.col(k);
            dX.col(k) =
              _params.alpha * dX_hat.col(k) + (1 - _params.alpha) * dX.col(k);
            dU.col(k) =
              _params.alpha * dU_hat.col(k) + (1 - _params.alpha) * dU.col(k);

            VecX z_prev = z.col(k);
            z.col(k)    = (-_qp.stages[k].c)
                         .cwiseMax(z_hat.col(k) + 1 / _params.rho * y.col(k));
            y.col(k) += _params.rho * (z_hat.col(k) - z.col(k));

            _r_prim.col(k) = z_hat.col(k) - z.col(k);

            VecX z_diff = z.col(k) - z_prev;
            _r_dual.col(k).segment(0, _qp.nx) =
              _params.rho * _qp.stages[k].D.transpose() * z_diff;
            _r_dual.col(k).segment(_qp.nx, _qp.nu) =
              _params.rho * _qp.stages[k].E.transpose() * z_diff;
        }
        z_hat.col(_qp.horizon) =
          _params.alpha * _qp.Dt * dX_hat.col(_qp.horizon) +
          (1 - _params.alpha) * z.col(_qp.horizon);
        dX.col(_qp.horizon) = _params.alpha * dX_hat.col(_qp.horizon) +
                              (1 - _params.alpha) * dX.col(_qp.horizon);

        VecX z_prev        = z.col(_qp.horizon);
        z.col(_qp.horizon) = (-_qp.ct).cwiseMax(
          z_hat.col(_qp.horizon) + 1 / _params.rho * y.col(_qp.horizon));
        y.col(_qp.horizon) +=
          _params.rho * (z_hat.col(_qp.horizon) - z.col(_qp.horizon));

        _r_prim.col(_qp.horizon) = z_hat.col(_qp.horizon) - z.col(_qp.horizon);

        VecX z_diff = z.col(_qp.horizon) - z_prev;
        _r_dual.col(_qp.horizon).segment(0, _qp.nx) =
          _params.rho * _qp.Dt.transpose() * z_diff;
        _r_dual.col(_qp.horizon).segment(_qp.nx, _qp.nu) = VecX::Zero(_qp.nu);
    }
};
}
}

#endif
