#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "solvers/lqr.hpp"
#include "constraint/AbstractConstraint.hpp"
#include "cost/AbstractCost.hpp"
#include "dynamics/AbstractDynamics.hpp"

#ifndef SSQP_ADMM_QP_HPP
#define SSQP_ADMM_QP_HPP

namespace ssqp
{
namespace solvers
{
class admm_qp
{
  public:
    struct Solution
    {
        MatX              dX, dU;
        MatX              dX_hat;
        std::vector<MatX> P;
        std::vector<VecX> p;
        MatX              y, z;
    };

    admm_qp(const OCP &ocp, const Guess &guess)
      : _ocp(ocp)
      , _guess(guess)
      , _horizon(guess.u.cols())
      , _nx(guess.x.rows())
      , _nu(guess.u.rows())
      , _nc(guess.mu.rows())
      , _stages(std::vector<StageData>(_horizon + 1))
    {
        assert(guess.x.cols() == guess.u.cols() + 1 &&
               "Inconsistent time horizon"); // ensure consistent time horizon

        _solution.dX = MatX::Zero(_nx, _horizon + 1);
        _solution.dU = MatX::Zero(_nu, _horizon);
        _solution.y  = MatX::Zero(_nc, _horizon + 1);
        _solution.z  = MatX::Zero(_nc, _horizon + 1);

        MatX tmp;
        VecX tmp2;

        for (uint k = 0; k < _horizon; ++k)
        {
            // Get A_k and B_k matrices
            _ocp.dynamics->gradient(k,
                                    _guess.x.col(k),
                                    _guess.u.col(k),
                                    _stages[k].A,
                                    _stages[k].B);

            // Get gamma_k
            _stages[k].gamma =
              _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k)) -
              _guess.x.col(k + 1);

            // Calculate Q_k
            _ocp.costs->hessian_xx(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   _stages[k].Q);

            _ocp.dynamics->hessprod_xx(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmp);
            _stages[k].Q += tmp;

            _ocp.constraints->hessprod_xx(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmp);
            _stages[k].Q += tmp;

            // Calculate S_k
            _ocp.costs->hessian_xu(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   _stages[k].S);

            _ocp.dynamics->hessprod_xu(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmp);
            _stages[k].S += tmp;

            _ocp.constraints->hessprod_xu(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmp);
            _stages[k].S += tmp;

            // Calculate R_k
            _ocp.costs->hessian_uu(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   _stages[k].R);

            _ocp.dynamics->hessprod_uu(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmp);
            _stages[k].R += tmp;

            _ocp.constraints->hessprod_uu(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmp);
            _stages[k].R += tmp;

            // Get q and r
            _ocp.costs->gradient(k,
                                 _guess.x.col(k),
                                 _guess.u.col(k),
                                 _stages[k].q,
                                 _stages[k].r);

            // Get D and E
            _ocp.constraints->gradient(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _stages[k].D,
                                       _stages[k].E);

            // Get c
            _stages[k].c =
              _ocp.constraints->evaluate(k, _guess.x.col(k), _guess.u.col(k));
        }

        // Terminal values for Q, q, D and c
        _ocp.costs->hessian_xx(_horizon,
                               _guess.x.col(_horizon),
                               VecX::Zero(1),
                               _stages[_horizon].Q);
        _ocp.constraints->hessprod_xx(_horizon,
                                      _guess.x.col(_horizon),
                                      VecX::Zero(1),
                                      _guess.mu.col(_horizon),
                                      tmp);
        _stages[_horizon].Q += tmp;

        _ocp.costs->gradient(_horizon,
                             _guess.x.col(_horizon),
                             VecX::Zero(1),
                             _stages[_horizon].q,
                             tmp2);

        _ocp.constraints->gradient(_horizon,
                                   _guess.x.col(_horizon),
                                   VecX::Zero(1),
                                   _stages[_horizon].D,
                                   tmp);

        _stages[_horizon].c = _ocp.constraints->evaluate(_horizon,
                                                         _guess.x.col(_horizon),
                                                         VecX::Zero(1));
    }

    Solution solve()
    {
        MatX   r(_nc, _horizon + 1);
        MatX   sx(_nx, _horizon + 1), su(_nu, _horizon);
        double r_prim_norm, r_dual_norm;

        for (uint i = 0; i < 100; ++i)
        {
            // _solution.y.setZero();
            // _solution.z.setZero();

            solvers::lqr::Output out = _lqr();
            _solution.dX_hat         = out.x;
            _solution.P              = out.P;
            _solution.p              = out.p;

            MatX prev_z = _solution.z;
            _step(_solution.dX_hat,
                  out.u,
                  _solution.dX,
                  _solution.dU,
                  _solution.y,
                  _solution.z);

            MatX dz = _solution.z - prev_z;

            // Calculate residuals

            // TODO: correct rho update should ensure constraints aren't
            // violated
            r_prim_norm = 0.0;
            r_dual_norm = 0.0;
            for (int k = 0; k <= _horizon; ++k)
            {
                r.col(k) =
                  _stages[k].D * _solution.dX.col(k) - _solution.z.col(k);
                if (k != _horizon)
                {
                    r.col(k) += _stages[k].E * _solution.dU.col(k);
                }
                r_prim_norm += r.col(k).squaredNorm();

                sx.col(k) = _stages[k].D.transpose() * dz.col(k);

                r_dual_norm += sx.col(k).squaredNorm();

                if (k != _horizon)
                {
                    su.col(k)    = _stages[k].E.transpose() * dz.col(k);
                    r_dual_norm += su.col(k).squaredNorm();
                }
            }

            r_prim_norm = std::sqrt(r_prim_norm);
            r_dual_norm = std::sqrt(r_dual_norm);

            if (r_prim_norm > 10 * r_dual_norm)
            {
                _rho *= 2;
            }
            else if (r_dual_norm > 10 * r_prim_norm)
            {
                _rho /= 2;
            }

            // TODO: termination
        }

        return _solution;
    }

  protected:
    // Data for each stage of the time horizon
    struct StageData
    {
        MatX A;       // (nx) x (nx)
        MatX B;       // (nx) x (nu)
        VecX gamma;   // (nx) x 1
        MatX Q, S, R; // Q: (nx) x (nx), S: (nx) x (nu), R: (nu) x (nu)
        VecX q, r;    // q: (nx) x 1, r: (nu) x 1
        MatX D, E;    // D: (nc) x (nx), E: (nc) x (nu)
        VecX c;       // (nc) x 1
    };

    uint _horizon;
    uint _nx, _nu, _nc;

    // Vector of stage data (including final stage, size H + 1)
    std::vector<StageData> _stages;

    OCP   _ocp;
    Guess _guess;

    Solution _solution;

    const double _alpha = 1.6;
    const double _sigma = 10e-6;
    double       _rho   = 0.1;

    const int _lqr_repeat_iter = 25;

    solvers::lqr::Output _lqr()
    {
        solvers::lqr::Input input(_horizon, _nx, _nu);

        for (int k = 0; k < _horizon; ++k)
        {
            input.A[k]     = _stages[k].A;
            input.B[k]     = _stages[k].B;
            input.gamma[k] = _stages[k].gamma;
            input.Qxx[k]   = _stages[k].Q;
            input.Qxu[k]   = _stages[k].S;
            input.Quu[k]   = _stages[k].R;
            input.qx[k]    = _stages[k].q;
            input.qu[k]    = _stages[k].r;
            input.D[k]     = _stages[k].D;
            input.E[k]     = _stages[k].E;
        }

        input.Qt = _stages[_horizon].Q;
        input.qt = _stages[_horizon].q;
        input.Dt = _stages[_horizon].D;

        input.x = _solution.dX;
        input.u = _solution.dU;
        input.z = _solution.z;
        input.y = _solution.y;

        return solvers::lqr::lqr(input, _rho, _alpha, _sigma);
    }

    void _step(const MatX &dX_hat, const MatX &dU_hat, MatX &dX, MatX &dU,
               MatX &y, MatX &z)
    {
        VecX z_hat;
        for (uint k = 0; k < _horizon; ++k)
        {
            z_hat = _alpha * (_stages[k].D * dX_hat.col(k) +
                              _stages[k].E * dU_hat.col(k)) +
                    (1 - _alpha) * z.col(k);
            dX.col(k) = _alpha * dX_hat.col(k) + (1 - _alpha) * dX.col(k);
            dU.col(k) = _alpha * dU_hat.col(k) + (1 - _alpha) * dU.col(k);
            z.col(k)  = _stages[k].c.cwiseMin(z_hat + 1 / _rho * y.col(k));
            y.col(k)  = y.col(k) + _rho * (z_hat - z.col(k));
        }

        z_hat = _alpha * _stages[_horizon].D * dX_hat.col(_horizon) +
                (1 - _alpha) * z.col(_horizon);
        dX.col(_horizon) =
          _alpha * dX_hat.col(_horizon) + (1 - _alpha) * dX.col(_horizon);

        z.col(_horizon) =
          _stages[_horizon].c.cwiseMin(z_hat + 1 / _rho * y.col(_horizon));
        y.col(_horizon) = y.col(_horizon) + _rho * (z_hat - z.col(_horizon));
        // std::cout << "Next step:" << std::endl;
        // std::cout << "z: " << z << std::endl;
        // std::cout << "zhat: " << z_hat << std::endl;
        // std::cout << "y: " << y << std::endl;
    }
};
}
}

#endif
