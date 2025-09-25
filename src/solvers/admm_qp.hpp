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
        MatX              z, y;
    };

    admm_qp(const OCP &ocp, const Guess &guess)
      : _ocp(ocp)
      , _guess(guess)
      , _horizon(guess.u.cols())
      , _nx(guess.x.rows())
      , _nu(guess.u.rows())
      , _nc(guess.mu.rows())
      , _stages(std::vector<StageData>(_horizon + 1))
      , _r_prim(MatX(_nc, _horizon + 1))
      , _r_dual(MatX(_nc, _horizon + 1))
    {
        assert(guess.x.cols() == guess.u.cols() + 1 &&
               "Inconsistent time horizon"); // ensure consistent time horizon

        _solution.dX = MatX::Zero(_nx, _horizon + 1);
        _solution.dU = MatX::Zero(_nu, _horizon);
        _solution.z  = MatX::Zero(_nc, _horizon + 1);
        _solution.y  = MatX::Zero(_nc, _horizon + 1);

        MatX tmp;
        VecX tmp2;

#ifdef DEBUG
        std::cout << "Initialization" << std::endl;
        std::cout << "=============" << std::endl;
#endif

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

#ifdef DEBUG
            std::cout << "Stage " << k << std::endl;
            std::cout << "=============" << std::endl;
            std::cout << "A:" << std::endl << _stages[k].A << std::endl;
            std::cout << "B:" << std::endl << _stages[k].B << std::endl;
            std::cout << "gamma:" << std::endl << _stages[k].gamma << std::endl;
            std::cout << "Q:" << std::endl << _stages[k].Q << std::endl;
            std::cout << "S:" << std::endl << _stages[k].S << std::endl;
            std::cout << "R:" << std::endl << _stages[k].R << std::endl;
            std::cout << "q:" << std::endl << _stages[k].q << std::endl;
            std::cout << "r:" << std::endl << _stages[k].r << std::endl;
            std::cout << "D:" << std::endl << _stages[k].D << std::endl;
            std::cout << "E:" << std::endl << _stages[k].E << std::endl;
            std::cout << "c:" << std::endl << _stages[k].c << std::endl;
            std::cout << "=============" << std::endl;
            std::cin.get();
#endif
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
#ifdef DEBUG
        std::cout << "Terminal stage " << std::endl;
        std::cout << "=============" << std::endl;
        std::cout << "Q:" << std::endl << _stages[_horizon].Q << std::endl;
        std::cout << "q:" << std::endl << _stages[_horizon].q << std::endl;
        std::cout << "D:" << std::endl << _stages[_horizon].D << std::endl;
        std::cout << "c:" << std::endl << _stages[_horizon].c << std::endl;
        std::cout << "=============" << std::endl;
        std::cin.get();
#endif
    }

    Solution solve()
    {
#ifdef DEBUG
        std::cout << "Solving with ADMM" << std::endl;
        std::cout << "=============" << std::endl;
#endif

        for (uint i = 0; i < 100; ++i)
        {
            solvers::lqr::Output out = _lqr();
            _solution.dX_hat         = out.x;
            _solution.P              = out.P;
            _solution.p              = out.p;

#ifdef DEBUG
            std::cout << "LQR solution, iteration " << i << std::endl;
            std::cout << "=============" << std::endl;
            std::cout << "x:" << std::endl;
            std::cout << out.x << std::endl;
            std::cout << "u:" << std::endl;
            std::cout << out.u << std::endl;
            std::cout << "=============" << std::endl;
#endif
            _step(_solution.dX_hat,
                  out.u,
                  _solution.dX,
                  _solution.dU,
                  _solution.y,
                  _solution.z);

#ifdef DEBUG
            std::cout << "Update step, iteration " << i << std::endl;
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
            std::cin.get();
#endif

            // TODO: correct rho update should ensure constraints aren't
            // violated
            /*
            bool updated = false;
            for (uint k = 0; k < _horizon; ++k)
            {
                VecX c = _ocp.constraints->evaluate(k,
                                                    _solution.dX.col(k),
                                                    _solution.dU.col(k));
                for (uint l = 0; l < _nc; ++l)
                {
                    if (c(l) < 0)
                    {
                        _rho    += 0.3;
                        updated  = true;
                        break;
                    }
                }
            }

            if (!updated)
            {
                _rho = std::max(_rho - 0.05, 0.01);
            }
            */

#ifdef DEBUG
            std::cout << "rho: " << _rho << std::endl;
#endif

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

    const double _alpha = 1;
    const double _sigma = 10e-6;
    double       _rho   = 0.1;

    const int _lqr_repeat_iter = 25;

    MatX _r_prim, _r_dual;

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
            dX.col(k)  = _alpha * dX_hat.col(k) + (1 - _alpha) * dX.col(k);
            dU.col(k)  = _alpha * dU_hat.col(k) + (1 - _alpha) * dU.col(k);
            z.col(k)   = (-_stages[k].c).cwiseMax(z_hat + 1 / _rho * y.col(k));
            y.col(k)  += _rho * (z_hat - z.col(k));
        }
        z_hat = _alpha * _stages[_horizon].D * dX_hat.col(_horizon) +
                (1 - _alpha) * z.col(_horizon);
        dX.col(_horizon) =
          _alpha * dX_hat.col(_horizon) + (1 - _alpha) * dX.col(_horizon);

        z.col(_horizon) =
          (-_stages[_horizon].c).cwiseMax(z_hat + 1 / _rho * y.col(_horizon));
        y.col(_horizon) += _rho * (z_hat - z.col(_horizon));
    }
};
}
}

#endif
