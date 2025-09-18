#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
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
        std::vector<MatX> V;
        std::vector<VecX> u;
        MatX              y;
    };

    admm_qp(const OCP &ocp, const Guess &guess)
      : _ocp(ocp)
      , _guess(guess)
      , _horizon(guess.u.cols())
      , _nx(guess.x.rows())
      , _nu(guess.u.rows())
      , _nc(guess.mu.rows())
      , _stages(std::vector<StageData>(_horizon + 1))
      , _lqr_stages(std::vector<LQRStage>(_horizon + 1))
    {
        assert(guess.x.cols() == guess.u.cols() + 1 &&
               "Inconsistent time horizon"); // ensure consistent time horizon

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
            _stages[k + 1].gamma =
              _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k)) -
              _guess.x.col(k); // TODO: gamma_0 ?

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
        // Initialization
        _solution.dX = MatX::Zero(_nx, _horizon + 1);
        _solution.dU = MatX::Zero(_nu, _horizon);
        _solution.y  = MatX::Zero(_nc, _horizon + 1);
        MatX z       = MatX::Zero(_nc, _horizon + 1);

        _solution.dX_hat = MatX(_solution.dX.rows(), _solution.dX.cols());
        MatX dU_hat(_solution.dU.rows(), _solution.dU.cols());

        _solution.V = std::vector<MatX>(_horizon);
        _solution.u = std::vector<VecX>(_horizon);

        for (uint i = 0;; ++i)
        {
            if (!(i % _lqr_repeat_iter))
            {
                // TODO: rho update
                _lqr(_solution.dX, _solution.dU, _solution.dX_hat, dU_hat);
            }

            _step(_solution.dX_hat,
                  dU_hat,
                  _solution.dX,
                  _solution.dU,
                  _solution.y,
                  z);

            // TODO: termination
            if (i == _lqr_repeat_iter - 1) break;
        }

        return _solution;
    }

  protected:
    // Data for each stage of the time horizon
    // TODO: are sizes correct?
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

    // Data for LQR
    struct LQRStage
    {
        VecX h;
        MatX G, H, V;
        VecX u;
        MatX K;
        VecX k;
    };

    uint _horizon;
    uint _nx, _nu, _nc;

    // Vector of stage data (including final stage, size H + 1)
    std::vector<StageData> _stages;
    std::vector<LQRStage>  _lqr_stages;

    OCP   _ocp;
    Guess _guess;

    Solution _solution;

    const double _alpha = 1.6;
    const double _sigma = 10e-6;
    double       _rho   = 0.1;

    const int _lqr_repeat_iter = 15;

    void _lqr(const MatX &dX, const MatX &dU, MatX &dX_hat, MatX &dU_hat)
    {
        // LQR step
        // Initialization
        _lqr_stages[_horizon].V = _stages[_horizon].Q;
        _lqr_stages[_horizon].u = _stages[_horizon].q;

        // TODO: Add additional terms from (15)
        // TODO: figure out indexing (paper is H-1 to 1)

        // Backward pass
        for (int k = _horizon - 1; k >= 0; --k)
        {
            _lqr_stages[k].h =
              _stages[k].r + _stages[k].B.transpose() *
                               (_lqr_stages[k + 1].u +
                                _lqr_stages[k + 1].V * _stages[k + 1].gamma);

            _lqr_stages[k].G =
              _stages[k].S.transpose() +
              _stages[k].B.transpose() * _lqr_stages[k + 1].V * _stages[k].A;

            _lqr_stages[k].H = _stages[k].R + _stages[k].B.transpose() *
                                                _lqr_stages[k + 1].V *
                                                _stages[k].B;

            // TODO: is LLT appropriate?
            if (_nu > 1)
            {
                Eigen::LLT H_llt = _lqr_stages[k].H.llt();
                _lqr_stages[k].K = -H_llt.solve(_lqr_stages[k].G);
                _lqr_stages[k].k = -H_llt.solve(_lqr_stages[k].h);
            }
            else
            {
                double coeff     = 1 / _lqr_stages[k].H(0, 0);
                _lqr_stages[k].K = -coeff * _lqr_stages[k].G;
                _lqr_stages[k].k = -coeff * _lqr_stages[k].h;
            }

            _lqr_stages[k].V =
              _stages[k].Q +
              _stages[k].A.transpose() * _lqr_stages[k + 1].V * _stages[k].A -
              _lqr_stages[k].K.transpose() * _lqr_stages[k].H *
                _lqr_stages[k].K;

            _lqr_stages[k].u =
              _stages[k].q + _lqr_stages[k].K.transpose() * _stages[k].r +
              (_stages[k].A + _stages[k].B * _lqr_stages[k].K).transpose() *
                (_lqr_stages[k + 1].u +
                 _lqr_stages[k + 1].V * _stages[k + 1].gamma);

            _solution.V[k] = _lqr_stages[k].V;
            _solution.u[k] = _lqr_stages[k].u;
        }

        // Forward pass
        for (uint k = 0; k < _horizon; ++k)
        {
            dX_hat.col(k + 1) =
              (_stages[k].A + _stages[k].B * _lqr_stages[k].K) * dX.col(k) +
              _stages[k].B * _lqr_stages[k].k + _stages[k + 1].gamma;
            dU_hat.col(k) = _lqr_stages[k].K * dX.col(k) + _lqr_stages[k].k;
        }
    }

    void _step(const MatX &dX_hat, const MatX &dU_hat, MatX &dX, MatX &dU,
               MatX &y, MatX &z)
    {
        VecX z_hat;
        for (uint k = 1; k < _horizon; ++k)
        {
            z_hat = _alpha * (_stages[k].D * dX_hat.col(k) +
                              _stages[k].E * dU_hat.col(k)) +
                    (1 - _alpha) * z.col(k);
            dX.col(k) = _alpha * dX_hat.col(k) + (1 - _alpha) * dX.col(k);
            dU.col(k) = _alpha * dU_hat.col(k) + (1 - _alpha) * dU.col(k);
            z.col(k)  = _stages[k].c.cwiseMax(z_hat + 1 / _rho * y.col(k));
            y.col(k)  = y.col(k) + _rho * (z_hat - z.col(k + 1));
        }

        z_hat = _alpha * _stages[_horizon].D * dX_hat.col(_horizon) +
                (1 - _alpha) * z.col(_horizon);
        dX.col(_horizon) =
          _alpha * dX_hat.col(_horizon) + (1 - _alpha) * dX.col(_horizon);

        VecX tst = z_hat + 1 / _rho * y.col(_horizon);

        z.col(_horizon) =
          _stages[_horizon].c.cwiseMax(z_hat + 1 / _rho * y.col(_horizon));
        y.col(_horizon) = y.col(_horizon) + _rho * (z_hat - z.col(_horizon));
    }
};
}
}

#endif
