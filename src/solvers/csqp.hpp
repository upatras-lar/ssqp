#include <iostream>
#include <vector>
#include <functional>

#include "types.hpp"
#include "model/constraint/AbstractConstraint.hpp"
#include "model/cost/AbstractCost.hpp"
#include "model/dynamics/AbstractDynamics.hpp"

#include "solvers/admm_qp.hpp"
#include "sqp_step/gss.hpp"
#include "sqp_step/filter.hpp"

#ifndef SSQP_CSQP_HPP
#define SSQP_CSQP_HPP

namespace ssqp
{
namespace solvers
{
struct csqpParams
{
    uint   max_iter  = 10'000;
    double tolerance = 1e-2;
};

class csqp
{
  public:
    csqp(const OCP &ocp, const csqpParams &params = {})
      : _params(params)
      , _ocp(ocp)
      , _horizon(ocp.horizon)
      , _nx(ocp.dynamics->output_size())
      , _nu(ocp.dynamics->control_size())
      , _nc(ocp.constraints->output_size())
      , _filter_ls_costs(
          { [this](const MatX &x, const MatX &u) { return this->_lf(x, u); },
            [this](const MatX &x, const MatX &u)
            { return this->_gammaf(x, u); },
            [this](const MatX &x, const MatX &u) { return this->_cf(x, u); } })
      , _guess({ MatX::Zero(_nx, _horizon + 1),
                 MatX::Zero(_nu, _horizon),
                 MatX::Zero(_nx, _horizon),
                 MatX::Zero(_nc, _horizon + 1) })
    {
        /*
        for (int i = 0; i < _horizon; ++i)
        {
            _guess.x(0, i) = static_cast<double>(rand() % 100) / 100;
            _guess.x(1, i) = static_cast<double>(rand() % 100) / 100;
            _guess.u(0, i) = static_cast<double>(rand() % 100) / 100;
        }
        */
    }

    Guess solve()
    {
        for (_iter = 0;; ++_iter)
        {
            QPData         qp = _setup_qp();
            admm_qp        qp_solver(qp, _guess);
            admmQPSolution sol = qp_solver.solve();

            // Guess (x, u) update
            std::tie(_guess.x, _guess.u, std::ignore) = step::filter(_guess.x,
                                                               _guess.u,
                                                               sol.dX,
                                                               sol.dU,
                                                               _filter_ls_costs);

            // Lagrange multiplier update
            for (uint k = 0; k < _horizon; ++k)
            {
                _guess.lambda.col(k) = sol.P[k + 1] * sol.dX_hat.col(k + 1) + sol.p[k + 1];
            }
            _guess.mu = -sol.y; // we need a minus because the QP (ADMM) assumes
                                // the opposite sign in the Lagrangian

            // Termination
            if (_check_termination())
            {
                break;
            }
        }

        return _guess;
    }

  protected:
    const int _horizon;
    const int _nx, _nu, _nc;

    const std::vector<std::function<double(const MatX &, const MatX &)>>
      _filter_ls_costs;

    csqpParams _params;
    OCP        _ocp;
    Guess      _guess;

    uint _iter;

    QPData _setup_qp()
    {
        QPData qp(_horizon, _nx, _nu, _nc);

        VecX tmpVec;
        MatX tmpMat;

        for (uint k = 0; k < _horizon; ++k)
        {
            // Get A_k and B_k matrices
            _ocp.dynamics->gradient(k,
                                    _guess.x.col(k),
                                    _guess.u.col(k),
                                    qp.stages[k].A,
                                    qp.stages[k].B);

            // Get gamma_k
            qp.stages[k].gamma =
              _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k)) -
              _guess.x.col(k + 1);

            // Calculate Q_k
            _ocp.costs->hessian_xx(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   qp.stages[k].Qxx);

            _ocp.dynamics->hessprod_xx(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmpMat);
            qp.stages[k].Qxx += tmpMat;

            _ocp.constraints->hessprod_xx(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmpMat);
            qp.stages[k].Qxx += tmpMat;

            // Calculate S_k
            _ocp.costs->hessian_xu(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   qp.stages[k].Qxu);

            _ocp.dynamics->hessprod_xu(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmpMat);
            qp.stages[k].Qxu += tmpMat;

            _ocp.constraints->hessprod_xu(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmpMat);
            qp.stages[k].Qxu += tmpMat;

            // Calculate R_k
            _ocp.costs->hessian_uu(k,
                                   _guess.x.col(k),
                                   _guess.u.col(k),
                                   qp.stages[k].Quu);

            _ocp.dynamics->hessprod_uu(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       _guess.lambda.col(k),
                                       tmpMat);
            qp.stages[k].Quu += tmpMat;

            _ocp.constraints->hessprod_uu(k,
                                          _guess.x.col(k),
                                          _guess.u.col(k),
                                          _guess.mu.col(k),
                                          tmpMat);
            qp.stages[k].Quu += tmpMat;

            // Get q and r
            _ocp.costs->gradient(k,
                                 _guess.x.col(k),
                                 _guess.u.col(k),
                                 qp.stages[k].qx,
                                 qp.stages[k].qu);

            // Get D and E
            _ocp.constraints->gradient(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       qp.stages[k].D,
                                       qp.stages[k].E);

            // Get c
            qp.stages[k].c =
              _ocp.constraints->evaluate(k, _guess.x.col(k), _guess.u.col(k));
        }

        // Terminal values for Q, q, D and c
        _ocp.costs->hessian_xx(_horizon,
                               _guess.x.col(_horizon),
                               VecX::Zero(1),
                               qp.Qxx_t);
        _ocp.constraints->hessprod_xx(_horizon,
                                      _guess.x.col(_horizon),
                                      VecX::Zero(1),
                                      _guess.mu.col(_horizon),
                                      tmpMat);
        qp.Qxx_t += tmpMat;

        _ocp.costs->gradient(_horizon,
                             _guess.x.col(_horizon),
                             VecX::Zero(1),
                             qp.qx_t,
                             tmpVec);

        _ocp.constraints->gradient(_horizon,
                                   _guess.x.col(_horizon),
                                   VecX::Zero(1),
                                   qp.Dt,
                                   tmpMat);

        qp.ct = _ocp.constraints->evaluate(_horizon,
                                           _guess.x.col(_horizon),
                                           VecX::Zero(1));

        return qp;
    }

    bool _check_termination()
    {
// #ifdef DEBUG
        std::cout << "Guess " << _iter + 1 << std::endl;
        std::cout << "=============" << std::endl;

        std::cout << "x: " << std::endl << _guess.x << std::endl;
        std::cout << "u: " << std::endl << _guess.u << std::endl;
        std::cout << "lambda: " << std::endl << _guess.lambda << std::endl;
        std::cout << "mu: " << std::endl << _guess.mu << std::endl;

        std::cout << "=============" << std::endl;
// #endif

        if (_iter == _params.max_iter)
        {
            std::cout << "CSQP: Max iteration reached" << std::endl;
            return true;
        }

        // Terminal constraint violation
        if (std::max(
              -_ocp.constraints
                 ->evaluate(_horizon, _guess.x.col(_horizon), VecX::Zero(1))
                 .lpNorm<Eigen::Infinity>(),
              0.0) > _params.tolerance)
        {
            return false;
        }

        for (uint k = 0; k < _horizon; ++k)
        {
            // Stage gap violation
            if ((_guess.x.col(k + 1) -
                 _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k)))
                  .lpNorm<Eigen::Infinity>() > _params.tolerance)
            {
                return false;
            }
            // Stage constraint violation
            if (std::max(-_ocp.constraints
                            ->evaluate(k, _guess.x.col(k), _guess.u.col(k))
                            .lpNorm<Eigen::Infinity>(),
                         0.0) > _params.tolerance)
            {
                return false;
            }
        }

        // Jacobian over x and y of lagrangian at current guess
        VecX lx_val = VecX::Zero(_nx * (_horizon + 1));
        VecX lu_val = VecX::Zero(_nu * _horizon);

        VecX dx, du;

        // Add terminal terms for x
        _ocp.costs->gradient(_horizon,
                             _guess.x.col(_horizon),
                             VecX::Zero(_nu),
                             dx,
                             du);

        lx_val.tail(_nx) += dx;
        // Add contribution from previous integration step (λ)
        lx_val.tail(_nx) -= _guess.lambda.col(_horizon - 1);

        MatX dX, dU;
        // Add constraint terms for terminal x
        _ocp.constraints->gradient(_horizon,
                                   _guess.x.col(_horizon),
                                   VecX::Zero(_nu),
                                   dX,
                                   dU);

        lx_val.tail(_nx) += dX.transpose() * _guess.mu.col(_horizon);

        // Add k=0 terms for u
        _ocp.costs->gradient(0, _guess.x.col(0), _guess.u.col(0), dx, du);
        lu_val.head(_nu) += du;

        _ocp.dynamics->gradient(0, _guess.x.col(0), _guess.u.col(0), dX, dU);
        lu_val.head(_nu) += dU.transpose() * _guess.lambda.col(0);

        _ocp.constraints->gradient(0, _guess.x.col(0), _guess.u.col(0), dX, dU);
        lu_val.head(_nu) -= dU.transpose() * _guess.mu.col(0);

        // Ignore x0 terms since it is not being optimized
        for (uint k = 1; k < _horizon; ++k)
        {
            // Compute values of jacobian of lagrangian
            _ocp.costs->gradient(k, _guess.x.col(k), _guess.u.col(k), dx, du);
            lx_val.segment(k * _nx, _nx) += dx;
            lu_val.segment(k * _nu, _nu) += du;

            _ocp.dynamics->gradient(k,
                                    _guess.x.col(k),
                                    _guess.u.col(k),
                                    dX,
                                    dU);

            lx_val.segment(k * _nx, _nx) +=
              dX.transpose() * _guess.lambda.col(k);
            lx_val.segment(k * _nx, _nx) -= _guess.lambda.col(k - 1);

            lu_val.segment(k * _nu, _nu) +=
              dU.transpose() * _guess.lambda.col(k);

            _ocp.constraints->gradient(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       dX,
                                       dU);
            lx_val.segment(k * _nx, _nx) -= dX.transpose() * _guess.mu.col(k);
            lu_val.segment(k * _nu, _nu) -= dU.transpose() * _guess.mu.col(k);
        }

// #ifdef DEBUG
        std::cout << "lx_val: " << lx_val.transpose() << std::endl;
        std::cout << lx_val.lpNorm<Eigen::Infinity>() << std::endl;
        std::cout << "lu_val: " << lu_val.transpose() << std::endl;
        std::cout << lu_val.lpNorm<Eigen::Infinity>() << std::endl;
// #endif

        if (lx_val.lpNorm<Eigen::Infinity>() <= _params.tolerance &&
            lu_val.lpNorm<Eigen::Infinity>() <= _params.tolerance)
        {
            std::cout << "CSQP: Convergence reached in " << _iter + 1
                      << " iterations" << std::endl;
            return true;
        }

        return false;
    }

    // Trajectory cost
    double _lf(const MatX &x, const MatX &u) const
    {
        double sum = 0.0;
        for (uint k = 0; k < _horizon; ++k)
        {
            sum += _ocp.costs->evaluate(k, x.col(k), u.col(k));
        }

        return sum +
               _ocp.costs->evaluate(_horizon, x.col(_horizon), VecX::Zero(1));
    }

    // Gap violation
    double _gammaf(const MatX &x, const MatX &u) const
    {
        double sum = 0.0;
        for (uint k = 0; k < _horizon; ++k)
        {
            sum +=
              (x.col(k + 1) - _ocp.dynamics->evaluate(k, x.col(k), u.col(k)))
                .lpNorm<Eigen::Infinity>();
        }

        return sum;
    }

    // Constraint violation
    double _cf(const MatX &x, const MatX &u) const
    {
        double sum = 0.0;
        for (uint k = 0; k < _horizon; ++k)
        {
            sum += _ocp.constraints->evaluate(k, x.col(k), u.col(k))
                     .lpNorm<Eigen::Infinity>();
        }

        return sum + _ocp.constraints
                       ->evaluate(_horizon, x.col(_horizon), VecX::Zero(1))
                       .lpNorm<Eigen::Infinity>();
    }
};
}
}

#endif
