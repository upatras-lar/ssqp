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
    uint   max_iter  = 100'000;
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
            /*
            std::cout << "SQP iteration: " << _iter + 1 << std::endl;
            std::cout << "Guess: " << std::endl
                      << "x =" << std::endl
                      << _guess.x << std::endl
                      << "u =" << std::endl
                      << _guess.u << std::endl;*/
            admm_qp qp_solver(_ocp, _guess);

            admm_qp::Solution sol = qp_solver.solve();

            // std::cout << "dX: " << sol.dX << std::endl;
            // std::cout << "dU: " << sol.dU << std::endl;

            // Guess (x, u) update
            std::tie(_guess.x, _guess.u) = step::filter(_guess.x,
                                                        _guess.u,
                                                        sol.dX,
                                                        sol.dU,
                                                        _filter_ls_costs);

            // Lagrange multiplier update
            for (uint k = 0; k < _horizon; ++k)
            {
                _guess.lambda.col(k) = sol.P[k] * sol.dX_hat.col(k) + sol.p[k];
            }
            _guess.mu = sol.y;

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

    // TODO: fix
    // check: criteria 4 & 5 incorrect
    // criteria 3 & 4 should be checked for each k, not sum
    bool _check_termination()
    {
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
        VecX lx_val, lu_val = VecX::Zero(_nu);

        VecX dx, du;
        _ocp.costs->gradient(_horizon,
                             _guess.x.col(_horizon),
                             VecX::Zero(1),
                             lx_val,
                             du);

        MatX dX, dU;
        _ocp.constraints->gradient(_horizon,
                                   _guess.x.col(_horizon),
                                   VecX::Zero(1),
                                   dX,
                                   dU);

        lx_val += _guess.mu.col(_horizon).transpose() * dX;

        for (uint k = 0; k < _horizon; ++k)
        {
            // Compute values of jacobian of lagrangian
            _ocp.costs->gradient(k, _guess.x.col(k), _guess.u.col(k), dx, du);
            lx_val += dx;
            lu_val += du;

            _ocp.dynamics->gradient(k,
                                    _guess.x.col(k),
                                    _guess.u.col(k),
                                    dX,
                                    dU);

            // No term for x_0 since it is fixed
            if (k > 0)
            {
                lx_val += (_guess.lambda.col(k).transpose() * dX).transpose();
                lx_val -= _guess.lambda.col(k - 1);
            }
            lu_val += _guess.lambda.col(k).transpose() * dU;

            _ocp.constraints->gradient(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       dX,
                                       dU);
            lx_val -= (_guess.mu.col(k).transpose() * dX).transpose();
            lu_val -= (_guess.mu.col(k).transpose() * dU).transpose();
        }

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
            // TODO: what is [ck(...)]_{-} ?
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
