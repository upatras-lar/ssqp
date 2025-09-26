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
class csqp
{
  public:
    csqp(const OCP &ocp)
      : _ocp(ocp)
      , _horizon(10)
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
    }

    Guess solve()
    {
        int i = 0;

        while (true)
        {
            std::cout << "SQP iteration: " << i++ << std::endl;
            std::cout << "Guess: " << std::endl
                      << "x =" << std::endl
                      << _guess.x << std::endl
                      << "u =" << std::endl
                      << _guess.u << std::endl;
            admm_qp qp_solver(_ocp, _guess);

            admm_qp::Solution sol = qp_solver.solve();

            std::cout << "dX: " << sol.dX << std::endl;
            std::cout << "dU: " << sol.dU << std::endl;

            // Guess (x, u) update
            /*
            std::tie(_guess.x, _guess.u) = step::gss(_guess.x,
                                                     _guess.u,
                                                     sol.dX,
                                                     sol.dU,
                                                     _filter_ls_costs,
                                                     { 1, 0.1, 0.1 });
            */
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

            if (i == 50) break;
        }

        return _guess;
    }

  protected:
    const int _horizon;
    int       _nx, _nu, _nc;

    const double _tolerance = 10e-3;
    const std::vector<std::function<double(const MatX &, const MatX &)>>
      _filter_ls_costs;

    OCP   _ocp;
    Guess _guess;

    // TODO: fix
    // check: criteria 4 & 5 incorrect
    // criteria 3 & 4 should be checked for each k, not sum
    bool _check_termination()
    {
        // Jacobian over x and y of lagrangian at current guess
        VecX lx_val, lu_val = VecX::Zero(_nu);
        // Gap violation
        VecX dyn_val = VecX::Zero(_nx);
        // Constraint violation
        VecX con_val = VecX::Zero(_nc);
        // Terminal constraint violation
        VecX tcon_val = VecX::Zero(_nc);

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

        tcon_val = _ocp.constraints->evaluate(_horizon,
                                              _guess.x.col(_horizon),
                                              VecX::Zero(1));

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

            // TODO: check validity
            lx_val += -_guess.lambda.col(k).transpose() * dX;
            if (k > 0)
            {
                lx_val += _guess.lambda.col(
                  k - 1); // TODO: lx_val is scalar, lambda_k is vector
            }
            lu_val += _guess.lambda.col(k).transpose() * dU;

            _ocp.constraints->gradient(k,
                                       _guess.x.col(k),
                                       _guess.u.col(k),
                                       dX,
                                       dU);
            lx_val -= _guess.mu.col(k).transpose() * dX;
            lu_val -= _guess.mu.col(k).transpose() * dU;

            // Stage gap violation
            dyn_val +=
              _guess.x.col(k + 1) -
              _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k));

            // Stage constraint violation
            con_val +=
              _ocp.constraints->evaluate(k, _guess.x.col(k), _guess.u.col(k));
        }

        // std::cout << lx_val.lpNorm<Eigen::Infinity>() << " "
        //           << lu_val.lpNorm<Eigen::Infinity>() << " "
        //           << dyn_val.lpNorm<Eigen::Infinity>() << " "
        //           << con_val.lpNorm<Eigen::Infinity>() << " "
        //           << tcon_val.lpNorm<Eigen::Infinity>() << std::endl;

        return lx_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               lu_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               dyn_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               con_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               tcon_val.lpNorm<Eigen::Infinity>() <= _tolerance;
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
