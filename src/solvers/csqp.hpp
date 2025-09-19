#include "types.hpp"
#include "constraint/AbstractConstraint.hpp"
#include "cost/AbstractCost.hpp"
#include "dynamics/AbstractDynamics.hpp"

#include "solvers/admm_qp.hpp"

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
      , _guess(
          { Eigen::MatrixXd::Zero(ocp.dynamics->output_size(), _horizon + 1),
            Eigen::MatrixXd::Zero(ocp.dynamics->control_size(), _horizon),
            Eigen::MatrixXd::Zero(ocp.dynamics->output_size(), _horizon),
            Eigen::MatrixXd::Zero(ocp.constraints->output_size(),
                                  _horizon + 1) })
    {
        for (int i = 0; i < _horizon; ++i)
        {
            _guess.x(0, i) = static_cast<double>(rand() % 100) / 100;
            _guess.x(1, i) = static_cast<double>(rand() % 100) / 100;
            _guess.u(0, i) = static_cast<double>(rand() % 100) / 100;
        }
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
            _golden_section_search(_guess.x, _guess.u, sol.dX, sol.dU);

            // Lagrange multiplier update
            for (uint k = 0; k < _horizon; ++k)
            {
                _guess.lambda.col(k) = sol.V[k] * sol.dX_hat.col(k) + sol.u[k];
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

    OCP   _ocp;
    Guess _guess;

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

        return lx_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               lu_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               dyn_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               con_val.lpNorm<Eigen::Infinity>() <= _tolerance &&
               tcon_val.lpNorm<Eigen::Infinity>() <= _tolerance;
    }

    // Line search, updating x, u
    void _golden_section_search(MatX &x, MatX &u, MatX &dX, MatX &dU)
    {
        static const double invphi    = (std::sqrt(5) - 1) / 2;
        static const double tolerance = 1e-5;

        MatX a1 = x;
        MatX a2 = u;

        MatX b1 = x + dX;
        MatX b2 = u + dU;

        while ((b1 - a1).maxCoeff() > tolerance ||
               (b2 - a2).maxCoeff() > tolerance)
        {
            MatX c1 = b1 - (b1 - a1) * invphi;
            MatX c2 = b2 - (b2 - a2) * invphi;

            MatX d1 = a1 + (b1 - a1) * invphi;
            MatX d2 = a2 + (b2 - a2) * invphi;

            // TODO: this assumes c or d are min
            if (_isCostSmaller(c1, c2, d1, d2))
            {
                b1 = d1;
                b2 = d2;
            }
            else
            {
                a1 = c1;
                a2 = c2;
            }
        }

        x = (b1 + a1) / 2;
        u = (b2 + a2) / 2;
    }

    // Line search cost functions

    // Using any cost function (filter search), returns whether (x1, u1) is
    // smaller than (x2, u2)
    // TODO: reuse function evaluations
    bool _isCostSmaller(const MatX &x1, const MatX &u1, const MatX &x2,
                        const MatX &u2)
    {
        return _lf(x1, u1) < _lf(x2, u2) || _gammaf(x1, u1) < _gammaf(x2, u2) ||
               _cf(x1, u1) < _cf(x2, u2);
    }

    // Trajectory cost
    double _lf(const MatX &x, const MatX &u)
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
    double _gammaf(const MatX &x, const MatX &u)
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
    double _cf(const MatX &x, const MatX &u)
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
