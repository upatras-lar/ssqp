#include "solvers/csqp.hpp"
#include "model/constraint/AbstractConstraint.hpp"
#include "model/cost/AbstractCost.hpp"
#include "model/dynamics/AbstractDynamics.hpp"
#include "solvers/admm_qp.hpp"
#include "sqp_step/gss.hpp"
#include "sqp_step/filter.hpp"

namespace ssqp
{
namespace solvers
{
csqp::csqp(const OCP &ocp, const Params &params)
  : _params(params)
  , _ocp(ocp)
  , _horizon(ocp.horizon)
  , _nx(ocp.dynamics->output_size())
  , _nu(ocp.dynamics->control_size())
  , _nc(ocp.constraints->output_size())
  , _filter_ls_costs(
      { [this](const MatX &x, const MatX &u) { return this->_lf(x, u); },
        [this](const MatX &x, const MatX &u) { return this->_gammaf(x, u); },
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

SolverState csqp::solve(Guess &guess)
{
    SolverState csqp_state;

    for (_iter = 0;; ++_iter)
    {
        QPData            qp = _setup_qp();
        admm_qp           qp_solver(qp, _guess);
        admm_qp::Solution sol;
        SolverState       state = qp_solver.solve(sol);

        switch (state)
        {
        case MAX_ITER_REACHED:
            std::cout << "Warning: ADMM: Max iterations reached" << std::endl;
            break;
        case PRIMAL_INFEASIBLE:
        case DUAL_INFEASIBLE:
            return state;
        default:
            break;
        }

        // Guess (x, u) update
        std::tie(_guess.x, _guess.u, std::ignore) =
          step::filter(_guess.x, _guess.u, sol.dX, sol.dU, _filter_ls_costs);

        // Lagrange multiplier update
        for (uint k = 0; k < _horizon; ++k)
        {
            _guess.lambda.col(k) =
              sol.P[k + 1] * sol.dX.col(k + 1) + sol.p[k + 1];
        }
        _guess.mu = -sol.y; // we need a minus because the QP (ADMM) assumes
                            // the opposite sign in the Lagrangian

        // Termination
        if ((csqp_state = _check_termination()) != ONGOING)
        {
            break;
        }
    }

    guess = _guess;

    return csqp_state;
}

QPData csqp::_setup_qp()
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

SolverState csqp::_check_termination() const
{
#ifdef DEBUG
    std::cout << "Guess " << _iter + 1 << std::endl;
    std::cout << "=============" << std::endl;

    std::cout << "x: " << std::endl << _guess.x << std::endl;
    std::cout << "u: " << std::endl << _guess.u << std::endl;
    std::cout << "lambda: " << std::endl << _guess.lambda << std::endl;
    std::cout << "mu: " << std::endl << _guess.mu << std::endl;

    std::cout << "=============" << std::endl;
#endif

    if (_iter == _params.max_iter)
    {
        return MAX_ITER_REACHED;
    }

    // Terminal constraint violation
    if (_ocp.constraints
          ->evaluate(_horizon, _guess.x.col(_horizon), VecX::Zero(1))
          .cwiseMin(0)
          .lpNorm<Eigen::Infinity>() > _params.tolerance)
    {
        return ONGOING;
    }

    for (uint k = 0; k < _horizon; ++k)
    {
        // Stage gap violation
        if ((_guess.x.col(k + 1) -
             _ocp.dynamics->evaluate(k, _guess.x.col(k), _guess.u.col(k)))
              .lpNorm<Eigen::Infinity>() > _params.tolerance)
        {
            return ONGOING;
        }
        // Stage constraint violation
        if (_ocp.constraints->evaluate(k, _guess.x.col(k), _guess.u.col(k))
              .cwiseMin(0)
              .lpNorm<Eigen::Infinity>() > _params.tolerance)
        {
            return ONGOING;
        }
    }

    // Jacobian over x and y of lagrangian at current guess
    MatX lagrangian_dx = MatX::Zero(_nx, _horizon + 1);
    MatX lagrangian_du = MatX::Zero(_nu, _horizon);

    VecX dx, du;
    MatX dX, dU;

    // Add terminal terms for x
    _ocp.costs->gradient(_horizon,
                         _guess.x.col(_horizon),
                         VecX::Zero(_nu),
                         dx,
                         du);

    lagrangian_dx.col(_horizon) += dx;
    // Add contribution from previous integration step (λ)
    lagrangian_dx.col(_horizon) -= _guess.lambda.col(_horizon - 1);

    // Add constraint terms for terminal x
    _ocp.constraints->gradient(_horizon,
                               _guess.x.col(_horizon),
                               VecX::Zero(_nu),
                               dX,
                               dU);

    lagrangian_dx.col(_horizon) -= dX.transpose() * _guess.mu.col(_horizon);

    // Add k=0 terms for u
    _ocp.costs->gradient(0, _guess.x.col(0), _guess.u.col(0), dx, du);
    lagrangian_du.col(0) += du;

    _ocp.dynamics->gradient(0, _guess.x.col(0), _guess.u.col(0), dX, dU);
    lagrangian_du.col(0) += dU.transpose() * _guess.lambda.col(0);

    _ocp.constraints->gradient(0, _guess.x.col(0), _guess.u.col(0), dX, dU);
    lagrangian_du.col(0) -= dU.transpose() * _guess.mu.col(0);

    // Ignore x0 terms since it is not being optimized
    for (uint k = 1; k < _horizon; ++k)
    {
        // Compute cost jacobian
        _ocp.costs->gradient(k, _guess.x.col(k), _guess.u.col(k), dx, du);
        lagrangian_dx.col(k) += dx;
        lagrangian_du.col(k) += du;

        // Compute dynamics gap jacobian
        _ocp.dynamics->gradient(k, _guess.x.col(k), _guess.u.col(k), dX, dU);
        lagrangian_dx.col(k) += dX.transpose() * _guess.lambda.col(k);
        lagrangian_dx.col(k) -= _guess.lambda.col(k - 1);
        lagrangian_du.col(k) += dU.transpose() * _guess.lambda.col(k);

        // Compute constraint jacobian
        _ocp.constraints->gradient(k, _guess.x.col(k), _guess.u.col(k), dX, dU);
        lagrangian_dx.col(k) -= dX.transpose() * _guess.mu.col(k);
        lagrangian_du.col(k) -= dU.transpose() * _guess.mu.col(k);
    }

#ifdef DEBUG
    std::cout << "Lagrangian dx:  " << std::endl << lagrangian_dx << std::endl;
    std::cout << "Norm: " << lagrangian_dx.lpNorm<Eigen::Infinity>()
              << std::endl;
    std::cout << "Lagrangian du: " << std::endl << lagrangian_du << std::endl;
    std::cout << "Norm: " << lagrangian_du.lpNorm<Eigen::Infinity>()
              << std::endl;
#endif

    if (lagrangian_dx.lpNorm<Eigen::Infinity>() <= _params.tolerance &&
        lagrangian_du.lpNorm<Eigen::Infinity>() <= _params.tolerance)
    {
        std::cout << "CSQP: Convergence reached in " << _iter + 1
                  << " iterations" << std::endl;
        return SUCCESS;
    }

    return ONGOING;
}

double csqp::_lf(const MatX &x, const MatX &u) const
{
    double sum = 0.0;
    for (uint k = 0; k < _horizon; ++k)
    {
        sum += _ocp.costs->evaluate(k, x.col(k), u.col(k));
    }

    return sum + _ocp.costs->evaluate(_horizon, x.col(_horizon), VecX::Zero(1));
}

double csqp::_gammaf(const MatX &x, const MatX &u) const
{
    double sum = 0.0;
    for (uint k = 0; k < _horizon; ++k)
    {
        sum += (x.col(k + 1) - _ocp.dynamics->evaluate(k, x.col(k), u.col(k)))
                 .lpNorm<Eigen::Infinity>();
    }

    return sum;
}

double csqp::_cf(const MatX &x, const MatX &u) const
{
    double sum = 0.0;
    for (uint k = 0; k < _horizon; ++k)
    {
        sum += _ocp.constraints->evaluate(k, x.col(k), u.col(k))
                 .cwiseMin(0)
                 .lpNorm<Eigen::Infinity>();
    }

    return sum +
           _ocp.constraints->evaluate(_horizon, x.col(_horizon), VecX::Zero(1))
             .cwiseMin(0)
             .lpNorm<Eigen::Infinity>();
}
}
}