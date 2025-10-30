#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "sqp_step/filter.hpp"

// TODO: noalias

namespace ssqp
{
namespace solvers
{
ssqp::Guess::Guess(const model::Dimensions &dims,
                   const model::Dimensions &lin_dims)
  : x(dims.state, dims.stage_state)
  , u(dims.control, dims.stage_control)
  , lambda(lin_dims.state, lin_dims.stage_state)
  , mu(dims.constraint, dims.stage_constraint)
{
}

ssqp::ssqp(const model::Model &model, const Params &params)
  : _params(params)
  , _model(model)
  , _filter_ls_costs({ [this](const FlatVec &x, const FlatVec &u)
                       { return this->_lf(x, u); },
                       [this](const FlatVec &x, const FlatVec &u)
                       { return this->_gammaf(x, u); },
                       [this](const FlatVec &x, const FlatVec &u)
                       { return this->_cf(x, u); } })
  , _guess(model.dims(), model.linear_dims())
  , _prev_cost_dx(_guess.x, false)
  , _cost_dx(_guess.x, false)
  , _prev_cost_du(_guess.u, false)
  , _cost_du(_guess.u, false)
{
    model::Dimensions              dims = model.dims();
    std::vector<ManifoldOperators> ops(dims.horizon + 1);

    const model::DynamicsContainer dynamics = model.dynamics();

    for (uint k = 0; k <= dims.horizon; ++k)
    {
        ops[k].right_plus = [dynamics, k](const VecX &x, const VecX &dx)
        { return dynamics[k]->state_right_plus(x, dx); };
        ops[k].right_minus = [dynamics, k](const VecX &x, const VecX &y)
        { return dynamics[k]->state_right_minus(x, y); };
        _guess.x.stage(k) = dynamics[k]->init_state();
    }

    _guess.x.set_ops(ops);
}

SolverState ssqp::solve(Guess &guess)
{
    _stale_iters = 0;

    SolverState ssqp_state;

    DEBUG_LOGLN("Starting SSQP solver loop");

    for (_iter = 0; _iter < _params.max_iter; ++_iter)
    {
        QPData          qp = _setup_qp();
        admm_qp::Params admm_params {};
        admm_params.warm_start = _params.warm_start_admm;

        DEBUG_LOGLN("QP setup complete");

        admm_qp           qp_solver(qp, admm_params);
        admm_qp::Solution sol(qp.dims);
        SolverState       state = qp_solver.solve(sol);

        switch (state)
        {
        case MAX_ITER_REACHED:
            LOGLN("Warning: ADMM: Max iterations reached");
            break;
        case PRIMAL_INFEASIBLE:
        case DUAL_INFEASIBLE:
            return state;
        default:
            break;
        }

        // Guess (x, u) update
        std::tie(_guess.x, _guess.u, std::ignore) =
          step::filter(_guess.x,
                       _guess.u,
                       sol.dx,
                       sol.du,
                       _filter_ls_costs,
                       2,
                       1e-1);

        LOGLN("Current x: ");
        LOGLN(_guess.x);
        LOGLN("Current u: ");
        LOGLN(_guess.u);

        // Lagrange multiplier update
        for (uint k = 0; k < _model.horizon(); ++k)
        {
            _guess.lambda.stage(k).noalias() =
              sol.P[k + 1] * sol.dx.stage(k + 1) + sol.p[k + 1];
        }
        // Need a minus because the QP (ADMM) assumes the opposite sign in the
        // Lagrangian
        _guess.mu = -sol.y;

        // Termination
        if ((ssqp_state = _check_termination()) != ONGOING)
        {
            guess = _guess;
            return ssqp_state;
        }

        _prev_cost_dx = _cost_dx;
        _prev_cost_du = _cost_du;
    }

    guess = _guess;
    return MAX_ITER_REACHED;
}

bool ssqp::write(const std::string &filename)
{
    std::ofstream out;
    out.open(filename + ".trj");
    if (!out)
    {
        return false;
    }
    _guess.x.write_data(out);
    _guess.u.write_data(out);
    out.close();

    return true;
}

QPData ssqp::_setup_qp()
{
    QPData qp(_model.linear_dims());

    VecX tmpVec;
    MatX tmpMat;

    static const model::CostsContainer costs = _model.costs();

    static const model::DynamicsContainer dynamics = _model.dynamics();

    static const model::ConstraintsContainer constraints = _model.constraints();

    for (uint k = 0; k < qp.dims.horizon; ++k)
    {
        // Get A_k and B_k matrices
        dynamics[k]->gradient(_guess.x.stage(k),
                              _guess.u.stage(k),
                              qp.stages[k].A,
                              qp.stages[k].B);

        // Get gamma_k
        qp.stages[k].gamma = dynamics[k]->state_right_minus(
          dynamics[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k)),
          _guess.x.stage(k + 1));

        // Calculate Q_k
        costs[k]->hessian_xx(_guess.x.stage(k),
                             _guess.u.stage(k),
                             qp.stages[k].Qxx);

        dynamics[k]->hessprod_xx(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        qp.stages[k].Qxx += tmpMat;

        // Calculate S_k
        costs[k]->hessian_xu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             qp.stages[k].Qxu);

        dynamics[k]->hessprod_xu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);

        qp.stages[k].Qxu += tmpMat;

        // Calculate R_k
        costs[k]->hessian_uu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             qp.stages[k].Quu);

        dynamics[k]->hessprod_uu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        qp.stages[k].Quu += tmpMat;

        // Get q and r
        costs[k]->gradient(_guess.x.stage(k),
                           _guess.u.stage(k),
                           qp.stages[k].qx,
                           qp.stages[k].qu);

        // Get D and E
        if (constraints[k] != nullptr)
        {
            constraints[k]->hessprod_xx(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            qp.stages[k].Qxx += tmpMat;

            constraints[k]->hessprod_xu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            qp.stages[k].Qxu += tmpMat;

            constraints[k]->hessprod_uu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            qp.stages[k].Quu += tmpMat;

            constraints[k]->gradient(_guess.x.stage(k),
                                     _guess.u.stage(k),
                                     qp.stages[k].D,
                                     qp.stages[k].E);

            // Get cl, cu
            // TODO: manifold constraint elements
            VecX c =
              constraints[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k));

            qp.stages[k].cl = c - constraints[k]->lower_bound();
            qp.stages[k].cu = c - constraints[k]->upper_bound();
        }
        else
        {
            qp.stages[k].D  = MatX::Zero(0, qp.dims.stage_state[k]);
            qp.stages[k].E  = MatX::Zero(0, qp.dims.stage_control[k]);
            qp.stages[k].cl = VecX::Zero(0);
            qp.stages[k].cu = VecX::Zero(0);
        }
    }

    // Terminal values for Q, q, D and c
    costs[qp.dims.horizon]->hessian_xx(_guess.x.stage(qp.dims.horizon),
                                       VecX::Zero(1),
                                       qp.Qxx_t);

    costs[qp.dims.horizon]->gradient(_guess.x.stage(qp.dims.horizon),
                                     VecX::Zero(1),
                                     qp.qx_t,
                                     tmpVec);

    if (constraints[qp.dims.horizon] != nullptr)
    {
        constraints[qp.dims.horizon]->hessprod_xx(
          _guess.x.stage(qp.dims.horizon),
          VecX::Zero(1),
          _guess.mu.stage(qp.dims.horizon),
          tmpMat);
        qp.Qxx_t += tmpMat;

        constraints[qp.dims.horizon]->gradient(_guess.x.stage(qp.dims.horizon),
                                               VecX::Zero(1),
                                               qp.Dt,
                                               tmpMat);

        VecX ct = constraints[qp.dims.horizon]->evaluate(
          _guess.x.stage(qp.dims.horizon),
          VecX::Zero(1));
        qp.cl_t = ct - constraints[qp.dims.horizon]->lower_bound();
        qp.cu_t = ct - constraints[qp.dims.horizon]->upper_bound();
    }
    else
    {
        qp.Dt   = MatX::Zero(0, qp.dims.stage_state[qp.dims.horizon]);
        qp.cl_t = VecX::Zero(0);
        qp.cu_t = VecX::Zero(0);
    }

    return qp;
}

SolverState ssqp::_check_termination()
{
    DEBUG_LOGLN("Guess " << _iter + 1);
    DEBUG_LOGLN("=============");
    DEBUG_LOGLN("x: " << std::endl << _guess.x);
    DEBUG_LOGLN("u: " << std::endl << _guess.u);
    DEBUG_LOGLN("lambda: " << std::endl << _guess.lambda);
    DEBUG_LOGLN("mu: " << std::endl << _guess.mu);
    DEBUG_LOGLN("=============");

    static const uint horizon = _model.dims().horizon;

    static const model::CostsContainer costs = _model.costs();

    static const model::DynamicsContainer dynamics = _model.dynamics();

    static const model::ConstraintsContainer constraints = _model.constraints();

    // Jacobian over x and y of lagrangian at current guess
    FlatVec lagrangian_dx(_guess.x, false);
    FlatVec lagrangian_du(_guess.u, false);

    // Cost gradients for loose success
    VecX dx, du;
    MatX dX, dU;

    // Add terminal terms for x
    costs[horizon]->gradient(_guess.x.stage(horizon), VecX::Zero(1), dx, du);

    lagrangian_dx.stage(horizon) += dx;
    // Add contribution from previous integration step (λ)
    lagrangian_dx.stage(horizon) -= _guess.lambda.stage(horizon - 1);

    _cost_dx.stage(horizon) = dx;

    // Add constraint terms for terminal x
    if (constraints[horizon] != nullptr)
    {
        constraints[horizon]->gradient(_guess.x.stage(horizon),
                                       VecX::Zero(1),
                                       dX,
                                       dU);

        lagrangian_dx.stage(horizon).noalias() -=
          dX.transpose() * _guess.mu.stage(horizon);
    }

    // Add k=0 terms for u
    costs[0]->gradient(_guess.x.stage(0), _guess.u.stage(0), dx, du);
    lagrangian_du.stage(0) += du;

    _cost_du.stage(0) = du;

    dynamics[0]->gradient(_guess.x.stage(0), _guess.u.stage(0), dX, dU);
    lagrangian_du.stage(0).noalias() += dU.transpose() * _guess.lambda.stage(0);

    if (constraints[0] != nullptr)
    {
        constraints[0]->gradient(_guess.x.stage(0), _guess.u.stage(0), dX, dU);
        lagrangian_du.stage(0).noalias() -= dU.transpose() * _guess.mu.stage(0);
    }

    // Ignore x0 terms since it is not being optimized
    for (uint k = 1; k < horizon; ++k)
    {
        // Compute cost jacobian
        costs[k]->gradient(_guess.x.stage(k), _guess.u.stage(k), dx, du);
        lagrangian_dx.stage(k) += dx;
        lagrangian_du.stage(k) += du;

        _cost_dx.stage(k) = dx;
        _cost_du.stage(k) = du;

        // Compute dynamics gap jacobian
        dynamics[k]->gradient(_guess.x.stage(k), _guess.u.stage(k), dX, dU);
        lagrangian_dx.stage(k).noalias() +=
          dX.transpose() * _guess.lambda.stage(k);
        lagrangian_dx.stage(k) -= _guess.lambda.stage(k - 1);
        lagrangian_du.stage(k).noalias() +=
          dU.transpose() * _guess.lambda.stage(k);

        // Compute constraint jacobian
        if (constraints[k] != nullptr)
        {
            constraints[k]->gradient(_guess.x.stage(k),
                                     _guess.u.stage(k),
                                     dX,
                                     dU);
            lagrangian_dx.stage(k).noalias() -=
              dX.transpose() * _guess.mu.stage(k);
            lagrangian_du.stage(k).noalias() -=
              dU.transpose() * _guess.mu.stage(k);
        }
    }

    LOGLN("Norm: " << lagrangian_dx.vec.lpNorm<Eigen::Infinity>());
    LOGLN("Norm: " << lagrangian_du.vec.lpNorm<Eigen::Infinity>());

    // Terminal constraint violation
    if (constraints[horizon] != nullptr &&
        !constraints[horizon]->is_satisfied(_guess.x.stage(horizon),
                                            VecX::Zero(1),
                                            _params.tolerance))
    {
        _stale_iters = 0;
        return ONGOING;
    }

    for (uint k = 0; k < horizon; ++k)
    {
        // Stage gap violation
        if (dynamics[k]
              ->state_right_minus(
                dynamics[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k)),
                _guess.x.stage(k + 1))
              .lpNorm<Eigen::Infinity>() > _params.tolerance)
        {
            _stale_iters = 0;
            return ONGOING;
        }
        // Stage constraint violation
        if (constraints[k] != nullptr &&
            !constraints[k]->is_satisfied(_guess.x.stage(k),
                                          _guess.u.stage(k),
                                          _params.tolerance))
        {
            _stale_iters = 0;
            return ONGOING;
        }
    }

    if (lagrangian_dx.vec.lpNorm<Eigen::Infinity>() <= _params.tolerance &&
        lagrangian_du.vec.lpNorm<Eigen::Infinity>() <= _params.tolerance)
    {
        LOGLN("SSQP: Convergence reached in " << _iter + 1 << " iterations");
        return SUCCESS;
    }

    if ((_cost_dx - _prev_cost_dx).vec.cwiseAbs().maxCoeff() <
          _params.loose_success_tolerance &&
        (_cost_du - _prev_cost_du).vec.cwiseAbs().maxCoeff() <
          _params.loose_success_tolerance)
    {
        if (++_stale_iters == _params.loose_success_stale_iters)
        {
            LOGLN("SSQP: Loose criteria satisfied after " << _iter + 1
                                                          << " iterations");
            return LOOSE_SUCCESS;
        }
    }
    else
    {
        _stale_iters = 0;
    }

    return ONGOING;
}

double ssqp::_lf(const FlatVec &x, const FlatVec &u) const
{
    static const uint           horizon = _model.horizon();
    double                      sum     = 0.0;
    const model::CostsContainer costs   = _model.costs();

    for (uint k = 0; k < horizon; ++k)
    {
        sum += costs[k]->evaluate(x.stage(k), u.stage(k))(0);
    }

    return sum + costs[horizon]->evaluate(x.stage(horizon), VecX::Zero(1))(0);
}

double ssqp::_gammaf(const FlatVec &x, const FlatVec &u) const
{
    static const uint              horizon  = _model.horizon();
    const model::DynamicsContainer dynamics = _model.dynamics();
    double                         sum      = 0.0;

    for (uint k = 0; k < horizon; ++k)
    {
        sum += (x.stage(k + 1) - dynamics[k]->evaluate(x.stage(k), u.stage(k)))
                 .lpNorm<Eigen::Infinity>();
    }

    return sum;
}

double ssqp::_cf(const FlatVec &x, const FlatVec &u) const
{
    static const uint                 horizon     = _model.horizon();
    const model::ConstraintsContainer constraints = _model.constraints();
    double                            sum         = 0.0;

    for (uint k = 0; k < horizon; ++k)
    {
        if (constraints[k] != nullptr)
        {
            sum += constraints[k]
                     ->evaluate(x.stage(k), u.stage(k))
                     .cwiseMin(0)
                     .lpNorm<Eigen::Infinity>();
        }
    }

    if (constraints[horizon] != nullptr)
    {
        sum += constraints[horizon]
                 ->evaluate(x.stage(horizon), VecX::Zero(1))
                 .cwiseMin(0)
                 .lpNorm<Eigen::Infinity>();
    }

    return sum;
}
}
}