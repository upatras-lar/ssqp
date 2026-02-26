#include "solvers/ssqp.hpp"

namespace ssqp
{
namespace solvers
{
template<bool withGradients>
void ssqp::_setup_qp()
{
    ZoneScopedN("SSQP: _setup_qp");

    VecX tmpVec;
    MatX tmpMat;

    const model::CostsContainer       costs       = _model.costs();
    const model::DynamicsContainer    dynamics    = _model.dynamics();
    const model::ConstraintsContainer constraints = _model.constraints();

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        DEBUG_LOGLN("Stage " << k << ":");

        if constexpr (withGradients)
        {
            // Get A_k and B_k matrices
            dynamics[k]->gradient(_guess.x.stage(k),
                                  _guess.u.stage(k),
                                  _qp.stages[k].A,
                                  _qp.stages[k].B);
        }

        const VecX f =
          dynamics[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k));

        // Get gamma_k
        _qp.stages[k].gamma =
          dynamics[k]->state_right_minus(f, _guess.x.stage(k + 1));

        MatX Jx = _qp.stages[k].A.topLeftCorner(_model.nv(k), _model.nv(k));
        MatX Ju = _qp.stages[k].B.topLeftCorner(_model.nv(k), _model.nv(k));

        MatX Jr_inv = MatX::Zero(_model.nv(k), _model.nv(k));
        _model.dDifference(f.head(_model.nq(k)),
                           _guess.x.stage(k + 1).head(_model.nq(k)),
                           Jr_inv,
                           0);

        _qp.stages[k].A.topLeftCorner(_model.nv(k), _model.nv(k)).noalias() =
          Jr_inv * Jx;
        _qp.stages[k].B.topLeftCorner(_model.nv(k), _model.nv(k)).noalias() =
          Jr_inv * Ju;

        DEBUG_LOGLN("A: " << std::endl << _qp.stages[k].A);
        DEBUG_LOGLN("B: " << std::endl << _qp.stages[k].B);
        DEBUG_LOGLN("Jr_inv: " << std::endl << Jr_inv);
        DEBUG_LOGLN("gamma: " << _qp.stages[k].gamma.transpose());

        // Calculate Q_k
        costs[k]->hessian_xx(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Qxx);

        dynamics[k]->hessprod_xx(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);

        _qp.stages[k].Qxx += tmpMat;

        // Calculate S_k
        costs[k]->hessian_xu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Qxu);
        dynamics[k]->hessprod_xu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);

        _qp.stages[k].Qxu += tmpMat;

        // Calculate R_k
        costs[k]->hessian_uu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Quu);

        dynamics[k]->hessprod_uu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        _qp.stages[k].Quu += tmpMat;

        if constexpr (withGradients)
        {
            // Get q and r
            costs[k]->gradient(_guess.x.stage(k),
                               _guess.u.stage(k),
                               _qp.stages[k].qx,
                               _qp.stages[k].qu);
        }

        if (constraints[k] != nullptr)
        {
            // Add constraint hessprod terms to cost matrices
            constraints[k]->hessprod_xx(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            _qp.stages[k].Qxx += tmpMat;

            constraints[k]->hessprod_xu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            _qp.stages[k].Qxu += tmpMat;

            constraints[k]->hessprod_uu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            _qp.stages[k].Quu += tmpMat;

            // Get D and E
            if constexpr (withGradients)
            {
                constraints[k]->gradient(_guess.x.stage(k),
                                         _guess.u.stage(k),
                                         _qp.stages[k].D,
                                         _qp.stages[k].E);
            }

            // Get cl, cu
            VecX c =
              constraints[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k));
            _qp.stages[k].cl =
              constraints[k]->right_minus(constraints[k]->lower_bound(), c);
            _qp.stages[k].cu =
              constraints[k]->right_minus(constraints[k]->upper_bound(), c);
        }
        else
        {
            _qp.stages[k].D  = MatX::Zero(0, _qp.dims.stage_state[k]);
            _qp.stages[k].E  = MatX::Zero(0, _qp.dims.stage_control[k]);
            _qp.stages[k].cl = VecX::Zero(0);
            _qp.stages[k].cu = VecX::Zero(0);
        }

        DEBUG_LOGLN("Qxx: " << std::endl << _qp.stages[k].Qxx);
        DEBUG_LOGLN("Qxu: " << std::endl << _qp.stages[k].Qxu);
        DEBUG_LOGLN("Quu: " << std::endl << _qp.stages[k].Quu);
        DEBUG_LOGLN("qx: " << _qp.stages[k].qx.transpose());
        DEBUG_LOGLN("qu: " << _qp.stages[k].qu.transpose());
        DEBUG_LOGLN("D: " << std::endl << _qp.stages[k].D);
        DEBUG_LOGLN("E: " << std::endl << _qp.stages[k].E);
        DEBUG_LOGLN("cl: " << _qp.stages[k].cl.transpose());
        DEBUG_LOGLN("cu: " << _qp.stages[k].cu.transpose());
    }

    // Terminal values for Q, q, D and c
    costs[_qp.dims.horizon]->hessian_xx(_guess.x.stage(_qp.dims.horizon),
                                        VecX::Zero(1),
                                        _qp.Qxx_t);

    if constexpr (withGradients)
    {
        costs[_qp.dims.horizon]->gradient(_guess.x.stage(_qp.dims.horizon),
                                          VecX::Zero(1),
                                          _qp.qx_t,
                                          tmpVec);
    }

    if (constraints[_qp.dims.horizon] != nullptr)
    {
        constraints[_qp.dims.horizon]->hessprod_xx(
          _guess.x.stage(_qp.dims.horizon),
          VecX::Zero(1),
          _guess.mu.stage(_qp.dims.horizon),
          tmpMat);
        _qp.Qxx_t += tmpMat;

        if constexpr (withGradients)
        {
            constraints[_qp.dims.horizon]->gradient(
              _guess.x.stage(_qp.dims.horizon),
              VecX::Zero(1),
              _qp.Dt,
              tmpMat);
        }
        VecX ct = constraints[_qp.dims.horizon]->evaluate(
          _guess.x.stage(_qp.dims.horizon),
          VecX::Zero(1));
        _qp.cl_t = constraints[_qp.dims.horizon]->right_minus(
          constraints[_qp.dims.horizon]->lower_bound(),
          ct);
        _qp.cu_t = constraints[_qp.dims.horizon]->right_minus(
          constraints[_qp.dims.horizon]->upper_bound(),
          ct);
    }
    else
    {
        _qp.Dt   = MatX::Zero(0, _qp.dims.stage_state[_qp.dims.horizon]);
        _qp.cl_t = VecX::Zero(0);
        _qp.cu_t = VecX::Zero(0);
    }
}

template<bool fullViolations>
SolverState ssqp::_check_termination()
{
    ZoneScopedN("SSQP: _check_termination");

    DEBUG_LOGLN("Guess " << _iter + 1);
    DEBUG_LOGLN("=============");
    DEBUG_LOGLN("x: " << std::endl << _guess.x);
    DEBUG_LOGLN("u: " << std::endl << _guess.u);
    DEBUG_LOGLN("lambda: " << std::endl << _guess.lambda);
    DEBUG_LOGLN("mu: " << std::endl << _guess.mu);
    DEBUG_LOGLN("=============");

    static const uint                 horizon     = _model.dims().horizon;
    const model::CostsContainer       costs       = _model.costs();
    const model::DynamicsContainer    dynamics    = _model.dynamics();
    const model::ConstraintsContainer constraints = _model.constraints();

    // Jacobian over x and y of lagrangian at current guess
    FlatVec lagrangian_dx(_cost_dx, false);
    FlatVec lagrangian_du(_cost_du, false);

    // Cost gradients for loose success
    VecX tmp_vec;
    MatX tmp_mat;

    // Add terminal terms for x
    costs[horizon]->gradient(_guess.x.stage(horizon),
                             VecX::Zero(1),
                             _qp.qx_t,
                             tmp_vec);

    lagrangian_dx.stage(horizon) += _qp.qx_t;
    // Add contribution from previous integration step (λ)
    lagrangian_dx.stage(horizon) -= _guess.lambda.stage(horizon - 1);

    _cost_dx.stage(horizon) = _qp.qx_t;

    // Add constraint terms for terminal x
    if (constraints[horizon] != nullptr)
    {
        constraints[horizon]->gradient(_guess.x.stage(horizon),
                                       VecX::Zero(1),
                                       _qp.Dt,
                                       tmp_mat);

        lagrangian_dx.stage(horizon).noalias() -=
          _qp.Dt.transpose() * _guess.mu.stage(horizon);
    }

    // Add k=0 terms for u
    costs[0]->gradient(_guess.x.stage(0),
                       _guess.u.stage(0),
                       _qp.stages[0].qx,
                       _qp.stages[0].qu);
    lagrangian_du.stage(0) += _qp.stages[0].qu;

    _cost_du.stage(0) = _qp.stages[0].qu;

    dynamics[0]->gradient(_guess.x.stage(0),
                          _guess.u.stage(0),
                          _qp.stages[0].A,
                          _qp.stages[0].B);
    lagrangian_du.stage(0).noalias() +=
      _qp.stages[0].B.transpose() * _guess.lambda.stage(0);

    if (constraints[0] != nullptr)
    {
        constraints[0]->gradient(_guess.x.stage(0),
                                 _guess.u.stage(0),
                                 _qp.stages[0].D,
                                 _qp.stages[0].E);
        lagrangian_du.stage(0).noalias() -=
          _qp.stages[0].E.transpose() * _guess.mu.stage(0);
    }

    // Ignore x0 terms since it is not being optimized
    for (uint k = 1; k < horizon; ++k)
    {
        // Compute cost jacobian
        costs[k]->gradient(_guess.x.stage(k),
                           _guess.u.stage(k),
                           _qp.stages[k].qx,
                           _qp.stages[k].qu);
        lagrangian_dx.stage(k) += _qp.stages[k].qx;
        lagrangian_du.stage(k) += _qp.stages[k].qu;

        _cost_dx.stage(k) = _qp.stages[k].qx;
        _cost_du.stage(k) = _qp.stages[k].qu;

        // Compute dynamics gap jacobian
        dynamics[k]->gradient(_guess.x.stage(k),
                              _guess.u.stage(k),
                              _qp.stages[k].A,
                              _qp.stages[k].B);
        lagrangian_dx.stage(k).noalias() +=
          _qp.stages[k].A.transpose() * _guess.lambda.stage(k);
        lagrangian_dx.stage(k) -= _guess.lambda.stage(k - 1);
        lagrangian_du.stage(k).noalias() +=
          _qp.stages[k].B.transpose() * _guess.lambda.stage(k);

        // Compute constraint jacobian
        if (constraints[k] != nullptr)
        {
            constraints[k]->gradient(_guess.x.stage(k),
                                     _guess.u.stage(k),
                                     _qp.stages[k].D,
                                     _qp.stages[k].E);
            lagrangian_dx.stage(k).noalias() -=
              _qp.stages[k].D.transpose() * _guess.mu.stage(k);
            lagrangian_du.stage(k).noalias() -=
              _qp.stages[k].E.transpose() * _guess.mu.stage(k);
        }
    }

    _convergence_data.iteration_data.back().norm_dx =
      lagrangian_dx.vec.lpNorm<Eigen::Infinity>();
    _convergence_data.iteration_data.back().norm_du =
      lagrangian_du.vec.lpNorm<Eigen::Infinity>();

    _convergence_data.iteration_data.back().first_violation_stage = -1;

    if constexpr (fullViolations)
    {
        _convergence_data.iteration_data.back().gap_violations.resize(horizon);
        _convergence_data.iteration_data.back().constraint_violations.resize(
          horizon + 1);
    }

    for (uint k = 0; k < horizon; ++k)
    {
        // Stage gap violation
        double gap_norm =
          dynamics[k]
            ->state_right_minus(
              dynamics[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k)),
              _guess.x.stage(k + 1))
            .lpNorm<Eigen::Infinity>();
        if (gap_norm > _params.tolerance)
        {
            if constexpr (fullViolations)
            {
                if (_convergence_data.iteration_data.back()
                      .first_violation_stage == -1)
                {
                    _stale_iters = 0;
                    _convergence_data.iteration_data.back()
                      .first_violation_stage = k;
                    _convergence_data.iteration_data.back().violation =
                      gap_norm;
                }
            }
            else
            {

                _stale_iters = 0;
                _convergence_data.iteration_data.back().first_violation_stage =
                  k;
                _convergence_data.iteration_data.back().violation = gap_norm;
                return ONGOING;
            }
        }

        double csnorm = 0.0;

        // Stage constraint violation
        if (constraints[k] != nullptr)
        {
            csnorm =
              constraints[k]->violation(_guess.x.stage(k), _guess.u.stage(k));
            if (csnorm > _params.tolerance)
            {
                if constexpr (fullViolations)
                {
                    if (_convergence_data.iteration_data.back()
                          .first_violation_stage == -1)
                    {
                        _stale_iters = 0;
                        _convergence_data.iteration_data.back()
                          .first_violation_stage = k;
                        _convergence_data.iteration_data.back().violation =
                          csnorm;
                    }
                }
                else
                {
                    _stale_iters = 0;
                    _convergence_data.iteration_data.back()
                      .first_violation_stage                          = k;
                    _convergence_data.iteration_data.back().violation = csnorm;

                    return ONGOING;
                }
            }
        }

        if constexpr (fullViolations)
        {
            _convergence_data.iteration_data.back().gap_violations[k] =
              gap_norm;
            _convergence_data.iteration_data.back().constraint_violations[k] =
              csnorm;
        }
    }

    double csnorm = 0.0;

    // Terminal constraint violation
    if (constraints[horizon] != nullptr)
    {
        csnorm = constraints[horizon]->violation(_guess.x.stage(horizon),
                                                 VecX::Zero(1));

        if (csnorm > _params.tolerance)
        {
            if constexpr (fullViolations)
            {
                if (_convergence_data.iteration_data.back()
                      .first_violation_stage == -1)
                {
                    _stale_iters = 0;
                    _convergence_data.iteration_data.back()
                      .first_violation_stage                          = horizon;
                    _convergence_data.iteration_data.back().violation = csnorm;
                }
            }
            else
            {

                _stale_iters = 0;
                _convergence_data.iteration_data.back().first_violation_stage =
                  horizon;
                _convergence_data.iteration_data.back().violation = csnorm;

                return ONGOING;
            }
        }
    }

    if constexpr (fullViolations)
    {
        _convergence_data.iteration_data.back().constraint_violations[horizon] =
          csnorm;

        if (_convergence_data.iteration_data.back().first_violation_stage != -1)
        {
            return ONGOING;
        }
    }

    if (lagrangian_dx.vec.lpNorm<Eigen::Infinity>() <= _params.tolerance &&
        lagrangian_du.vec.lpNorm<Eigen::Infinity>() <= _params.tolerance)
    {
        _convergence_data.iterations = _iter + 1;
        return SUCCESS;
    }
    else if (_params.stop_after_constraint_satisfaction)
    {
        _convergence_data.iterations         = _iter + 1;
        _convergence_data.constraint_success = true;
        return CONSTRAINT_SUCCESS;
    }

    if (_params.loose_success &&
        (_cost_dx - _prev_cost_dx).vec.cwiseAbs().maxCoeff() <
          _params.loose_success_tolerance &&
        (_cost_du - _prev_cost_du).vec.cwiseAbs().maxCoeff() <
          _params.loose_success_tolerance)
    {
        if (++_stale_iters == _params.loose_success_stale_iters)
        {
            _convergence_data.iterations    = _iter + 1;
            _convergence_data.loose_success = true;
            return LOOSE_SUCCESS;
        }
    }
    else
    {
        _stale_iters = 0;
    }

    return ONGOING;
}

}
}