#include "solvers/ssqp.hpp"

namespace ssqp
{
namespace solvers
{
template<bool withGradients, bool XMLModel>
void ssqp::_setup_qp()
{
    ZoneScopedN("SSQP: _setup_qp");

    VecX tmpVec;
    MatX tmpMat;

    const model::CostsContainer       costs       = _model.costs();
    const model::DynamicsContainer    dynamics    = _model.dynamics();
    const model::ConstraintsContainer constraints = _model.constraints();

    uint nq, nv, nu;
    if constexpr (XMLModel)
    {
        nq = _model.nq();
        nv = _model.nv();
        nu = _model.nu();
    }

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        logln_at(LogLevel::Debug, "Stage " << k << ":");

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

        if constexpr (XMLModel)
        {
            MatX Jx = _qp.stages[k].A.topLeftCorner(nv, nv);
            MatX Ju = _qp.stages[k].B.topLeftCorner(nu, nu);

            MatX Jr_inv = MatX::Zero(nv, nv);
            _model.dDifference(f.head(nq),
                               _guess.x.stage(k + 1).head(nq),
                               Jr_inv,
                               0);

            _qp.stages[k].A.topLeftCorner(nv, nv).noalias() = Jr_inv * Jx;
            _qp.stages[k].B.topLeftCorner(nu, nu).noalias() =
              _model.getMatrixActuatedPart(Jr_inv) * Ju;
        }

        logln_at(LogLevel::Debug, "A: " << std::endl << _qp.stages[k].A);
        logln_at(LogLevel::Debug, "B: " << std::endl << _qp.stages[k].B);
        logln_at(LogLevel::Debug, "gamma: " << _qp.stages[k].gamma.transpose());

        // Calculate Q_k
        costs[k]->hessian_xx(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Qxx);
        logln_at(LogLevel::Debug,
                 "Set Qxx via costs: size " << _qp.stages[k].Qxx.rows() << "x"
                                            << _qp.stages[k].Qxx.cols());
        dynamics[k]->hessprod_xx(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        logln_at(LogLevel::Debug,
                 "Got dynamics hessprod_xx, size " << tmpMat.rows() << "x"
                                                   << tmpMat.cols());
        logln_at(LogLevel::Debug, "Adding to Qxx...");
        _qp.stages[k].Qxx += tmpMat;

        logln_at(LogLevel::Debug, "Qxx: added costs & dynamics");

        // Calculate S_k
        costs[k]->hessian_xu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Qxu);
        logln_at(LogLevel::Debug,
                 "Set Qxu via costs: size " << _qp.stages[k].Qxu.rows() << "x"
                                            << _qp.stages[k].Qxu.cols());
        dynamics[k]->hessprod_xu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        logln_at(LogLevel::Debug,
                 "Got dynamics hessprod_xu, size " << tmpMat.rows() << "x"
                                                   << tmpMat.cols());
        logln_at(LogLevel::Debug, "Adding to Qxu...");
        _qp.stages[k].Qxu += tmpMat;

        logln_at(LogLevel::Debug, "Qxu: added costs & dynamics");

        // Calculate R_k
        costs[k]->hessian_uu(_guess.x.stage(k),
                             _guess.u.stage(k),
                             _qp.stages[k].Quu);
        logln_at(LogLevel::Debug,
                 "Set Quu via costs: size " << _qp.stages[k].Quu.rows() << "x"
                                            << _qp.stages[k].Quu.cols());
        dynamics[k]->hessprod_uu(_guess.x.stage(k),
                                 _guess.u.stage(k),
                                 _guess.lambda.stage(k),
                                 tmpMat);
        logln_at(LogLevel::Debug,
                 "Got dynamics hessprod_uu, size " << tmpMat.rows() << "x"
                                                   << tmpMat.cols());
        logln_at(LogLevel::Debug, "Adding to Quu...");
        _qp.stages[k].Quu += tmpMat;

        logln_at(LogLevel::Debug, "Quu: added costs & dynamics");

        if constexpr (withGradients)
        {
            // Get q and r
            costs[k]->gradient(_guess.x.stage(k),
                               _guess.u.stage(k),
                               _qp.stages[k].qx,
                               _qp.stages[k].qu);

            logln_at(LogLevel::Debug, "qx: " << _qp.stages[k].qx.transpose());
            logln_at(LogLevel::Debug, "qu: " << _qp.stages[k].qu.transpose());
        }

        if (constraints[k] != nullptr)
        {
            // Add constraint hessprod terms to cost matrices
            constraints[k]->hessprod_xx(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            logln_at(LogLevel::Debug,
                     "Got constraints hessprod_xx, size "
                       << tmpMat.rows() << "x" << tmpMat.cols());
            logln_at(LogLevel::Debug, "Adding to Qxx...");
            _qp.stages[k].Qxx += tmpMat;

            logln_at(LogLevel::Debug, "Qxx: added constraints");

            constraints[k]->hessprod_xu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            logln_at(LogLevel::Debug,
                     "Got constraints hessprod_xu, size "
                       << tmpMat.rows() << "x" << tmpMat.cols());
            logln_at(LogLevel::Debug, "Adding to Qxu...");
            _qp.stages[k].Qxu += tmpMat;

            logln_at(LogLevel::Debug, "Qxu: added constraints");

            constraints[k]->hessprod_uu(_guess.x.stage(k),
                                        _guess.u.stage(k),
                                        _guess.mu.stage(k),
                                        tmpMat);
            logln_at(LogLevel::Debug,
                     "Got constraints hessprod_uu, size "
                       << tmpMat.rows() << "x" << tmpMat.cols());
            logln_at(LogLevel::Debug, "Adding to Quu...");
            _qp.stages[k].Quu += tmpMat;

            logln_at(LogLevel::Debug, "Quu: added constraints");

            // Get D and E
            if constexpr (withGradients)
            {
                constraints[k]->gradient(_guess.x.stage(k),
                                         _guess.u.stage(k),
                                         _qp.stages[k].D,
                                         _qp.stages[k].E);

                logln_at(LogLevel::Debug,
                         "D: " << std::endl
                               << _qp.stages[k].D);
                logln_at(LogLevel::Debug,
                         "E: " << std::endl
                               << _qp.stages[k].E);
            }

            // Get cl, cu
            VecX c =
              constraints[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k));
            _qp.stages[k].cl =
              constraints[k]->right_minus(constraints[k]->lower_bound(), c);
            _qp.stages[k].cu =
              constraints[k]->right_minus(constraints[k]->upper_bound(), c);

            logln_at(LogLevel::Debug, "cl: " << _qp.stages[k].cl.transpose());
            logln_at(LogLevel::Debug, "cu: " << _qp.stages[k].cu.transpose());
        }
        else
        {
            _qp.stages[k].D.setZero(0, _qp.dims.stage_state[k]);
            _qp.stages[k].E.setZero(0, _qp.dims.stage_control[k]);
            _qp.stages[k].cl = _emptyVec;
            _qp.stages[k].cu = _emptyVec;
        }
    }

    logln_at(LogLevel::Debug, "Terminal stage");
    // Terminal values for Q, q, D and c
    costs[_qp.dims.horizon]->hessian_xx(_guess.x.stage(_qp.dims.horizon),
                                        _emptyVec,
                                        _qp.Qxx_t);
    logln_at(LogLevel::Debug,
             "Set Qxx_t via costs: size " << _qp.Qxx_t.rows() << "x"
                                          << _qp.Qxx_t.cols());

    if constexpr (withGradients)
    {
        costs[_qp.dims.horizon]->gradient(_guess.x.stage(_qp.dims.horizon),
                                          _emptyVec,
                                          _qp.qx_t,
                                          tmpVec);
    }

    if (constraints[_qp.dims.horizon] != nullptr)
    {
        constraints[_qp.dims.horizon]->hessprod_xx(
          _guess.x.stage(_qp.dims.horizon),
          _emptyVec,
          _guess.mu.stage(_qp.dims.horizon),
          tmpMat);
        logln_at(LogLevel::Debug,
                 "Got constraints hessprod_xx, size " << tmpMat.rows() << "x"
                                                      << tmpMat.cols());
        logln_at(LogLevel::Debug, "Adding to Qxx_t...");
        _qp.Qxx_t += tmpMat;

        logln_at(LogLevel::Debug, "Qxx: added constraints");

        if constexpr (withGradients)
        {
            constraints[_qp.dims.horizon]->gradient(
              _guess.x.stage(_qp.dims.horizon),
              _emptyVec,
              _qp.Dt,
              tmpMat);
        }

        logln_at(LogLevel::Debug, "D: " << std::endl << _qp.Dt);

        VecX ct = constraints[_qp.dims.horizon]->evaluate(
          _guess.x.stage(_qp.dims.horizon),
          _emptyVec);
        _qp.cl_t = constraints[_qp.dims.horizon]->right_minus(
          constraints[_qp.dims.horizon]->lower_bound(),
          ct);
        _qp.cu_t = constraints[_qp.dims.horizon]->right_minus(
          constraints[_qp.dims.horizon]->upper_bound(),
          ct);

        logln_at(LogLevel::Debug, "cl: " << _qp.cl_t.transpose());
        logln_at(LogLevel::Debug, "cu: " << _qp.cu_t.transpose());
    }
    else
    {
        _qp.Dt.setZero(0, _qp.dims.stage_state[_qp.dims.horizon]);
        _qp.cl_t = _emptyVec;
        _qp.cu_t = _emptyVec;
    }
}

template<bool fullStagewiseConvergence>
SolverState ssqp::_check_termination()
{
    ZoneScopedN("SSQP: _check_termination");

    logln_at(LogLevel::Debug, "Guess " << _iter + 1);
    logln_at(LogLevel::Debug, "=============");
    logln_at(LogLevel::Debug, "x: " << std::endl << _guess.x);
    logln_at(LogLevel::Debug, "u: " << std::endl << _guess.u);
    logln_at(LogLevel::Debug, "lambda: " << std::endl << _guess.lambda);
    logln_at(LogLevel::Debug, "mu: " << std::endl << _guess.mu);
    logln_at(LogLevel::Debug, "=============");

    const uint                 horizon     = _model.dims().horizon;
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
                             _emptyVec,
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
                                       _emptyVec,
                                       _qp.Dt,
                                       tmp_mat);

        const uint nx = _qp.dims.stage_state[horizon];
        auto       x1 = _tmp.x1.head(nx);

        x1.noalias() = _qp.Dt.transpose() * _guess.mu.stage(horizon);
        lagrangian_dx.stage(horizon) -= x1;
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

    const uint nu = _qp.dims.stage_control[0];
    auto       u1 = _tmp.u1.head(nu);

    u1.noalias() = _qp.stages[0].B.transpose() * _guess.lambda.stage(0);
    lagrangian_du.stage(0).noalias() += u1;

    if (constraints[0] != nullptr)
    {
        constraints[0]->gradient(_guess.x.stage(0),
                                 _guess.u.stage(0),
                                 _qp.stages[0].D,
                                 _qp.stages[0].E);

        u1.noalias() = _qp.stages[0].E.transpose() * _guess.mu.stage(0);
        lagrangian_du.stage(0).noalias() -= u1;
    }

    // Ignore x0 terms since it is not being optimized
    for (int k = 1; k < horizon; ++k)
    {
        auto &stage = _qp.stages[k];

        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];

        auto x1 = _tmp.x1.head(nx);
        auto u1 = _tmp.u1.head(nu);

        // =========================
        // Cost
        // =========================
        costs[k]->gradient(_guess.x.stage(k),
                           _guess.u.stage(k),
                           stage.qx,
                           stage.qu);

        lagrangian_dx.stage(k) += stage.qx;
        lagrangian_du.stage(k) += stage.qu;

        _cost_dx.stage(k) = stage.qx;
        _cost_du.stage(k) = stage.qu;

        // =========================
        // Dynamics
        // =========================
        dynamics[k]->gradient(_guess.x.stage(k),
                              _guess.u.stage(k),
                              stage.A,
                              stage.B);

        // dx += Aᵀ λ_k
        x1.noalias()            = stage.A.transpose() * _guess.lambda.stage(k);
        lagrangian_dx.stage(k) += x1;

        // dx -= λ_{k-1}
        lagrangian_dx.stage(k) -= _guess.lambda.stage(k - 1);

        // du += Bᵀ λ_k
        u1.noalias()            = stage.B.transpose() * _guess.lambda.stage(k);
        lagrangian_du.stage(k) += u1;

        // =========================
        // Constraints
        // =========================
        if (constraints[k] != nullptr)
        {
            constraints[k]->gradient(_guess.x.stage(k),
                                     _guess.u.stage(k),
                                     stage.D,
                                     stage.E);

            // dx -= Dᵀ μ
            x1.noalias()            = stage.D.transpose() * _guess.mu.stage(k);
            lagrangian_dx.stage(k) -= x1;

            // du -= Eᵀ μ
            u1.noalias()            = stage.E.transpose() * _guess.mu.stage(k);
            lagrangian_du.stage(k) -= u1;
        }
    }

    _convergence_data.iteration_data.back().norm_dx =
      lagrangian_dx.vec.lpNorm<Eigen::Infinity>();
    _convergence_data.iteration_data.back().norm_du =
      lagrangian_du.vec.lpNorm<Eigen::Infinity>();

    _convergence_data.iteration_data.back().first_violation_stage = -1;

    if constexpr (fullStagewiseConvergence)
    {
        _convergence_data.iteration_data.back().costs.resize(horizon + 1);
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
        if (gap_norm > _params.constraint_tolerance)
        {
            if constexpr (fullStagewiseConvergence)
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
            if (csnorm > _params.constraint_tolerance)
            {
                if constexpr (fullStagewiseConvergence)
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

        if constexpr (fullStagewiseConvergence)
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
        csnorm =
          constraints[horizon]->violation(_guess.x.stage(horizon), _emptyVec);

        if (csnorm > _params.constraint_tolerance)
        {
            if constexpr (fullStagewiseConvergence)
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

    if constexpr (fullStagewiseConvergence)
    {
        _convergence_data.iteration_data.back().constraint_violations[horizon] =
          csnorm;

        const model::CostsContainer costs = _model.costs();
        for (uint k = 0; k < horizon; ++k)
        {
            _convergence_data.iteration_data.back().costs[k] =
              costs[k]->evaluate(_guess.x.stage(k), _guess.u.stage(k))[0];
        }
        _convergence_data.iteration_data.back().costs[horizon] =
          costs[horizon]->evaluate(_guess.x.stage(horizon), _emptyVec)[0];

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