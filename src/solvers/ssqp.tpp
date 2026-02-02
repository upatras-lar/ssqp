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

        // Get D and E
        if (constraints[k] != nullptr)
        {
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
}
}