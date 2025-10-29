#include "solvers/lqr.hpp"

namespace ssqp
{
namespace solvers
{

lqr::lqr(const QPData &qp, const admm_qp::Solution &current_sol,
         const FlatVec &rho, const double sigma)
  : _qp(qp)
  , _Qxx(qp.dims.horizon)
  , _Qxu(qp.dims.horizon)
  , _Quu(qp.dims.horizon)
  , _qx(qp.dims.horizon)
  , _qu(qp.dims.horizon)
  , _rho(rho, true)
  , _sigma(sigma)
  , _G(qp.dims.horizon)
  , _H(qp.dims.horizon)
  , _H_llt(qp.dims.horizon)
  , _output({ FlatVec(current_sol.dx, false),
              FlatVec(current_sol.du, false),
              std::vector<MatX>(qp.dims.horizon + 1),
              std::vector<MatX>(qp.dims.horizon),
              std::vector<VecX>(qp.dims.horizon + 1),
              std::vector<VecX>(qp.dims.horizon) })
{
}

lqr::Output lqr::output_structure()
{
    return _output;
}

lqr::Output lqr::unconstrained()
{
    // Initialize unconstrained
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _Qxx[k] = _qp.stages[k].Qxx;
        _Qxu[k] = _qp.stages[k].Qxu;
        _Quu[k] = _qp.stages[k].Quu;
        _qx[k]  = _qp.stages[k].qx;
        _qu[k]  = _qp.stages[k].qu;
    }
    _Qt = _qp.Qxx_t;
    _qt = _qp.qx_t;

    _solve();

    return _output;
}

lqr::Output lqr::constrained(const admm_qp::Solution &current_sol)
{
    // Calculate constraint terms, assuming rho did not change
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _qx[k].noalias() =
          _qp.stages[k].qx +
          _qp.stages[k].D.transpose() * current_sol.y.stage(k) -
          _qp.stages[k].D.transpose() * _rho.stage(k).asDiagonal() *
            current_sol.z.stage(k) -
          _sigma * current_sol.dx.stage(k);

        _qu[k].noalias() =
          _qp.stages[k].qu +
          _qp.stages[k].E.transpose() * current_sol.y.stage(k) -
          _qp.stages[k].E.transpose() * _rho.stage(k).asDiagonal() *
            current_sol.z.stage(k) -
          _sigma * current_sol.du.stage(k);
    }
    _qt.noalias() =
      _qp.qx_t + _qp.Dt.transpose() * current_sol.y.stage(_qp.dims.horizon) -
      _qp.Dt.transpose() * _rho.stage(_qp.dims.horizon).asDiagonal() *
        current_sol.z.stage(_qp.dims.horizon) +
      _sigma * current_sol.dx.stage(_qp.dims.horizon);

    _solve_cache_rho();

    return _output;
}

lqr::Output lqr::constrained(const admm_qp::Solution &current_sol,
                             const FlatVec           &rho)
{
    _rho = rho;

    // Calculate constraint terms
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _Qxx[k].noalias() = _qp.stages[k].Qxx +
                            _qp.stages[k].D.transpose() *
                              rho.stage(k).asDiagonal() * _qp.stages[k].D +
                            _sigma * MatX::Identity(_qp.dims.stage_state[k],
                                                    _qp.dims.stage_state[k]);
        _Qxu[k].noalias() = _qp.stages[k].Qxu + _qp.stages[k].D.transpose() *
                                                  rho.stage(k).asDiagonal() *
                                                  _qp.stages[k].E;
        _Quu[k].noalias() = _qp.stages[k].Quu +
                            _qp.stages[k].E.transpose() *
                              rho.stage(k).asDiagonal() * _qp.stages[k].E +
                            _sigma * MatX::Identity(_qp.dims.stage_control[k],
                                                    _qp.dims.stage_control[k]);

        _qx[k].noalias() =
          _qp.stages[k].qx +
          _qp.stages[k].D.transpose() * current_sol.y.stage(k) -
          _qp.stages[k].D.transpose() * rho.stage(k).asDiagonal() *
            current_sol.z.stage(k) -
          _sigma * current_sol.dx.stage(k);

        _qu[k].noalias() =
          _qp.stages[k].qu +
          _qp.stages[k].E.transpose() * current_sol.y.stage(k) -
          _qp.stages[k].E.transpose() * rho.stage(k).asDiagonal() *
            current_sol.z.stage(k) -
          _sigma * current_sol.du.stage(k);
    }

    _Qt.noalias() =
      _qp.Qxx_t +
      _qp.Dt.transpose() * rho.stage(_qp.dims.horizon).asDiagonal() * _qp.Dt +
      _sigma * MatX::Identity(_qp.dims.stage_state[_qp.dims.horizon],
                              _qp.dims.stage_state[_qp.dims.horizon]);
    _qt.noalias() =
      _qp.qx_t + _qp.Dt.transpose() * current_sol.y.stage(_qp.dims.horizon) -
      _qp.Dt.transpose() * rho.stage(_qp.dims.horizon).asDiagonal() *
        current_sol.z.stage(_qp.dims.horizon) +
      _sigma * current_sol.dx.stage(_qp.dims.horizon);

    _solve();

    return _output;
}

void lqr::_solve_cache_rho()
{
    _output.p[_qp.dims.horizon] = _qt;

    // Backward pass
    for (int k = _qp.dims.horizon - 1; k >= 0; --k)
    {
        VecX h = _qu[k] +
                 _qp.stages[k].B.transpose() *
                   (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);

        // TODO: is LLT appropriate?
        if (_qp.dims.stage_control[k] > 1)
        {
            _output.k[k].noalias() = -_H_llt[k].solve(h);
        }
        else
        {
            double coeff = 1. / _H[k](0, 0);
            _output.k[k] = -coeff * h;
        }

        _output.p[k] =
          _qx[k] + _output.K[k].transpose() * _qu[k] +
          (_qp.stages[k].A + _qp.stages[k].B * _output.K[k]).transpose() *
            (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);
    }

    // Forward pass
    _output.x.stage(0) = VecX::Zero(_qp.dims.stage_state[0]);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _output.x.stage(k + 1) =
          (_qp.stages[k].A + _qp.stages[k].B * _output.K[k]) *
            _output.x.stage(k) +
          _qp.stages[k].B * _output.k[k] + _qp.stages[k].gamma;
        _output.u.stage(k).noalias() =
          _output.K[k] * _output.x.stage(k) + _output.k[k];
    }
}

void lqr::_solve()
{
    // DEBUG_LOGLN("Qt:" << std::endl << _Qt);
    _output.P[_qp.dims.horizon] = _Qt;
    _output.p[_qp.dims.horizon] = _qt;

    // Backward pass
    for (int k = _qp.dims.horizon - 1; k >= 0; --k)
    {
        VecX h = _qu[k] +
                 _qp.stages[k].B.transpose() *
                   (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);

        _G[k].noalias() = _Qxu[k].transpose() + _qp.stages[k].B.transpose() *
                                                  _output.P[k + 1] *
                                                  _qp.stages[k].A;

        // DEBUG_LOGLN("Quu[" << k << "]:" << std::endl << _Quu[k]);
        // DEBUG_LOGLN("P[" << k + 1 << "]:" << std::endl << _output.P[k + 1]);

        _H[k].noalias() = _Quu[k] + _qp.stages[k].B.transpose() *
                                      _output.P[k + 1] * _qp.stages[k].B;

        // TODO: is LLT appropriate?
        if (_qp.dims.stage_control[k] > 1)
        {
            // DEBUG_LOGLN("H[" << k << "]:" << std::endl << _H[k]);
            _H_llt[k]    = _H[k].llt();
            _output.K[k] = -_H_llt[k].solve(_G[k]);
            _output.k[k] = -_H_llt[k].solve(h);
        }
        else
        {
            double coeff = 1. / _H[k](0, 0);
            _output.K[k] = -coeff * _G[k];
            _output.k[k] = -coeff * h;
        }

        _output.P[k] =
          _Qxx[k] +
          _qp.stages[k].A.transpose() * _output.P[k + 1] * _qp.stages[k].A -
          _output.K[k].transpose() * _H[k] * _output.K[k];

        _output.p[k] =
          _qx[k] + _output.K[k].transpose() * _qu[k] +
          (_qp.stages[k].A + _qp.stages[k].B * _output.K[k]).transpose() *
            (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);
    }

    // Forward pass
    _output.x.stage(0) = VecX::Zero(_qp.dims.stage_state[0]);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        // DEBUG_LOGLN("Setting x[" << k << "]:");
        // DEBUG_LOGLN("A:" << std::endl << _qp.stages[k].A);
        // DEBUG_LOGLN("B:" << std::endl << _qp.stages[k].B);
        // DEBUG_LOGLN("K:" << std::endl << _output.K[k]);
        // DEBUG_LOGLN("Previous x:" << std::endl
        //                           << _output.x.stage(k).transpose());
        // DEBUG_LOGLN("k:" << std::endl << _output.k[k].transpose());
        // DEBUG_LOGLN("gamma:" << std::endl <<
        // _qp.stages[k].gamma.transpose());
        _output.x.stage(k + 1) =
          (_qp.stages[k].A + _qp.stages[k].B * _output.K[k]) *
            _output.x.stage(k) +
          _qp.stages[k].B * _output.k[k] + _qp.stages[k].gamma;
        _output.u.stage(k).noalias() =
          _output.K[k] * _output.x.stage(k) + _output.k[k];
    }
}
}
}