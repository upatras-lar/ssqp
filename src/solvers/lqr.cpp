#include "solvers/lqr.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace solvers
{

lqr::lqr(const QPData &qp, const admm_qp::Solution &current_sol,
         const FlatVec &rho, const lqr::Params &params)
  : _qp(qp)
  , _Qxx(qp.dims.horizon)
  , _Qxu(qp.dims.horizon)
  , _Quu(qp.dims.horizon)
  , _qx(qp.dims.horizon)
  , _qu(qp.dims.horizon)
  , _rho(rho, true)
  , _sigma(params.sigma)
  , _G(qp.dims.horizon)
  , _H(qp.dims.horizon)
  , _H_llt(qp.dims.horizon)
  , _solve_success_sma_k(params.solve_window)
  , _solve_success_sma_ratio(params.solve_success_ratio)
  , _solve_success_sum(0)
  , _reg_init(params.reg_init)
  , _reg_factor(params.reg_factor)
  , _reg_max(params.reg_max)
  , _reg(params.reg_init)
  , _output({ FlatVec(current_sol.dx, false),
              FlatVec(current_sol.du, false),
              std::vector<MatX>(qp.dims.horizon + 1),
              std::vector<MatX>(qp.dims.horizon),
              std::vector<VecX>(qp.dims.horizon + 1),
              std::vector<VecX>(qp.dims.horizon) })
{
}

lqr::Output &lqr::output_structure()
{
    return _output;
}

lqr::Output &lqr::unconstrained()
{
    ZoneScopedN("LQR: unconstrained");
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

lqr::Output &lqr::constrained(const admm_qp::Solution &current_sol)
{
    ZoneScopedN("LQR: constrained (cached)");
    // Calculate constraint terms, assuming rho did not change
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        if (_qp.stages[k].D.rows() == 0)
        {
            continue;
        }

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

    if (_qp.Dt.rows() != 0)
    {
        _qt.noalias() =
          _qp.qx_t +
          _qp.Dt.transpose() * current_sol.y.stage(_qp.dims.horizon) -
          _qp.Dt.transpose() * _rho.stage(_qp.dims.horizon).asDiagonal() *
            current_sol.z.stage(_qp.dims.horizon) +
          _sigma * current_sol.dx.stage(_qp.dims.horizon);
    }

    _solve_cache_rho();

    return _output;
}

lqr::Output &lqr::constrained(const admm_qp::Solution &current_sol,
                              const FlatVec           &rho)
{
    ZoneScopedN("LQR: constrained (uncached)");
    _rho = rho;

    // Calculate constraint terms
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        if (_qp.stages[k].D.rows() == 0)
        {
            continue;
        }

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

    if (_qp.Dt.rows() != 0)
    {
        _Qt.noalias() =
          _qp.Qxx_t +
          _qp.Dt.transpose() * rho.stage(_qp.dims.horizon).asDiagonal() *
            _qp.Dt +
          _sigma * MatX::Identity(_qp.dims.stage_state[_qp.dims.horizon],
                                  _qp.dims.stage_state[_qp.dims.horizon]);
        _qt.noalias() =
          _qp.qx_t +
          _qp.Dt.transpose() * current_sol.y.stage(_qp.dims.horizon) -
          _qp.Dt.transpose() * rho.stage(_qp.dims.horizon).asDiagonal() *
            current_sol.z.stage(_qp.dims.horizon) -
          _sigma * current_sol.dx.stage(_qp.dims.horizon);
    }

    _solve();

    return _output;
}

void lqr::_solve_cache_rho()
{
    ZoneScopedN("LQR: _solve_cache_rho");
    _output.p[_qp.dims.horizon] = _qt;

    // Backward pass
    for (int k = _qp.dims.horizon - 1; k >= 0; --k)
    {
        VecX h = _qu[k] +
                 _qp.stages[k].B.transpose() *
                   (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);

        _output.k[k].noalias() = -_H_llt[k].solve(h);

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
    ZoneScopedN("LQR: _solve");
    bool backward_fail = false;

    do {
        DEBUG_LOGLN("Qt:" << std::endl << _Qt);
        _output.P[_qp.dims.horizon] = _Qt;
        _output.p[_qp.dims.horizon] = _qt;

        // Backward pass
        for (int k = _qp.dims.horizon - 1; k >= 0; --k)
        {
            VecX h = _qu[k] + _qp.stages[k].B.transpose() *
                                (_output.p[k + 1] +
                                 _output.P[k + 1] * _qp.stages[k].gamma);

            _G[k].noalias() =
              _Qxu[k].transpose() +
              _qp.stages[k].B.transpose() * _output.P[k + 1] * _qp.stages[k].A;

            DEBUG_LOGLN("Quu[" << k << "]:" << std::endl << _Quu[k]);
            DEBUG_LOGLN("P[" << k + 1 << "]:" << std::endl << _output.P[k + 1]);

            _H[k].noalias() = _Quu[k] + _qp.stages[k].B.transpose() *
                                          _output.P[k + 1] * _qp.stages[k].B;

            _H[k] = _H[k].selfadjointView<Eigen::Upper>();

            MatX H_k = _H[k] + _reg * MatX::Identity(_qp.dims.stage_control[k],
                                                     _qp.dims.stage_control[k]);

            DEBUG_LOGLN("G[" << k << "]:" << std::endl << _G[k]);
            DEBUG_LOGLN("H[" << k << "]:" << std::endl << H_k);
            DEBUG_LOGLN("h[" << k << "]: " << h.transpose());

            _H_llt[k] = H_k.llt();
            _solve_successes.push_front(_H_llt[k].info() == Eigen::Success);
            _solve_success_sum += _solve_successes.front();

            if (_solve_successes.size() > _solve_success_sma_k)
            {
                _solve_success_sum -= _solve_successes.back();
                _solve_successes.pop_back();
            }

            if (!_solve_successes.front())
            {
                DEBUG_LOGLN("Backward pass failed");
                backward_fail = true;
                break;
            }

            _output.K[k] = -_H_llt[k].solve(_G[k]);
            _output.k[k] = -_H_llt[k].solve(h);

            _output.P[k] =
              _Qxx[k] +
              _qp.stages[k].A.transpose() * _output.P[k + 1] * _qp.stages[k].A -
              _output.K[k].transpose() * _H[k] * _output.K[k];

            _output.P[k] = _output.P[k].selfadjointView<Eigen::Upper>();

            _output.p[k] =
              _qx[k] + _output.K[k].transpose() * _qu[k] +
              (_qp.stages[k].A + _qp.stages[k].B * _output.K[k]).transpose() *
                (_output.p[k + 1] + _output.P[k + 1] * _qp.stages[k].gamma);
        }

        if (backward_fail)
        {
            DEBUG_LOGLN("Increasing LQR regularization");
            _reg *= _reg_factor;
            if (_reg > _reg_max)
            {
                throw std::runtime_error("LQR: Regularization failed");
            }
        }
    } while (backward_fail);

    if (_reg > _reg_init &&
        static_cast<double>(_solve_success_sum) / _solve_success_sma_k >
          _solve_success_sma_ratio)
    {
        _reg /= _reg_factor;
    }

    // Forward pass
    _output.x.stage(0) = VecX::Zero(_qp.dims.stage_state[0]);

    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        DEBUG_LOGLN("Setting x[" << k << "]:");
        DEBUG_LOGLN("A:" << std::endl << _qp.stages[k].A);
        DEBUG_LOGLN("B:" << std::endl << _qp.stages[k].B);
        DEBUG_LOGLN("K:" << std::endl << _output.K[k]);
        DEBUG_LOGLN("Previous x:" << std::endl
                                  << _output.x.stage(k).transpose());
        DEBUG_LOGLN("k:" << std::endl << _output.k[k].transpose());
        DEBUG_LOGLN("gamma:" << std::endl << _qp.stages[k].gamma.transpose());
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