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
    uint max_x =
      *std::max_element(qp.dims.stage_state.begin(), qp.dims.stage_state.end());
    uint max_u = *std::max_element(qp.dims.stage_control.begin(),
                                   qp.dims.stage_control.end());

    _tmp.x1.resize(max_x);
    _tmp.x2.resize(max_x);
    _tmp.x3.resize(max_x);
    _tmp.u1.resize(max_u);

    _tmp.XX1.resize(max_x, max_x);
    _tmp.XX2.resize(max_x, max_x);
    _tmp.XX3.resize(max_x, max_x);
    _tmp.UU1.resize(max_u, max_u);
    _tmp.XU1.resize(max_x, max_u);
    _tmp.UX1.resize(max_u, max_x);

    _cache_A_BK.resize(_qp.dims.horizon);
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        _cache_A_BK[k].resize(_qp.dims.stage_state[k], _qp.dims.stage_state[k]);
    }
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

lqr::Output &lqr::constrained(const admm_qp::Solution &current_sol,
                              const FlatVec &z, const FlatVec &y)
{
    ZoneScopedN("LQR: constrained (cached)");
    // Calculate constraint terms, assuming rho did not change
    for (uint k = 0; k < _qp.dims.horizon; ++k)
    {
        if (_qp.stages[k].D.rows() == 0)
        {
            continue;
        }

        _qx[k].noalias() = _qp.stages[k].qx +
                           _qp.stages[k].D.transpose() * y.stage(k) -
                           _qp.stages[k].D.transpose() *
                             _rho.stage(k).asDiagonal() * z.stage(k) -
                           _sigma * current_sol.dx.stage(k);

        _qu[k].noalias() = _qp.stages[k].qu +
                           _qp.stages[k].E.transpose() * y.stage(k) -
                           _qp.stages[k].E.transpose() *
                             _rho.stage(k).asDiagonal() * z.stage(k) -
                           _sigma * current_sol.du.stage(k);
    }

    if (_qp.Dt.rows() != 0)
    {
        _qt.noalias() =
          _qp.qx_t + _qp.Dt.transpose() * y.stage(_qp.dims.horizon) -
          _qp.Dt.transpose() * _rho.stage(_qp.dims.horizon).asDiagonal() *
            z.stage(_qp.dims.horizon) +
          _sigma * current_sol.dx.stage(_qp.dims.horizon);
    }

    _solve_cache_rho();

    return _output;
}

lqr::Output &lqr::constrained(const admm_qp::Solution &current_sol,
                              const FlatVec &z, const FlatVec &y,
                              const FlatVec &rho)
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
          _qp.stages[k].qx + _qp.stages[k].D.transpose() * y.stage(k) -
          _qp.stages[k].D.transpose() * rho.stage(k).asDiagonal() * z.stage(k) -
          _sigma * current_sol.dx.stage(k);

        _qu[k].noalias() =
          _qp.stages[k].qu + _qp.stages[k].E.transpose() * y.stage(k) -
          _qp.stages[k].E.transpose() * rho.stage(k).asDiagonal() * z.stage(k) -
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
          _qp.qx_t + _qp.Dt.transpose() * y.stage(_qp.dims.horizon) -
          _qp.Dt.transpose() * rho.stage(_qp.dims.horizon).asDiagonal() *
            z.stage(_qp.dims.horizon) -
          _sigma * current_sol.dx.stage(_qp.dims.horizon);
    }

    _solve();

    return _output;
}

void lqr::_solve_cache_rho()
{
    ZoneScopedN("LQR: _solve_cache_rho");

    const int H = _qp.dims.horizon;

    _output.p[H] = _qt;

    // Backward pass
    for (int k = H - 1; k >= 0; --k)
    {
        const int nx = _qp.dims.stage_state[k];
        const int nu = _qp.dims.stage_control[k];

        const auto &A     = _qp.stages[k].A;     // nx x nx
        const auto &B     = _qp.stages[k].B;     // nx x nu
        const auto &gamma = _qp.stages[k].gamma; // nx

        auto x1 = _tmp.x1.head(nx);
        auto x2 = _tmp.x2.head(nx);
        auto x3 = _tmp.x3.head(nx);

        auto u1 = _tmp.u1.head(nu);

        // x1 = P_next * gamma
        x1.noalias() = _output.P[k + 1] * gamma;

        // x2 = p_next + x1
        x2  = _output.p[k + 1];
        x2 += x1;

        // h = qu + B^T * x2
        u1.noalias()  = B.transpose() * x2;
        u1           += _qu[k];

        // k = -H^{-1} h
        _output.k[k].noalias() = -_H_llt[k].solve(u1);

        // Cache M = A + B*K
        auto &M      = _cache_A_BK[k];
        M.noalias()  = B * _output.K[k];
        M           += A;

        // x1 = K^T * qu
        x1.noalias() = _output.K[k].transpose() * _qu[k];

        // x2 = P_next * gamma
        x2.noalias() = _output.P[k + 1] * gamma;

        // x3 = p_next + x2
        x3  = _output.p[k + 1];
        x3 += x2;

        // x2 = M^T * x3
        x2.noalias() = M.transpose() * x3;

        // p[k] = qx + x1 + x2
        _output.p[k]  = _qx[k];
        _output.p[k] += x1;
        _output.p[k] += x2;
    }

    // Forward pass
    _output.x.stage(0).setZero();

    for (int k = 0; k < H; ++k)
    {
        const int nx = _qp.dims.stage_state[k];
        const int nu = _qp.dims.stage_control[k];

        const auto &B     = _qp.stages[k].B;     // nx x nu
        const auto &gamma = _qp.stages[k].gamma; // nx
        const auto &M     = _cache_A_BK[k];      // nx x nx

        auto x1 = _tmp.x1.head(nx);
        auto x2 = _tmp.x2.head(nx);

        // x1 = M * x_k
        x1.noalias() = M * _output.x.stage(k);

        // x2 = B * k
        x2.noalias() = B * _output.k[k];

        // x_{k+1}
        _output.x.stage(k + 1)  = x1;
        _output.x.stage(k + 1) += x2;
        _output.x.stage(k + 1) += gamma;

        // u_k = K*x_k + k
        _output.u.stage(k).noalias()  = _output.K[k] * _output.x.stage(k);
        _output.u.stage(k)           += _output.k[k];
    }
}

void lqr::_solve()
{
    ZoneScopedN("LQR: _solve");

    const int H             = _qp.dims.horizon;
    bool      backward_fail = false;

    do
    {
        _output.P[H] = _Qt;
        _output.p[H] = _qt;

        // Backward pass
        for (int k = H - 1; k >= 0; --k)
        {
            const int nx = _qp.dims.stage_state[k];
            const int nu = _qp.dims.stage_control[k];

            const auto &A     = _qp.stages[k].A; // nx x nx
            const auto &B     = _qp.stages[k].B; // nx x nu
            const auto &gamma = _qp.stages[k].gamma;

            auto x1 = _tmp.x1.head(nx);
            auto x2 = _tmp.x2.head(nx);
            auto x3 = _tmp.x3.head(nx);

            auto u1 = _tmp.u1.head(nu);

            auto XX1 = _tmp.XX1.topLeftCorner(nx, nx);
            auto XX2 = _tmp.XX2.topLeftCorner(nx, nx);
            auto XX3 = _tmp.XX3.topLeftCorner(nx, nx);

            auto XU1 = _tmp.XU1.topLeftCorner(nx, nu);
            auto UX1 = _tmp.UX1.topLeftCorner(nu, nx);

            auto UU1 = _tmp.UU1.topLeftCorner(nu, nu);

            // x1 = P_next * gamma
            x1.noalias() = _output.P[k + 1] * gamma;

            // x2 = p_next + x1
            x2  = _output.p[k + 1];
            x2 += x1;

            // h = qu + B^T * x2
            u1.noalias()  = B.transpose() * x2;
            u1           += _qu[k];

            // G = Qxu^T + B^T * P_next * A
            auto &Gk = _G[k]; // nu x nx

            XX1.noalias()  = _output.P[k + 1] * A;
            Gk.noalias()   = B.transpose() * XX1;
            Gk            += _Qxu[k].transpose();

            // H = Quu + B^T * P_next * B
            auto &Hk      = _H[k];
            XU1.noalias() = _output.P[k + 1] * B; // nx x nu

            Hk.noalias()  = B.transpose() * XU1; // nu x nu
            Hk           += _Quu[k];

            Hk = Hk.selfadjointView<Eigen::Upper>();

            // regularization
            UU1                     = Hk;
            UU1.diagonal().array() += _reg;

            // LLT
            _H_llt[k].compute(UU1);

            _solve_successes.push_front(_H_llt[k].info() == Eigen::Success);
            _solve_success_sum += _solve_successes.front();

            if (_solve_successes.size() > _solve_success_sma_k)
            {
                _solve_success_sum -= _solve_successes.back();
                _solve_successes.pop_back();
            }

            if (!_solve_successes.front())
            {
                backward_fail = true;
                break;
            }

            // K = -H^{-1} G
            _output.K[k].noalias() = -_H_llt[k].solve(_G[k]);

            // k = -H^{-1} h
            _output.k[k].noalias() = -_H_llt[k].solve(u1);

            // Cache M = A + B*K
            auto &M      = _cache_A_BK[k];
            M.noalias()  = B * _output.K[k];
            M           += A;

            // P[k]
            // x1 = P_next * A
            XX1.noalias() = _output.P[k + 1] * A;

            // tmp_xx = A^T * x1
            XX2.noalias() = A.transpose() * XX1;

            // tmp_xx = M1 - K^T * H * K
            UX1.noalias() = Hk * _output.K[k]; // nu x nx

            XX3.noalias() = _output.K[k].transpose() * UX1;

            _output.P[k]  = _Qxx[k];
            _output.P[k] += XX2;
            _output.P[k] -= XX3;

            _output.P[k] = _output.P[k].selfadjointView<Eigen::Upper>();

            // p[k]
            // x1 = K^T * qu
            x1.noalias() = _output.K[k].transpose() * _qu[k];

            // x2 = P_next * gamma
            x2.noalias() = _output.P[k + 1] * gamma;

            // x3 = p_next + x2
            x3  = _output.p[k + 1];
            x3 += x2;

            // x2 = M^T * x3
            x2.noalias() = M.transpose() * x3;

            _output.p[k]  = _qx[k];
            _output.p[k] += x1;
            _output.p[k] += x2;
        }

        if (backward_fail)
        {
            _reg *= _reg_factor;
            if (_reg > _reg_max)
                throw std::runtime_error("LQR: Regularization failed");
        }

    } while (backward_fail);

    if (_reg > _reg_init &&
        static_cast<double>(_solve_success_sum) / _solve_success_sma_k >
          _solve_success_sma_ratio)
    {
        _reg /= _reg_factor;
    }

    // Forward pass
    _output.x.stage(0).setZero();

    for (int k = 0; k < H; ++k)
    {
        const int nx = _qp.dims.stage_state[k];
        const int nu = _qp.dims.stage_control[k];

        const auto &B     = _qp.stages[k].B;
        const auto &gamma = _qp.stages[k].gamma;
        const auto &M     = _cache_A_BK[k];

        auto x1 = _tmp.x1.head(nx);
        auto x2 = _tmp.x2.head(nx);

        // x_{k+1}
        x1.noalias() = M * _output.x.stage(k);
        x2.noalias() = B * _output.k[k];

        _output.x.stage(k + 1)  = x1;
        _output.x.stage(k + 1) += x2;
        _output.x.stage(k + 1) += gamma;

        // u_k
        _output.u.stage(k).noalias()  = _output.K[k] * _output.x.stage(k);
        _output.u.stage(k)           += _output.k[k];
    }
}
}
}