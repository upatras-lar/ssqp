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
  , _solve_successes(params.solve_window, false)
  , _solve_success_head(0)
  , _solve_success_count(0)
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
    const uint H = _qp.dims.horizon;
    _cache_A_BK.resize(H);

    for (uint k = 0; k < H; ++k)
    {
        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];

        _Qxx[k].resize(nx, nx);
        _Qxu[k].resize(nx, nu);
        _Quu[k].resize(nu, nu);

        _qx[k].resize(nx);
        _qu[k].resize(nu);

        _G[k].resize(nu, nx);
        _H[k].resize(nu, nu);

        _output.P[k].resize(nx, nx);
        _output.p[k].resize(nx);

        _output.K[k].resize(nu, nx);
        _output.k[k].resize(nx, nx);

        _cache_A_BK[k].resize(nx, nx);
    }

    const uint nx_t = _qp.dims.stage_state[H];

    _Qt.resize(nx_t, nx_t);
    _qt.resize(nx_t);

    _output.P[H].resize(nx_t, nx_t);
    _output.p[H].resize(nx_t);

    uint max_x =
      *std::max_element(qp.dims.stage_state.begin(), qp.dims.stage_state.end());
    uint max_u = *std::max_element(qp.dims.stage_control.begin(),
                                   qp.dims.stage_control.end());
    uint max_c = *std::max_element(qp.dims.stage_constraint.begin(),
                                   qp.dims.stage_constraint.end());

    _tmp.c1.resize(max_c);
    _tmp.c2.resize(max_c);

    _tmp.CX1.resize(max_c, max_x);
    _tmp.CU1.resize(max_c, max_u);

    _tmp.x1.resize(max_x);
    _tmp.x2.resize(max_x);
    _tmp.x3.resize(max_x);
    _tmp.u1.resize(max_u);
    _tmp.u2.resize(max_u);

    _tmp.XX1.resize(max_x, max_x);
    _tmp.XX2.resize(max_x, max_x);
    _tmp.XX3.resize(max_x, max_x);
    _tmp.UU1.resize(max_u, max_u);
    _tmp.XU1.resize(max_x, max_u);
    _tmp.UX1.resize(max_u, max_x);
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

    const uint H = _qp.dims.horizon;

    for (int k = 0; k < H; ++k)
    {
        const auto &stage = _qp.stages[k];

        // Initialize unconstrained
        _qx[k]  = stage.qx;
        _qx[k] -= _sigma * current_sol.dx.stage(k);

        _qu[k]  = stage.qu;
        _qu[k] -= _sigma * current_sol.du.stage(k);

        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];
        const uint nc = stage.D.rows();

        if (nc == 0) continue;

        const auto &D = stage.D; // nc x nx
        const auto &E = stage.E; // nc x nu

        auto r = _tmp.c1.head(nc);

        r          = y.stage(k);
        r.array() -= _rho.stage(k).array() * z.stage(k).array();

        _qx[k].noalias() += D.transpose() * r;

        _qu[k].noalias() += E.transpose() * r;
    }

    _qt  = _qp.qx_t;
    _qt -= _sigma * current_sol.dx.stage(H);

    // Terminal stage
    if (_qp.Dt.rows() != 0)
    {
        const uint nx = _qp.dims.stage_state[H];
        const uint nc = _qp.Dt.rows();

        const auto &Dt = _qp.Dt;

        auto r = _tmp.c1.head(nc);

        r          = y.stage(H);
        r.array() -= _rho.stage(H).array() * z.stage(H).array();

        _qt.noalias() += Dt.transpose() * r;
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

    const uint H = _qp.dims.horizon;

    for (int k = 0; k < H; ++k)
    {
        const auto &stage = _qp.stages[k];

        // Initialize unconstrained
        _Qxx[k] = stage.Qxx;
        _Qxu[k] = stage.Qxu;
        _Quu[k] = stage.Quu;

        _qx[k]  = stage.qx;
        _qx[k] -= _sigma * current_sol.dx.stage(k);

        _qu[k]  = stage.qu;
        _qu[k] -= _sigma * current_sol.du.stage(k);

        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];
        const uint nc = stage.D.rows();

        if (nc == 0) continue;

        const auto &D = stage.D; // nc x nx
        const auto &E = stage.E; // nc x nu

        // Temporaries
        auto r  = _tmp.c1.head(nc);
        auto CX = _tmp.CX1.topLeftCorner(nc, nx); // rho ∘ D
        auto CU = _tmp.CU1.topLeftCorner(nc, nu); // rho ∘ E

        r          = y.stage(k);
        r.array() -= _rho.stage(k).array() * z.stage(k).array();

        // CX = diag(rho) * D   (row scaling)
        CX                    = D;
        CX.array().colwise() *= rho.stage(k).array();

        // CU = diag(rho) * E
        CU                    = E;
        CU.array().colwise() *= rho.stage(k).array();

        // Qxx
        auto XX      = _tmp.XX1.topLeftCorner(nx, nx);
        XX.noalias() = D.transpose() * CX;

        _Qxx[k]                    += XX;
        _Qxx[k].diagonal().array() += _sigma;

        // Qxu
        _Qxu[k].noalias() += D.transpose() * CU;

        // Quu
        auto UU      = _tmp.UU1.topLeftCorner(nu, nu);
        UU.noalias() = E.transpose() * CU;

        _Quu[k]                    += UU;
        _Quu[k].diagonal().array() += _sigma;

        // qx
        _qx[k].noalias() += D.transpose() * r;

        // qu
        _qu[k].noalias() += E.transpose() * r;
    }

    _Qt                     = _qp.Qxx_t;
    _Qt.diagonal().array() += _sigma;

    _qt  = _qp.qx_t;
    _qt -= _sigma * current_sol.dx.stage(H);

    // Terminal stage
    if (_qp.Dt.rows() != 0)
    {
        const uint nx = _qp.dims.stage_state[H];
        const uint nc = _qp.Dt.rows();

        const auto &Dt = _qp.Dt;

        auto r  = _tmp.c1.head(nc);
        auto CX = _tmp.CX1.topLeftCorner(nc, nx);

        r          = y.stage(H);
        r.array() -= _rho.stage(H).array() * z.stage(H).array();

        // rho ∘ Dt
        CX                    = Dt;
        CX.array().colwise() *= rho.stage(H).array();

        // Qt
        auto XX      = _tmp.XX1.topLeftCorner(nx, nx);
        XX.noalias() = Dt.transpose() * CX;

        _Qt += XX;

        // qt
        _qt.noalias() += Dt.transpose() * r;
    }

    _solve();

    return _output;
}

void lqr::_solve_cache_rho()
{
    ZoneScopedN("LQR: _solve_cache_rho");

    const uint H = _qp.dims.horizon;

    _output.p[H] = _qt;

    // Backward pass
    for (int k = H - 1; k >= 0; --k)
    {
        const uint nx = _qp.dims.stage_state[k];
        const uint nu = _qp.dims.stage_control[k];

        const auto &B     = _qp.stages[k].B;     // nx x nu
        const auto &gamma = _qp.stages[k].gamma; // nx

        auto x1 = _tmp.x1.head(nx);
        auto x2 = _tmp.x2.head(nx);
        auto x3 = _tmp.x3.head(nx);

        auto u1 = _tmp.u1.head(nu);

        // x1 = P_next * gamma
        x1.noalias() = _output.P[k + 1] * gamma;

        // x2 = p_next + x1
        x3  = _output.p[k + 1];
        x3 += x1;

        // h = qu + B^T * x2
        u1.noalias()  = B.transpose() * x3;
        u1           += _qu[k];

        // k = -H^{-1} h
        _output.k[k].noalias() = -_H_llt[k].solve(u1);

        // Cache M = A + B*K
        auto &M = _cache_A_BK[k];

        // x1 = K^T * qu
        x1.noalias() = _output.K[k].transpose() * _qu[k];

        // x2 = M^T * x3
        x2.noalias() = M.transpose() * x3;

        // p[k] = qx + x1 + x2
        _output.p[k]  = _qx[k];
        _output.p[k] += x1;
        _output.p[k] += x2;
    }

    // Forward pass
    _forward_pass();
}

void lqr::_solve()
{
    ZoneScopedN("LQR: _solve");

    const uint H = _qp.dims.horizon;
    bool       backward_fail;

    do {
        backward_fail = false;

        _output.P[H] = _Qt;
        _output.p[H] = _qt;

        // Backward pass
        for (int k = H - 1; k >= 0; --k)
        {
            const uint nx = _qp.dims.stage_state[k];
            const uint nu = _qp.dims.stage_control[k];

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
            x3  = _output.p[k + 1];
            x3 += x1;

            // h = qu + B^T * x2
            u1.noalias()  = B.transpose() * x3;
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

            // Regularization
            UU1                     = Hk;
            UU1.diagonal().array() += _reg;

            // LLT
            _H_llt[k].compute(UU1);

            const bool solve_success = (_H_llt[k].info() == Eigen::Success);

            if (_solve_success_count < _solve_success_sma_k)
            {
                _solve_successes[_solve_success_count++] = solve_success;
            }
            else
            {
                _solve_success_sum -= _solve_successes[_solve_success_head];
                _solve_successes[_solve_success_head] = solve_success;
                _solve_success_head =
                  (_solve_success_head + 1) % _solve_success_sma_k;
            }
            _solve_success_sum += solve_success;

            if (!solve_success)
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
            // XX1 is already P_next * A

            // XX2 = A^T * x1
            XX2.noalias() = A.transpose() * XX1;

            // UX1 = M1 - K^T * H * K
            UX1.noalias() = Hk * _output.K[k]; // nu x nx

            XX3.noalias() = _output.K[k].transpose() * UX1;

            _output.P[k]  = _Qxx[k];
            _output.P[k] += XX2;
            _output.P[k] -= XX3;

            _output.P[k] = _output.P[k].selfadjointView<Eigen::Upper>();

            // p[k]
            // x1 = K^T * qu
            x1.noalias() = _output.K[k].transpose() * _qu[k];

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
    _forward_pass();
}

inline void lqr::_forward_pass()
{
    const uint H = _qp.dims.horizon;

    _output.x.stage(0).setZero();

    for (int k = 0; k < H; ++k)
    {
        const uint nx = _qp.dims.stage_state[k];

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