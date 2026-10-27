#ifndef SSQP_TESTS_LINEAR_PROBLEM_HPP
#define SSQP_TESTS_LINEAR_PROBLEM_HPP
#include "Checks.hpp"
#include "model/Model.hpp"
#include "solvers/ssqp.hpp"

namespace ssqp
{
namespace tests
{
// Scalar integrator: x[k+1] = x[k] + u[k]. Separate dynamics objects per stage
// are essential: each stage owns its own initial guess.
class Integrator : public model::stagewise::StageDynamics
{
  public:
    uint state_size() const override
    {
        return 1;
    }
    uint control_size() const override
    {
        return 1;
    }
    uint output_size() const override
    {
        return 1;
    }
    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        return x + u;
    }
    void gradient(const VecX &, const VecX &, MatX &dx, MatX &du) const override
    {
        dx = MatX::Ones(1, 1);
        du = dx;
    }
    void hessprod_xx(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }
    void hessprod_xu(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }
    void hessprod_uu(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }
};
class Quadratic : public model::stagewise::StageCost
{
  public:
    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        return VecX::Constant(1, 0.5 * (x.squaredNorm() + u.squaredNorm()));
    }
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        dx = x;
        du = u;
    }
    void hessian_xx(const VecX &x, const VecX &, MatX &h) const override
    {
        h = MatX::Identity(x.size(), x.size());
    }
    void hessian_xu(const VecX &x, const VecX &u, MatX &h) const override
    {
        h = MatX::Zero(x.size(), u.size());
    }
    void hessian_uu(const VecX &, const VecX &u, MatX &h) const override
    {
        h = MatX::Identity(u.size(), u.size());
    }
};
class ControlBound : public model::stagewise::StageConstraints
{
  public:
    explicit ControlBound(double limit = 0.2)
      : _limit(limit)
      , _lb(VecX::Constant(1, -limit))
      , _ub(VecX::Constant(1, limit))
    {
    }
    uint output_size() const override
    {
        return 1;
    }
    const VecX &lower_bound() const override
    {
        return _lb;
    }
    const VecX &upper_bound() const override
    {
        return _ub;
    }
    VecX evaluate(const VecX &, const VecX &u) const override
    {
        return u;
    }
    void gradient(const VecX &, const VecX &, MatX &dx, MatX &du) const override
    {
        dx = MatX::Zero(1, 1);
        du = MatX::Ones(1, 1);
    }
    void hessprod_xx(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }
    void hessprod_xu(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }
    void hessprod_uu(const VecX &, const VecX &, const VecX &,
                     MatX &h) const override
    {
        h = MatX::Zero(1, 1);
    }

  private:
    double _limit;
    VecX   _lb, _ub;
};
inline std::shared_ptr<model::Model> integratorProblem(uint   horizon,
                                                       double initial,
                                                       bool   bounded = false)
{
    model::Model::Config config;
    for (uint k = 0; k <= horizon; ++k)
        config.dynamics.push_back(std::make_shared<Integrator>());
    config.stage_cost    = std::make_shared<Quadratic>();
    config.terminal_cost = std::make_shared<Quadratic>();
    config.init_states   = { VecX::Constant(1, initial) };
    auto model           = std::make_shared<model::Model>(horizon, 1.0);
    if (bounded)
        for (uint k = 0; k < horizon; ++k)
            model->addConstraints<ControlBound>(k);
    model->finalize(config);
    return model;
}
inline solvers::ssqp::Params accurateParams(solvers::ssqp::QPSolverType backend,
                                            bool parallel = false)
{
    solvers::ssqp::Params p;
    p.qp_solver_type            = backend;
    p.max_iter                  = 20;
    p.tolerance                 = 1e-6;
    p.constraint_tolerance      = 1e-6;
    p.admm_params.max_iter      = 100000;
    p.admm_params.abs_tolerance = 1e-7;
    p.admm_params.rel_tolerance = 1e-7;
    p.admm_params.rho_update    = false;
    p.admm_params.init_rho      = 1.0;
    p.hpipm_params.tol_stat     = 1e-10;
    p.hpipm_params.tol_eq       = 1e-10;
    p.hpipm_params.tol_ineq     = 1e-10;
    p.hpipm_params.tol_comp     = 1e-10;
    p.piqp_params.eps_abs       = 1e-10;
    p.piqp_params.eps_rel       = 1e-10;
    p.piqp_params.reg_prim      = 1e-12;
    p.piqp_params.parallel      = parallel;
    p.piqp_params.num_threads   = 2;
    return p;
}
// Closed-form backward scalar Riccati recursion, independent of ssqp/LQR.
inline std::pair<VecX, VecX> integratorSolution(uint horizon, double initial)
{
    VecX p = VecX::Ones(horizon + 1);
    for (int k = static_cast<int>(horizon) - 1; k >= 0; --k)
        p[k] = 1 + p[k + 1] / (1 + p[k + 1]);
    VecX x(horizon + 1), u(horizon);
    x[0] = initial;
    for (uint k = 0; k < horizon; ++k)
    {
        u[k]     = -p[k + 1] * x[k] / (1 + p[k + 1]);
        x[k + 1] = x[k] + u[k];
    }
    return { x, u };
}
// Feedback gains u[k] = K[k] x[k] of the same closed-form Riccati recursion
inline VecX integratorGains(uint horizon)
{
    VecX p = VecX::Ones(horizon + 1);
    for (int k = static_cast<int>(horizon) - 1; k >= 0; --k)
        p[k] = 1 + p[k + 1] / (1 + p[k + 1]);
    VecX K(horizon);
    for (uint k = 0; k < horizon; ++k) K[k] = -p[k + 1] / (1 + p[k + 1]);
    return K;
}
inline void checkTrajectory(const solvers::FlatVec &x,
                            const solvers::FlatVec &u, const VecX &expected_x,
                            const VecX &expected_u)
{
    require(x.stages() == static_cast<uint>(expected_x.size()),
            "State stage count");
    require(u.stages() == static_cast<uint>(expected_u.size()),
            "Control stage count");
    requireClose(x.vec, expected_x, "State trajectory");
    requireClose(u.vec, expected_u, "Control trajectory");
    for (uint k = 0; k < u.stages(); ++k)
    {
        require(x.stage(k).size() == 1 && u.stage(k).size() == 1,
                "Stage dimensions");
        requireClose(x.stage(k + 1),
                     x.stage(k) + u.stage(k),
                     "Dynamics feasibility");
    }
}
}
}
#endif
