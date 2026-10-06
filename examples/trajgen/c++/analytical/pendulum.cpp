#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp

#include "model/Model.hpp"               // ssqp::model::Model
#include "model/stagewise/StageCost.hpp" // ssqp::model::stagewise::StageCost
#include "model/stagewise/StageDynamics.hpp" // ssqp::model::stagewise::StageDynamics
#include "model/stagewise/StageConstraints.hpp" // ssqp::model::stagewise::StageConstraints

#define HORIZON 100

// Pendulum parameters
constexpr double g_l = 1.0;
constexpr double g_m = 1.0;
constexpr double g_b = 0.2;
constexpr double g_g = 9.81;

// System dimensions
constexpr int N = 2;
constexpr int M = 1;

// Time step
constexpr double dt = 0.05;

// Target state (for cost functions)
Eigen::Vector<double, N> x_target =
  (Eigen::Vector<double, N>() << M_PI, 0.0).finished();

// Cost matrices
Eigen::Matrix<double, N, N> Q =
  (Eigen::Matrix<double, N, N>() << 1.0, 0.0, 0.0, 1.0).finished();
Eigen::Matrix<double, M, M> R = 0.1 * Eigen::Matrix<double, M, M>::Identity();
Eigen::Matrix<double, N, N> Qf =
  100.0 * Eigen::Matrix<double, N, N>::Identity();

class Dynamics : public ssqp::model::stagewise::StageDynamics
{
    uint output_size() const override
    {
        return N;
    }

    uint state_size() const override
    {
        return N;
    }

    uint control_size() const override
    {
        return M;
    }

    ssqp::VecX dynamics(const ssqp::VecX &x, const ssqp::VecX &u) const
    {
        Eigen::Vector2d x_dot;
        x_dot(0) = x(1);
        x_dot(1) = -(g_g / g_l) * std::sin(x(0)) +
                   (1 / (g_m * g_l * g_l)) * u(0) - g_b * x(1);

        return x_dot;
    }

    ssqp::MatX dynamics_dx(const ssqp::VecX &x, const ssqp::VecX &u) const
    {
        Eigen::Matrix2d jac;
        jac << 0.0, 1.0, -(g_g / g_l) * std::cos(x(0)), -g_b;
        return jac;
    }

    ssqp::MatX dynamics_du(const ssqp::VecX &x, const ssqp::VecX &u) const
    {
        Eigen::Matrix<double, N, M> jac;
        jac << 0.0, 1.0 / (g_m * g_l * g_l);
        return jac;
    }

    ssqp::VecX evaluate(const ssqp::VecX &x, const ssqp::VecX &u) const override
    {
        /*
        Eigen::Vector2d k1 = dynamics(x, u);
        Eigen::Vector2d k2 = dynamics(x + 0.5 * dt * k1, u);
        Eigen::Vector2d k3 = dynamics(x + 0.5 * dt * k2, u);
        Eigen::Vector2d k4 = dynamics(x + dt * k3, u);
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        */

        return x + dynamics(x, u) * dt;
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        /*
        Eigen::Vector2d x2 = x + 0.5 * dt * dynamics(x, u);
        Eigen::Vector2d x3 = x + 0.5 * dt * dynamics(x2, u);
        Eigen::Vector2d x4 = x + dt * dynamics(x3, u);

        Eigen::Matrix2d dfdx  = dynamics_dx(x, u);
        Eigen::Matrix2d dfdx2 = dynamics_dx(x2, u);
        Eigen::Matrix2d dfdx3 = dynamics_dx(x3, u);
        Eigen::Matrix2d dfdx4 = dynamics_dx(x4, u);

        Eigen::Matrix<double, N, M> dfdu  = dynamics_du(x, u);
        Eigen::Matrix<double, N, M> dfdu2 = dynamics_du(x2, u);
        Eigen::Matrix<double, N, M> dfdu3 = dynamics_du(x3, u);
        Eigen::Matrix<double, N, M> dfdu4 = dynamics_du(x4, u);

        Eigen::Matrix2d dx2dx = Eigen::Matrix2d::Identity() + 0.5 * dt * dfdx;
        Eigen::Matrix2d dx3dx =
          Eigen::Matrix2d::Identity() + 0.5 * dt * dfdx2 * dx2dx;
        Eigen::Matrix2d dx4dx =
          Eigen::Matrix2d::Identity() + dt * dfdx3 * dx3dx;

        Eigen::Matrix<double, N, M> dx2du = 0.5 * dt * dfdu;
        Eigen::Matrix<double, N, M> df2du = dfdx2 * dx2du + dfdu2;

        Eigen::Matrix<double, N, M> dx3du = 0.5 * dt * df2du;
        Eigen::Matrix<double, N, M> df3du = dfdx3 * dx3du + dfdu3;

        Eigen::Matrix<double, N, M> dx4du = dt * df3du;
        Eigen::Matrix<double, N, M> df4du = dfdx4 * dx4du + dfdu4;

        dX = Eigen::Matrix2d::Identity() +
             (dt / 6.0) * (dfdx + 2.0 * dfdx2 * dx2dx + 2.0 * dfdx3 * dx3dx +
                           dfdx4 * dx4dx);

        dU = (dt / 6.0) * (dfdu + 2.0 * df2du + 2.0 * df3du + df4du);
        */

        dX       = Eigen::Matrix2d::Zero();
        dX(0, 0) = 1;
        dX(0, 1) = dt;
        dX(1, 1) = 1;

        dU       = Eigen::Matrix<double, 2, 1>::Zero();
        dU(1, 0) = dt;
    }

    void hessprod_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

class TerminalCost : public ssqp::model::stagewise::StageCost
{
  public:
    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return 0.5 * (x - x_target).transpose() * Qf * (x - x_target);
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {
        dx = Qf * (x - x_target);
        du = Eigen::Vector<double, 1>::Zero();
    }

    void hessian_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx = Qf;
    }

    void hessian_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessian_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = Eigen::Matrix<double, 2, 1>::Zero();
    }
};

class Cost : public ssqp::model::stagewise::StageCost
{
  public:
    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return 0.5 * (x - x_target).transpose() * Q * (x - x_target) +
               0.5 * u.transpose() * R * u;
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {

        dx = Q * (x - x_target);
        du = R * u;
    }

    void hessian_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx = Q;
    }

    void hessian_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessian_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = R;
    }
};

class Constraints : public ssqp::model::stagewise::StageConstraints
{
    uint output_size() const override
    {
        return 1;
    }

    const ssqp::VecX &lower_bound() const override
    {
        return _lb;
    }

    const ssqp::VecX &upper_bound() const override
    {
        return _ub;
    }

    ssqp::VecX evaluate(const ssqp::VecX &x, const ssqp::VecX &u) const override
    {
        Eigen::VectorXd v = Eigen::Vector<double, 1>::Zero();

        v(0) = u(0);

        return v;
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix<double, 1, 2>::Zero();
        dU = Eigen::Matrix<double, 1, 1>::Zero();

        dU(0, 0) = 1;
    }

    void hessprod_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }

    const ssqp::VecX _lb = (Eigen::Vector<double, 1>() << -7).finished();
    const ssqp::VecX _ub = (Eigen::Vector<double, 1>() << 7).finished();
};

int main()
{
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints(HORIZON + 1);

    for (uint k = 0; k < HORIZON; ++k)
    {
        constraints[k] = std::make_shared<Constraints>();
    }

    constraints[HORIZON] = nullptr;

    // Use config to pass model stage functions
    ssqp::model::Model::Config model_config;
    // config.stage_cost: Add this to every non-terminal stage's cost
    model_config.stage_cost = std::make_shared<Cost>();
    // config.terminal_cost: Use this as the terminal cost
    model_config.terminal_cost = std::make_shared<TerminalCost>();

    model_config.dynamics.resize(HORIZON + 1);
    std::shared_ptr<Dynamics> dyn = std::make_shared<Dynamics>();

    for (uint i = 0; i <= HORIZON; ++i)
    {
        model_config.dynamics[i] = dyn;
    }

    model_config.constraints = constraints;

    ssqp::model::Model model(HORIZON, dt, model_config);

    // Custom parameters for SSQP
    ssqp::solvers::ssqp::Params params;
    params.tolerance = 1e-2;

    // Initialize SSQP solver with created model and custom SSQP parameters
    ssqp::solvers::ssqp        solver(model, params);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    if (state == ssqp::solvers::SolverState::SUCCESS ||
        state == ssqp::solvers::SolverState::LOOSE_SUCCESS)
    {
        std::cout << "x: " << std::endl << solution.x << std::endl;
        std::cout << "u: " << std::endl << solution.u << std::endl;
    }

    return 0;
}