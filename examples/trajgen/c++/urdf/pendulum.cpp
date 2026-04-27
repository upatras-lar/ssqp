#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp
#include "model/Model.hpp"  // ssqp::model::Model

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

int main()
{
    auto cost_eval = [](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return 0.5 * (x - x_target).transpose() * Q * (x - x_target) +
               0.5 * u.transpose() * R * u;
    };

    auto cost_grad = [](const Eigen::VectorXd &x,
                        const Eigen::VectorXd &u,
                        Eigen::VectorXd &dx,
                        Eigen::VectorXd &du)
    {
        dx = Q * (x - x_target);
        du = R * u;
    };

    auto cost_hess_xx = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Hxx) { Hxx = Q; };

    auto cost_hess_uu = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Huu) { Huu = R; };

    std::shared_ptr<ssqp::model::stagewise::StageCost> cost =
      ssqp::model::stagewise::factory::make_cost({},
                                                 cost_eval,
                                                 cost_grad,
                                                 ssqp::model::ZeroHessian_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    auto terminal_cost_eval =
      [](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    { return 0.5 * (x - x_target).transpose() * Qf * (x - x_target); };

    auto terminal_cost_grad = [](const Eigen::VectorXd &x,
                                 const Eigen::VectorXd &u,
                                 Eigen::VectorXd &dx,
                                 Eigen::VectorXd &du)
    { dx = Qf * (x - x_target); };

    auto terminal_cost_hess_xx = [](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Hxx) { Hxx = Qf; };

    std::shared_ptr<ssqp::model::stagewise::StageCost> terminal_cost =
      ssqp::model::stagewise::factory::make_cost({},
                                                 terminal_cost_eval,
                                                 terminal_cost_grad,
                                                 terminal_cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/assets/urdf/pendulum_point_mass.urdf";

    ssqp::model::Model::Config model_config;
    model_config.stage_cost    = cost;
    model_config.terminal_cost = terminal_cost;

    ssqp::model::Model model(HORIZON, dt, model_structure, model_config);

    ssqp::solvers::ssqp        solver(model);
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

    solver.write("examples/trajgen/data/pendulum.json");

    return 0;
}