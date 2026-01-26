#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp
#include "model/Model.hpp"  // ssqp::model::Model

#define HORIZON  200
#define dt       0.01
#define Q_WGT    2    // Configuration weight
#define U_WGT    1e-3 // Control weight
#define TERM_WGT 100  // Terminal weight

// Which type of move to make (values in [0, 3])
// @script-parameter
#define TARGET 1

int main()
{
    Eigen::VectorXd init(14);
    Eigen::VectorXd target(7);

    switch (TARGET)
    {
    // Small move
    case 0:
        init << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        target << 0.5, -0.3, 0.6, 0.0, 0.2, -0.2, 0.1;
        break;
    // Mid range move
    case 1:
        init << 0.0, -0.7, 0.0, 0.8, 0.0, -0.5, 0.0, 0, 0, 0, 0, 0, 0, 0;
        target << 1.2, 0.6, -1.0, 0.2, 0.4, 0.0, 0.8;
        break;
    // Elbow-up pose
    case 2:
        init << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        target << 0.9, -0.6, 1.1, 0.2, 0.15, -0.1, 0.0;
        break;
    // Elbow-down pose
    case 3:
        init << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0;
        target << -0.9, 0.6, -1.1, -0.2, -0.15, 0.1, 0.0;
    }

    // Define cost using lambdas
    auto cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << U_WGT * u.transpose() * u +
                     Q_WGT * (x.head(7) - target).squaredNorm())
          .finished();
    };

    auto cost_grad = [target](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u,
                              Eigen::VectorXd &dx,
                              Eigen::VectorXd &du)
    {
        dx         = Eigen::VectorXd::Zero(14);
        dx.head(7) = 2 * Q_WGT * (x.head(7) - target);
        du         = 2 * U_WGT * u;
    };

    auto cost_hess_xx = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Hxx)
    { Hxx = 2 * Q_WGT * Eigen::MatrixXd::Identity(2 * 7, 2 * 7); };

    auto cost_hess_uu = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Huu)
    { Huu = 2 * U_WGT * Eigen::MatrixXd::Identity(7, 7); };

    // Create StageCost object using the make_cost helper
    std::shared_ptr<ssqp::model::stagewise::StageCost> cost =
      ssqp::model::stagewise::factory::make_cost(cost_eval,
                                                 cost_grad,
                                                 cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    // Define terminal cost with a larger terminal weight
    auto terminal_cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << TERM_WGT * (Q_WGT * (x.head(7) - target).squaredNorm() +
                               x.tail(7).squaredNorm()))
          .finished();
    };

    auto terminal_cost_grad = [target](const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &u,
                                       Eigen::VectorXd &dx,
                                       Eigen::VectorXd &du)
    {
        dx         = Eigen::VectorXd::Zero(14);
        dx.head(7) = TERM_WGT * 2 * Q_WGT * (x.head(7) - target);
        dx.tail(7) = TERM_WGT * 2 * x.tail(7);
    };

    auto terminal_cost_hess_xx = [](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Hxx)
    { Hxx = TERM_WGT * 2 * Q_WGT * Eigen::MatrixXd::Identity(2 * 7, 2 * 7); };

    std::shared_ptr<ssqp::model::stagewise::StageCost> terminal_cost =
      ssqp::model::stagewise::factory::make_cost(terminal_cost_eval,
                                                 terminal_cost_grad,
                                                 terminal_cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 ssqp::model::ZeroHessian_uu);

    // Use model structure object to pass URDF path
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/urdf/iiwa.urdf";

    // Use model config object to set the stage and terminal costs, as well as
    // the initial state
    ssqp::model::Model::Config model_config;
    model_config.stage_cost    = cost;
    model_config.terminal_cost = terminal_cost;
    model_config.init_states   = { init };

    // Initialize model using mixed constructor (with both structure and config)
    // Since we pass a config, this constructor will also finalize the model
    ssqp::model::Model model(HORIZON, dt, model_structure, model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 1e-2;

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

    solver.write("examples/trajgen/data/iiwa.json");

    return 0;
}