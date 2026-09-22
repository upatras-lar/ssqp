#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp
#include "model/Model.hpp"  // ssqp::model::Model

#define STATE_SIZE 9
#define HORIZON    200
#define dt         0.01
#define Q_WGT      1.0  // Configuration weight
#define U_WGT      3e-3 // Control weight
#define QF_WGT     3.0  // Terminal weight

// Which type of move to make (values in [0, 4])
// @script-parameter
#define TARGET 2

int main()
{
    Eigen::VectorXd init(2 * STATE_SIZE);
    Eigen::VectorXd target(STATE_SIZE);

    init.tail(STATE_SIZE).setZero();

    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints(HORIZON + 1, nullptr);

    switch (TARGET)
    {
    // Simple reach
    case 0:
        init.head(STATE_SIZE) << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
          0.020, 0.020;
        target << 0.3, -0.5, 0.0, -2.0, 0.0, 2.0, 0.7, 0.020, 0.020;
        break;
    // Side sweep
    case 1:
        init.head(STATE_SIZE) << -0.2, -0.6, 0.2, -1.8, 0.1, 1.9, 0.4, 0.020,
          0.020;
        target << 0.4, -0.8, 0.3, -1.9, -0.1, 1.8, 0.5, 0.020, 0.020;
        break;
    // Overhead lift
    case 2:
        init.head(STATE_SIZE) << 0.1, -0.7, 0.1, -1.8, 0.0, 1.7, 0.8, 0.000,
          0.000;
        target << 0.0, -1.2, 0.0, -1.2, 0.0, 1.2, 0.0, 0.000, 0.000;
        break;
    // Grasp sequence
    case 3:
        init.head(STATE_SIZE) << 0.0, -0.5, 0.0, -2.2, 0.0, 2.0, 0.5, 0.040,
          0.040;
        target << 0.0, -0.5, 0.0, -2.2, 0.0, 2.0, 0.5, 0.005, 0.005;
        break;
    // Approach and grasp
    case 4:
        init.head(STATE_SIZE) << 0.2, -0.7, 0.1, -1.9, 0.1, 1.8, 0.6, 0.020,
          0.020;
        target << 0.15, -1.0, 0.1, -1.5, 0.0, 1.5, 0.6, 0.000, 0.000;
        break;
    }

    auto cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << U_WGT * u.transpose() * u +
                     Q_WGT * (x.head(STATE_SIZE) - target).squaredNorm())
          .finished();
    };

    auto cost_grad = [target](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u,
                              Eigen::VectorXd &dx,
                              Eigen::VectorXd &du)
    {
        dx                  = Eigen::VectorXd::Zero(2 * STATE_SIZE);
        dx.head(STATE_SIZE) = 2 * Q_WGT * (x.head(STATE_SIZE) - target);
        du                  = 2 * U_WGT * u;
    };

    auto cost_hess_xx = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Hxx)
    {
        Hxx = Eigen::MatrixXd::Zero(2 * STATE_SIZE, 2 * STATE_SIZE);
        Hxx.topLeftCorner(STATE_SIZE, STATE_SIZE)
          .diagonal()
          .setConstant(2 * Q_WGT);
    };

    auto cost_hess_uu = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Huu)
    { Huu = 2 * U_WGT * Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE); };

    std::shared_ptr<ssqp::model::stagewise::StageCost> cost =
      ssqp::model::stagewise::factory::make_cost({},
                                                 cost_eval,
                                                 cost_grad,
                                                 cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    auto terminal_cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << QF_WGT *
                     (Q_WGT * (x.head(STATE_SIZE) - target).squaredNorm() +
                      x.tail(STATE_SIZE).squaredNorm()))
          .finished();
    };

    auto terminal_cost_grad = [target](const Eigen::VectorXd &x,
                                       const Eigen::VectorXd &u,
                                       Eigen::VectorXd &dx,
                                       Eigen::VectorXd &du)
    {
        dx = Eigen::VectorXd::Zero(2 * STATE_SIZE);
        dx.head(STATE_SIZE) =
          QF_WGT * 2 * Q_WGT * (x.head(STATE_SIZE) - target);
        dx.tail(STATE_SIZE) = QF_WGT * 2 * x.tail(STATE_SIZE);
    };

    auto terminal_cost_hess_xx = [](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Hxx)
    {
        Hxx = QF_WGT * 2 * Q_WGT *
              Eigen::MatrixXd::Identity(2 * STATE_SIZE, 2 * STATE_SIZE);
    };

    std::shared_ptr<ssqp::model::stagewise::StageCost> terminal_cost =
      ssqp::model::stagewise::factory::make_cost({},
                                                 terminal_cost_eval,
                                                 terminal_cost_grad,
                                                 terminal_cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 ssqp::model::ZeroHessian_uu);

    ssqp::model::Model::Structure model_structure;
    model_structure.from_preset = ssqp::model::Preset::Panda;

    ssqp::model::Model::Config model_config;
    model_config.stage_cost    = cost;
    model_config.terminal_cost = terminal_cost;
    model_config.constraints   = constraints;
    model_config.init_states   = { init };

    ssqp::model::Model model(HORIZON, dt, model_structure, model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 1e-2;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-3;
    admm_params.max_iter      = 1000;

    params.admm_params = admm_params;

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

    solver.write("examples/trajgen/data/panda_poses.json");

    return 0;
}