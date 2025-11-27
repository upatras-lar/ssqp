#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define STATE_SIZE 9
#define HORIZON    200
#define dt         0.01
#define P_WGT      1
#define V_WGT      1e-5
#define TERM_WGT   10
#define TARGET     0

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
                << V_WGT * u.transpose() * u +
                     P_WGT * (x.head(STATE_SIZE) - target).squaredNorm())
          .finished();
    };

    auto cost_grad = [target](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u,
                              Eigen::VectorXd &dx,
                              Eigen::VectorXd &du)
    {
        dx                  = Eigen::VectorXd::Zero(2 * STATE_SIZE);
        dx.head(STATE_SIZE) = 2 * P_WGT * (x.head(STATE_SIZE) - target);
        du                  = 2 * V_WGT * u;
    };

    auto cost_hess_xx = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Hxx)
    {
        Hxx = Eigen::MatrixXd::Zero(2 * STATE_SIZE, 2 * STATE_SIZE);
        Hxx.topLeftCorner(STATE_SIZE, STATE_SIZE)
          .diagonal()
          .setConstant(2 * P_WGT);
    };

    auto cost_hess_uu = [](const Eigen::VectorXd &x,
                           const Eigen::VectorXd &u,
                           Eigen::MatrixXd &Huu)
    { Huu = 2 * V_WGT * Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE); };

    std::shared_ptr<ssqp::model::stagewise::StageCost> cost =
      ssqp::model::stagewise::factory::make_cost(cost_eval,
                                                 cost_grad,
                                                 cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    auto terminal_cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << TERM_WGT *
                     (P_WGT * (x.head(STATE_SIZE) - target).squaredNorm() +
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
          TERM_WGT * 2 * P_WGT * (x.head(STATE_SIZE) - target);
        dx.tail(STATE_SIZE) = TERM_WGT * 2 * x.tail(STATE_SIZE);
    };

    auto terminal_cost_hess_xx = [](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Hxx)
    {
        Hxx = TERM_WGT * 2 * P_WGT *
              Eigen::MatrixXd::Identity(2 * STATE_SIZE, 2 * STATE_SIZE);
    };

    std::shared_ptr<ssqp::model::stagewise::StageCost> terminal_cost =
      ssqp::model::stagewise::factory::make_cost(terminal_cost_eval,
                                                 terminal_cost_grad,
                                                 terminal_cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 ssqp::model::ZeroHessian_uu);

    Eigen::VectorXd lb(1), ub(1);
    lb(0) = 1.6;
    ub(0) = 1.6;

    auto constraint_eval =
      [](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    { return (Eigen::VectorXd(1) << x(0)).finished(); };

    auto constraint_grad = [](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u,
                              Eigen::MatrixXd &dX,
                              Eigen::MatrixXd &dU)
    {
        dX = Eigen::MatrixXd::Zero(1, x.size());
        dU = Eigen::MatrixXd::Zero(1, u.size());

        dX(0, 0) = 1;
    };

    std::shared_ptr<ssqp::model::stagewise::StageConstraints> constraint =
      ssqp::model::stagewise::factory::make_constraints(1,
                                                        1,
                                                        lb,
                                                        ub,
                                                        constraint_eval,
                                                        constraint_grad);

    constraints[HORIZON / 2] = constraint;

    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/urdf/panda/panda.urdf";

    ssqp::model::Model::Config model_config;
    model_config.stage_cost    = cost;
    model_config.terminal_cost = terminal_cost;
    model_config.constraints   = constraints;
    model_config.init_states   = { init };

    ssqp::model::Model model(HORIZON, dt, model_structure, model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 5e-2;

    ssqp::solvers::admm_qp::Params admm_params;

    ssqp::solvers::ssqp        solver(model, params, admm_params);
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

    solver.write("examples/data/panda_poses");

    return 0;
}