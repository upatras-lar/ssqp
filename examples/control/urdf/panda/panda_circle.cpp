#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define STATE_SIZE 9
#define HORIZON    300
#define dt         0.01
#define P_WGT      1
#define V_WGT      0.001
#define TERM_WGT   100

// Circle properties:
// Center
#define cx 0
#define cy 0
#define cz 0.5
// Radius
#define cr 0.4
// Angular speed
#define ω 2 * M_PI / 3
// Initial phase
#define φ0 0

int main()
{
    // Cost trajectory to track
    std::vector<Eigen::Vector3d> traj(HORIZON + 1);

    for (uint k = 0; k <= HORIZON; ++k)
    {
        double φk = φ0 + ω * k * dt;
        traj[k] << cx + cr * std::cos(φk), cy + cr * sin(φk), cz;
    }

    // Define SSQP model
    ssqp::model::Model model("examples/urdf/panda.urdf", HORIZON, dt);

    Eigen::VectorXd init_state =
      model.inverseKinematics(traj[0],
                              Eigen::Matrix3d::Identity(),
                              "panda_joint7");

    // Define cost functions
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageCost>> costs(
      HORIZON + 1);

    for (uint k = 0; k <= HORIZON; ++k)
    {
        Eigen::VectorXd traj_k = traj[k];

        ssqp::model::EvaluationFunction cost_eval =
          [model, traj_k](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
        {
            Eigen::VectorXd pos;
            Eigen::MatrixXd rot;

            std::tie(pos, rot) = model.forwardKinematics(x, "panda_ee");

            Eigen::Vector<double, 1> res;

            res(0) =
              P_WGT * (pos - traj_k).squaredNorm() + V_WGT * u.squaredNorm();
            return res;
        };

        ssqp::model::GradientFunction cost_grad =
          [model, traj_k](const Eigen::VectorXd &x,
                          const Eigen::VectorXd &u,
                          Eigen::VectorXd &dx,
                          Eigen::VectorXd &du)
        {
            Eigen::MatrixXd Jfk = model.frameJacobian(x, "panda_ee");
            dx = 2 * Jfk.topRows(3).transpose() * P_WGT * (x - traj_k);
            du = 2 * V_WGT * u;
        };

        costs[k] =
          ssqp::model::stagewise::factory::make_cost(cost_eval, cost_grad);
    }

    model.finalize(costs, costs[HORIZON], {}, init_state);

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

    solver.write("visualizer/data/panda_circle");

    return 0;
}