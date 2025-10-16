#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define HORIZON 30
#define dt      0.01
#define P_WGT   5
#define V_WGT   0.0001

int main()
{
    Eigen::VectorXd target(7);
    // target << 0.5, 0, 0, 0, 0, 0, 0;
    target << 0.05, -0.03, 0.06, 0.0, 0.02, -0.02, 0.01;

    ssqp::model::EvaluationFunction cost_eval =
      [target](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        return (Eigen::Vector<double, 1>()
                << V_WGT * u.transpose() * u +
                     P_WGT * (x.head(7) - target).squaredNorm())
          .finished();
    };

    ssqp::model::GradientFunction cost_grad = [target](const Eigen::VectorXd &x,
                                                       const Eigen::VectorXd &u,
                                                       Eigen::VectorXd &dx,
                                                       Eigen::VectorXd &du)
    {
        dx         = Eigen::VectorXd::Zero(14);
        dx.head(7) = 2 * P_WGT * (x.head(7) - target);
        du         = 2 * V_WGT * u;
    };

    ssqp::model::HessianFunction cost_hess_xx = [](const Eigen::VectorXd &x,
                                                   const Eigen::VectorXd &u,
                                                   Eigen::MatrixXd &Hxx)
    { Hxx = 2 * P_WGT * Eigen::MatrixXd::Identity(2 * 7, 2 * 7); };

    ssqp::model::HessianFunction cost_hess_uu = [](const Eigen::VectorXd &x,
                                                   const Eigen::VectorXd &u,
                                                   Eigen::MatrixXd &Huu)
    { Huu = 2 * V_WGT * Eigen::MatrixXd::Identity(7, 7); };

    std::shared_ptr<const ssqp::model::stagewise::StageCost> cost =
      ssqp::model::stagewise::factory::make_cost(cost_eval,
                                                 cost_grad,
                                                 cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 cost_hess_uu);

    ssqp::model::Model model("examples/urdf/iiwa.urdf",
                             HORIZON,
                             dt,
                             cost,
                             cost);

    ssqp::solvers::ssqp        solver(model);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    if (state == ssqp::solvers::SolverState::SUCCESS ||
        state == ssqp::solvers::SolverState::LOOSE_SUCCESS)
    {
        std::cout << "x: " << solution.x << std::endl;
        std::cout << "u: " << solution.u << std::endl;
    }

    return 0;
}