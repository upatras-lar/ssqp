#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"

std::shared_ptr<ssqp::model::stagewise::StageCost> getCost(
  const double weight = 1)
{
    auto cost_eval =
      [weight](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        Eigen::VectorXd q(6);
        q << std::sin(x[0]), std::sin(x[1]), 1 - std::cos(x[0]),
          1 - std::cos(x[1]), 0.1 * x[2], 0.1 * x[3];

        Eigen::Vector<double, 1> res;
        res[0] = weight * 0.5 * q.transpose() * q;

        return res;
    };

    auto cost_grad = [weight](const Eigen::VectorXd &x,
                              const Eigen::VectorXd &u,
                              Eigen::VectorXd &dx,
                              Eigen::VectorXd &du)
    {
        double s1 = std::sin(x[0]);
        double s2 = std::sin(x[1]);

        dx    = Eigen::Vector4d();
        dx[0] = weight * s1;
        dx[1] = weight * s2;
        dx[2] = weight * 0.01 * x[2];
        dx[3] = weight * 0.01 * x[3];

        du = Eigen::Vector2d();
        du.setZero();
    };

    auto cost_hess_xx = [weight](const Eigen::VectorXd &x,
                                 const Eigen::VectorXd &u,
                                 Eigen::MatrixXd &Hxx)
    {
        double c1 = std::cos(x[0]);
        double c2 = std::cos(x[1]);

        Hxx       = Eigen::Matrix4d::Zero();
        Hxx(0, 0) = weight * c1;
        Hxx(1, 1) = weight * c2;
        Hxx(2, 2) = weight * 0.01;
        Hxx(3, 3) = weight * 0.01;
    };

    return ssqp::model::stagewise::factory::make_cost(cost_eval,
                                                      cost_grad,
                                                      cost_hess_xx);
}

#define HORIZON 100
#define dt      1e-2

using namespace ssqp::model::stagewise::predefined;

int main()
{
    ssqp::model::Model::Structure structure;
    structure.urdf_path = "examples/urdf/double_pendulum/double_pendulum.urdf";
    structure.ignore_innate_constraints = true;
    //structure.integration     = ssqp::model::IntegrationScheme::ExplicitEuler;
    structure.actuated_joints = { 1 };
    ssqp::model::Model model(HORIZON, dt, structure);

    for (uint k = 0; k < HORIZON; ++k)
    {
        model.addCost(k, getCost());
        model.addCost<costs::SquaredControlCost>(k, 2, 1e-5);
    }

    model.addCost(HORIZON, getCost(100.0));

    ssqp::model::Model::Config config;
    config.init_states = {
        (Eigen::Vector4d() << 3.14, 0.0, 0.0, 0.0).finished()
    };
    model.finalize(config);

    ssqp::solvers::ssqp::Params params;
    params.max_iter  = 100;
    params.tolerance = 1e-4;

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

    solver.write("examples/trajgen/data/double_pendulum");

    return 0;
}