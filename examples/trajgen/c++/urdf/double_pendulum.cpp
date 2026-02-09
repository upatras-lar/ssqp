#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp
#include "model/Model.hpp"  // ssqp::model::Model

// ssqp::model::stagewise::predefined::costs namespace
#include "model/stagewise/predefined/Costs.hpp"

/* Create a custom StageCost object using lambda functions
 */
std::shared_ptr<ssqp::model::stagewise::StageCost> getCost(
  const double weight = 1)
{
    // Evaluation function
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

    // Gradient function
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

    // Hessian xx function
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

    // Use the make_cost helper to create the custom stage cost
    // By leaving hessian_xu and hessian_uu empty, they are set to zero
    return ssqp::model::stagewise::factory::make_cost(cost_eval,
                                                      cost_grad,
                                                      cost_hess_xx);
}

#define HORIZON 100
#define dt      1e-2

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Initialize model structure
    ssqp::model::Model::Structure structure;
    // Set the URDF file path
    structure.urdf_path =
      "examples/assets/urdf/double_pendulum/double_pendulum.urdf";
    // Ignore constraints outlined in URDF
    structure.ignore_innate_constraints = true;
    // Allow only joint 1 (second joint) to be actuated
    structure.actuated_joints = { 1 };

    // Initialize model using the structure
    // Note: Model will be unfinalized and not yet ready for use in SSQP
    ssqp::model::Model model(HORIZON, dt, structure);

    // Add costs to each stage
    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add cost using the addCost method for stage k
        // For the cost, we pass the custom StageCost object created by getCost
        model.addCost(k, getCost());
        // Also add a predefined cost of type SquaredControlCost to stage k
        // with arguments nv = 2 and weight = 1e-5
        model.addCost<costs::SquaredControlCost>(k, 2, 1e-5);
    }

    // Add terminal cost with larger weight
    model.addCost(HORIZON, getCost(100.0));

    // Create a model config to pass an initial state
    ssqp::model::Model::Config config;
    // init_states vector of size 1: Same initial state for every stage
    // This will set the stage 0 state to the given state (which cannot be
    // changed) and provide it as an initial guess for other stages
    config.init_states = {
        (Eigen::Vector4d() << 3.14, 0.0, 0.0, 0.0).finished()
    };

    // Finalize the model with the created configuration
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

    // Write the output to a JSON file
    solver.write("examples/trajgen/data/double_pendulum.json");

    return 0;
}