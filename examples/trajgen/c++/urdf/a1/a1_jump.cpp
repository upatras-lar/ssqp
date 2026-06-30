#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"    // ssqp::solvers::ssqp
#include "solvers/admm_qp.hpp" // ssqp::solvers::admm_qp::Params
#include "model/Model.hpp"     // ssqp::model::Model

// ssqp::stagewise::predefined::costs namespace
#include "model/stagewise/predefined/Costs.hpp"

// ssqp::stagewise::predefined::constraints namespace
#include "model/stagewise/predefined/Constraints.hpp"

#define STATE_SIZE    19
#define VELOCITY_SIZE 18
#define CONTROL_SIZE  18
#define HORIZON       325
#define dt            0.01
// Angle regularization weight
#define A_WGT 1e-3
// Velocity regularization weight
#define V_WGT 1e-2
// Control regularization weight
#define U_WGT 1e-3
// Force regularization weight
#define F_WGT 1e-4
// Terminal position weight
#define TF_WGT 3.0
// Terminal orientation weight
#define RF_WGT 2.8

// Jump
#define STEPS_BEFORE 40
#define STEPS_AFTER  45
#define REPEATS      3

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Trunk position
    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 0.3464;

    // Initial configuration
    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    init.segment(7, 12) << 0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3,
      0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3;

    Eigen::VectorXd init_control = Eigen::VectorXd::Zero(30);
    init_control[20]             = 3.43525 * 9.81;
    init_control[23]             = 3.43525 * 9.81;
    init_control[26]             = 3.43525 * 9.81;
    init_control[29]             = 3.43525 * 9.81;

    Eigen::VectorXd init_control_2 = Eigen::VectorXd::Zero(24);
    init_control_2[20]             = 6.8705 * 9.81;
    init_control_2[23]             = 6.8705 * 9.81;

    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    ssqp::model::Model model(HORIZON, dt, model_structure);

    std::vector<Eigen::VectorXd> init_controls;

    uint JUMP_STAGES = HORIZON / REPEATS;

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredJointVelocityCost>(k, model, V_WGT);
        model.addCost<costs::SquaredControlCost>(k, model, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, model, F_WGT);

        // Get stage index relative to jumps
        uint rel_k = k % JUMP_STAGES;

        // If we are on the ground
        if (rel_k < STEPS_BEFORE || rel_k > JUMP_STAGES - STEPS_AFTER)
        {
            // If we are at the start or end (all feet on the ground)
            if (k < STEPS_BEFORE || k > HORIZON - STEPS_AFTER)
            {
                // Add constraints for front feet to touch the ground
                init_controls.push_back(init_control);
                model.addConstraints<constraints::ContactConstraint>(k,
                                                                     model,
                                                                     "FL_foot",
                                                                     0.6);

                model.addConstraints<constraints::FrameLinearVelocityBound>(
                  k,
                  model,
                  "FL_foot",
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d::Zero());

                model.addConstraints<constraints::ContactConstraint>(k,
                                                                     model,
                                                                     "FR_foot",
                                                                     0.6);

                model.addConstraints<constraints::FrameLinearVelocityBound>(
                  k,
                  model,
                  "FR_foot",
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d::Zero());
            }
            // If we are not at the start or end, only back feet should touch
            // the ground
            else
            {
                init_controls.push_back(init_control_2);
            }

            // Add constraints for back feet to touch the ground
            model.addConstraints<constraints::ContactConstraint>(k,
                                                                 model,
                                                                 "RL_foot",
                                                                 0.6);

            model.addConstraints<constraints::FrameLinearVelocityBound>(
              k,
              model,
              "RL_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero());

            model.addConstraints<constraints::ContactConstraint>(k,
                                                                 model,
                                                                 "RR_foot",
                                                                 0.6);

            model.addConstraints<constraints::FrameLinearVelocityBound>(
              k,
              model,
              "RR_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero());
        }
        // If we are on air, do not add constraints
        else
        {
            init_controls.push_back(Eigen::VectorXd::Zero(18));
        }
    }

    // Add regularization costs for the terminal stage
    model.addCost<costs::SquaredJointVelocityCost>(HORIZON, model, V_WGT);

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    "trunk",
                                                    base_pos,
                                                    Eigen::Matrix3d::Identity(),
                                                    TF_WGT,
                                                    RF_WGT);

    // Add bound to keep final trunk velocity low
    model.addConstraints<constraints::FrameLinearVelocityBound>(
      HORIZON,
      model,
      "trunk",
      -1e-3 * Eigen::Vector3d::Ones(),
      1e-3 * Eigen::Vector3d::Ones());

    ssqp::model::Model::Config model_config;
    model_config.init_states   = { init };
    model_config.init_controls = init_controls;
    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance   = 1e-2;
    params.filter_size = 100;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-3;
    admm_params.max_iter      = 750;

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

    solver.write("examples/trajgen/data/a1_jump.json");

    return 0;
}