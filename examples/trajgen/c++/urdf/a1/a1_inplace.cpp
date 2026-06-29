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
#define dt            0.02
// Position weight
#define T_WGT 3.0
// Orientation weight
#define R_WGT 2.8
// State regularization weight
#define A_WGT 1e-3
// Velocity regularization weight
#define V_WGT 1e-2
// Control regularization weight
#define U_WGT 3e-3
// Force regularization weight
#define F_WGT 1e-4

// How many steps to stand for (all feet on the ground)
#define STANDING_STEPS 15
// Total steps for one cycle
#define INPLACE_STEPS 40
// How many times to repeat motion
#define REPEATS 3

using namespace ssqp::model::stagewise::predefined;

// Trunk position
Eigen::Vector3d base_pos(0, 0, 0.3464);

void addStandingPhase(const uint steps, const Eigen::VectorXd &init_control_4,
                      ssqp::model::Model           &model,
                      std::vector<Eigen::VectorXd> &init_controls,
                      const uint                    step_offset = 0)
{
    for (uint k = step_offset; k < steps + step_offset; ++k)
    {
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

        init_controls.push_back(init_control_4);
    }
}

// A1 raising feet in place example
int main()
{
    uint HORIZON = 2 * STANDING_STEPS + 4 * REPEATS * INPLACE_STEPS;

    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    ssqp::model::Model model(HORIZON, dt, model_structure);

    // Initial configuration
    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    init.segment(7, 12) << 0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3,
      0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3;

    // Initial controls
    std::vector<Eigen::VectorXd> init_controls;
    Eigen::VectorXd              init_control_4, init_control_2;

    init_control_4     = Eigen::VectorXd::Zero(30);
    init_control_4[20] = 3.43525 * 9.81;
    init_control_4[23] = 3.43525 * 9.81;
    init_control_4[26] = 3.43525 * 9.81;
    init_control_4[29] = 3.43525 * 9.81;

    init_control_2     = Eigen::VectorXd::Zero(24);
    init_control_2[20] = 6.8705 * 9.81;
    init_control_2[23] = 6.8705 * 9.81;

    addStandingPhase(STANDING_STEPS, init_control_4, model, init_controls);

    uint current_step = STANDING_STEPS;

    Eigen::Vector3d l(-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      0.05);
    Eigen::Vector3d u(std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity());

    for (uint i = 0; i < REPEATS; ++i)
    {
        for (uint k = current_step; k < current_step + INPLACE_STEPS; ++k)
        {
            // Place front left & rear right feet on the ground

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
                                                                 "RR_foot",
                                                                 0.6);

            model.addConstraints<constraints::FrameLinearVelocityBound>(
              k,
              model,
              "RR_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero());

            // If at the right part of cycle (first or last fourth), place
            // remaining feet on the ground
            if (k > current_step + INPLACE_STEPS / 4 &&
                k < current_step + 3 * INPLACE_STEPS / 4)
            {
                model.addConstraints<constraints::FrameTranslationBound>(
                  k,
                  model,
                  "FR_foot",
                  l,
                  u);

                model.addConstraints<constraints::FrameTranslationBound>(
                  k,
                  model,
                  "RL_foot",
                  l,
                  u);
            }

            init_controls.push_back(init_control_2);
        }

        current_step += INPLACE_STEPS;

        // Add standing phase in the middle
        addStandingPhase(INPLACE_STEPS,
                         init_control_4,
                         model,
                         init_controls,
                         current_step);

        current_step += INPLACE_STEPS;

        for (uint k = current_step; k < current_step + INPLACE_STEPS; ++k)
        {
            // Place front right & rear left feet on the ground

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

            // If at the right part of cycle (first or last fourth), place
            // remaining feet on the ground
            if (k > current_step + INPLACE_STEPS / 4 &&
                k < current_step + 3 * INPLACE_STEPS / 4)
            {
                model.addConstraints<constraints::FrameTranslationBound>(
                  k,
                  model,
                  "FL_foot",
                  l,
                  u);

                model.addConstraints<constraints::FrameTranslationBound>(
                  k,
                  model,
                  "RR_foot",
                  l,
                  u);
            }

            init_controls.push_back(init_control_2);
        }

        current_step += INPLACE_STEPS;

        // Add another standing phase
        addStandingPhase(INPLACE_STEPS,
                         init_control_4,
                         model,
                         init_controls,
                         current_step);

        current_step += INPLACE_STEPS;
    }

    addStandingPhase(STANDING_STEPS,
                     init_control_4,
                     model,
                     init_controls,
                     current_step);

    // Add costs for each stage, including horizon
    for (uint k = 0; k <= HORIZON; ++k)
    {
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          "trunk",
          base_pos,
          Eigen::Matrix3d::Identity(),
          T_WGT,
          R_WGT);

        model.addCost<costs::SquaredJointAngleCost>(k, model, A_WGT);
        model.addCost<costs::SquaredJointVelocityCost>(k, model, V_WGT);

        if (k != HORIZON)
        {
            model.addCost<costs::SquaredControlCost>(k, model, U_WGT);
            model.addCost<costs::SquaredForceCost>(k, model, F_WGT);
        }
    }

    ssqp::model::Model::Config model_config;
    model_config.init_states   = { init };
    model_config.init_controls = init_controls;

    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 1e-2;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-3;
    admm_params.max_iter      = 500;

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

    solver.write("examples/trajgen/data/a1_inplace.json");

    return 0;
}