#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp
#include "model/Model.hpp"  // ssqp::model::Model

// ssqp::stagewise::predefined::costs namespace
#include "model/stagewise/predefined/Costs.hpp"

// ssqp::stagewise::predefined::constraints namespace
#include "model/stagewise/predefined/Constraints.hpp"

#define STATE_SIZE    19
#define VELOCITY_SIZE 18
#define CONTROL_SIZE  18
#define HORIZON       400
#define dt            0.01
// Trunk position weight
#define TF_WGT 3.0
// Orientation weight
#define RF_WGT 2.8
// Velocity regularization weight
#define V_WGT 1e-2
// Control regularization weight
#define U_WGT 3e-3
// Force regularization weight
#define F_WGT 1e-4
// Whether we only use two feet to stand on
// @script-parameter
#define TWO_FEET true

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.from_preset = ssqp::model::Preset::A1;

    ssqp::model::Model model(HORIZON, dt, model_structure);

    // Trunk position
    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 0.3464;

    // Initial configuration
    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    init.segment(7, 12) << 0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3,
      0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3;

    // Initial controls
    Eigen::VectorXd init_control;

    if (TWO_FEET)
    {
        // If only standing on two feet, initial controls for forces should be
        // twice as large
        init_control     = Eigen::VectorXd::Zero(24);
        init_control[20] = 6.8705 * 9.81;
        init_control[23] = 6.8705 * 9.81;

        std::unordered_map<std::string, std::vector<Eigen::Vector3d>> forces;
        forces["FL_foot"].push_back(init_control.segment(18, 3));
        forces["RR_foot"].push_back(init_control.segment(21, 3));

        init_control.segment(6, 12) =
          model
            .inverseDynamics(init.head(STATE_SIZE),
                             init.tail(VELOCITY_SIZE),
                             Eigen::VectorXd::Zero(VELOCITY_SIZE),
                             forces)
            .tail(12);
    }
    else
    {
        // For 4 feet, we simply pass 0 (no need for initialization)
        init_control = Eigen::VectorXd::Zero(30);
    }

    Eigen::Vector3d l(-std::numeric_limits<double>::infinity(),
                      -std::numeric_limits<double>::infinity(),
                      0.001);
    Eigen::Vector3d u(std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::infinity());

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredJointVelocityCost>(k, model, V_WGT);
        model.addCost<costs::SquaredControlCost>(k, model, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, model, F_WGT);

        // Add contact constraints for front left and rear right feet
        // We also add velocity bounds to each contact constraint frame to
        // constrain its velocity to 0, since we do not want it slipping

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

        if (!TWO_FEET)
        {
            // If standing on four feet, add contact constraints and velocity
            // bounds for the other two feet as well

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
        }
        else
        {
            // If the remaining two feet are on the air, add position bounds
            // to keep them above ground

            model.addConstraints<constraints::FrameTranslationBound>(k,
                                                                     model,
                                                                     "FR_foot",
                                                                     l,
                                                                     u);

            model.addConstraints<constraints::FrameTranslationBound>(k,
                                                                     model,
                                                                     "RL_foot",
                                                                     l,
                                                                     u);
        }
    }

    // At the terminal stage, add a cost to keep the trunk at the initial position
    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    "trunk",
                                                    base_pos,
                                                    Eigen::Matrix3d::Identity(),
                                                    TF_WGT,
                                                    RF_WGT);

    ssqp::model::Model::Config model_config;
    model_config.init_states   = { init };
    model_config.init_controls = { init_control };

    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 1e-2;

    ssqp::solvers::ssqp        solver(model, params);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    solver.write("examples/trajgen/data/a1_standing.json");

    return 0;
}