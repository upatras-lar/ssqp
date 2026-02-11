#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#define STATE_SIZE   51
#define CONTROL_SIZE 50
#define HORIZON      20
#define dt           0.025
// Trunk position weight
#define T_WGT 3.0
// Orientation weight
#define R_WGT 3.0
// Velocity regularization weight
#define V_WGT 1e-1
// Control regularization weight
#define U_WGT 3e-3
// Force regularization weight
#define F_WGT 1e-4

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/talos/talos.urdf";
    model_structure.free_flyer = true;
    // model_structure.dynamics_mode = ssqp::model::DynamicsMode::Inverse;
    ssqp::model::Model model(HORIZON, dt, model_structure);
    // model.printFramesAndJoints();

    Eigen::VectorXd neutral = model.neutralPose();

    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 1.072;

    Eigen::VectorXd init = Eigen::VectorXd::Zero(STATE_SIZE + CONTROL_SIZE);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    // std::cout << model.forwardKinematics(init.head(STATE_SIZE), "right_mm")
    //                .first.transpose()
    //           << std::endl;

    for (uint k = 0; k < HORIZON; ++k)
    {
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          base_pos,
          Eigen::Matrix3d::Identity(),
          "torso_1_link",
          STATE_SIZE,
          CONTROL_SIZE,
          T_WGT,
          R_WGT);

        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredJointVelocityCost>(k,
                                                       CONTROL_SIZE,
                                                       V_WGT,
                                                       true);
        model.addCost<costs::SquaredControlCost>(k, CONTROL_SIZE, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, CONTROL_SIZE, F_WGT);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "left_mm",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "left_mm",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "left_mp",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "left_mp",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "left_pp",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "left_pp",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "left_pm",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "left_pm",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "right_mm",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "right_mm",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "right_mp",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "right_mp",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "right_pp",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "right_pp",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             "right_pm",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          "right_pm",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);
    }

    model.addCost<costs::SquaredJointVelocityCost>(HORIZON,
                                                   CONTROL_SIZE,
                                                   V_WGT,
                                                   true);

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    base_pos,
                                                    Eigen::Matrix3d::Identity(),
                                                    "torso_1_link",
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    T_WGT,
                                                    R_WGT);

    ssqp::model::Model::Config model_config;
    model_config.init_states = { init };

    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 1e-1;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-2;
    admm_params.rel_tolerance = 1e-3;
    admm_params.max_iter      = 1000;

    // params.loose_success_tolerance = 1e-3;
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

    solver.write("examples/trajgen/data/talos_standing");

    return 0;
}