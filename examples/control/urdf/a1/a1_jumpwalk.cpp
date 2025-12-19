#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#define STATE_SIZE   19
#define CONTROL_SIZE 18
#define HORIZON      200
#define dt           0.01
// Angle regularization weight
#define A_WGT 0
// Velocity regularization weight
#define V_WGT 1e-4
// Control regularization weight
#define U_WGT 1e-6
// Force regularization weight
#define F_WGT 1e-7
// Terminal position weight
#define TP_WEIGHT 10
// Terminal orientation weight
#define TO_WEIGHT 8

// Jump
#define STEPS_BEFORE 30
#define STEPS_AFTER  30
#define REPEATS      2

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

    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> forces;
    forces["FL_foot"].push_back(init_control.segment(18, 3));
    forces["FR_foot"].push_back(init_control.segment(21, 3));
    forces["RL_foot"].push_back(init_control.segment(24, 3));
    forces["RR_foot"].push_back(init_control.segment(27, 3));

    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    // model_structure.dynamics_mode = ssqp::model::DynamicsMode::Inverse;
    ssqp::model::Model model(HORIZON, dt, model_structure);

    init_control.segment(6, 12) =
      model
        .inverseDynamics(init.head(STATE_SIZE),
                         init.tail(CONTROL_SIZE),
                         Eigen::VectorXd::Zero(CONTROL_SIZE),
                         forces)
        .tail(12);

    std::vector<Eigen::VectorXd> init_controls;

    Eigen::Vector3d target(1.6, 0.0, base_pos[2]);

    uint JUMP_STAGES = HORIZON / REPEATS;

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredJointVelocityCost>(k,
                                                       CONTROL_SIZE,
                                                       V_WGT,
                                                       true);
        model.addCost<costs::SquaredControlCost>(k, CONTROL_SIZE, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, CONTROL_SIZE, F_WGT);

        uint rel_k = k % JUMP_STAGES;

        if (rel_k == 0 && k > 0)
        {
            Eigen::Vector3d lb(0.1,
                               -std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()),
              ub(std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity(),
                 std::numeric_limits<double>::infinity());

            model.addConstraints<constraints::FramePositionBound>(k,
                                                                  model,
                                                                  "trunk",
                                                                  lb,
                                                                  ub,
                                                                  STATE_SIZE,
                                                                  CONTROL_SIZE);
        }

        if (rel_k < STEPS_BEFORE || rel_k > JUMP_STAGES - STEPS_AFTER)
        {

            if (k < STEPS_BEFORE || k > HORIZON - STEPS_AFTER)
            {
                init_controls.push_back(init_control);
                model.addConstraints<constraints::ContactConstraint>(
                  k,
                  model,
                  "FL_foot",
                  STATE_SIZE,
                  CONTROL_SIZE,
                  0.6);

                model.addConstraints<constraints::FrameVelocityBound>(
                  k,
                  model,
                  "FL_foot",
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d::Zero(),
                  STATE_SIZE,
                  CONTROL_SIZE);

                model.addConstraints<constraints::ContactConstraint>(
                  k,
                  model,
                  "FR_foot",
                  STATE_SIZE,
                  CONTROL_SIZE,
                  0.6);

                model.addConstraints<constraints::FrameVelocityBound>(
                  k,
                  model,
                  "FR_foot",
                  Eigen::Vector3d::Zero(),
                  Eigen::Vector3d::Zero(),
                  STATE_SIZE,
                  CONTROL_SIZE);
            }
            else
            {
                init_controls.push_back(init_control_2);
            }

            model.addConstraints<constraints::ContactConstraint>(k,
                                                                 model,
                                                                 "RL_foot",
                                                                 STATE_SIZE,
                                                                 CONTROL_SIZE,
                                                                 0.6);

            model.addConstraints<constraints::FrameVelocityBound>(
              k,
              model,
              "RL_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(),
              STATE_SIZE,
              CONTROL_SIZE);

            model.addConstraints<constraints::ContactConstraint>(k,
                                                                 model,
                                                                 "RR_foot",
                                                                 STATE_SIZE,
                                                                 CONTROL_SIZE,
                                                                 0.6);

            model.addConstraints<constraints::FrameVelocityBound>(
              k,
              model,
              "RR_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(),
              STATE_SIZE,
              CONTROL_SIZE);
        }
        else
        {
            init_controls.push_back(Eigen::VectorXd::Zero(18));
        }
    }

    model.addCost<costs::SquaredJointVelocityCost>(HORIZON,
                                                   CONTROL_SIZE,
                                                   V_WGT,
                                                   true);

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    target,
                                                    Eigen::Matrix3d::Identity(),
                                                    "trunk",
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    TP_WEIGHT,
                                                    TO_WEIGHT);

    model.addConstraints<constraints::FrameVelocityBound>(
      HORIZON,
      model,
      "trunk",
      -1e-3 * Eigen::Vector3d::Ones(),
      1e-3 * Eigen::Vector3d::Ones(),
      STATE_SIZE,
      CONTROL_SIZE,
      true);

    ssqp::model::Model::Config model_config;
    model_config.init_states   = { init };
    model_config.init_controls = init_controls;
    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    params.loose_success = false;
    params.tolerance     = 2e-2;
    params.use_filter    = true;
    params.filter_size   = 100;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-4;
    admm_params.max_iter      = 500;

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

    solver.write("examples/data/a1_jumpwalk");

    return 0;
}