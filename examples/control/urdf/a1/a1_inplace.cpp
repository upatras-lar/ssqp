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
#define dt           0.01
// Position weight
#define P_WGT 10
// Orientation weight
#define O_WGT 1
// State regularization weight
#define A_WGT 1e-5
// Velocity regularization weight
#define V_WGT 1e-3
// Control regularization weight
#define U_WGT 1e-5
// Force regularization weight
#define F_WGT 1e-5

#define STANDING_STEPS 10
#define INPLACE_STEPS  12
#define REPEATS        2

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

        model.addConstraints<constraints::ContactConstraint>(k,
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

        init_controls.push_back(init_control_4);
    }
}

int main()
{
    uint HORIZON = 2 * STANDING_STEPS + 4 * REPEATS * INPLACE_STEPS;

    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;
    //model_structure.integration = ssqp::model::IntegrationScheme::ExplicitEuler;
    ssqp::model::Model model(HORIZON, dt, model_structure);

    // Initial configuration
    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    init.segment(7, 12) << 0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3,
      0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3;

    // Initial controls
    std::vector<Eigen::VectorXd> init_controls;
    Eigen::VectorXd init_control_4, init_control_21, init_control_22;

    init_control_4     = Eigen::VectorXd::Zero(30);
    init_control_4[20] = 3.43525 * 9.81;
    init_control_4[23] = 3.43525 * 9.81;
    init_control_4[26] = 3.43525 * 9.81;
    init_control_4[29] = 3.43525 * 9.81;

    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> forces;
    forces["FL_foot"].push_back(init_control_4.segment(18, 3));
    forces["FR_foot"].push_back(init_control_4.segment(21, 3));
    forces["RL_foot"].push_back(init_control_4.segment(24, 3));
    forces["RR_foot"].push_back(init_control_4.segment(27, 3));

    init_control_4.segment(6, 12) =
      model
        .inverseDynamics(init.head(STATE_SIZE),
                         init.tail(CONTROL_SIZE),
                         Eigen::VectorXd::Zero(CONTROL_SIZE),
                         forces)
        .tail(12);

    init_control_21     = Eigen::VectorXd::Zero(24);
    init_control_21[20] = 6.8705 * 9.81;
    init_control_21[23] = 6.8705 * 9.81;

    forces.clear();
    forces["FL_foot"].push_back(init_control_21.segment(18, 3));
    forces["RR_foot"].push_back(init_control_21.segment(21, 3));

    init_control_21.segment(6, 12) =
      model
        .inverseDynamics(init.head(STATE_SIZE),
                         init.tail(CONTROL_SIZE),
                         Eigen::VectorXd::Zero(CONTROL_SIZE),
                         forces)
        .tail(12);

    init_control_22     = Eigen::VectorXd::Zero(24);
    init_control_22[20] = 6.8705 * 9.81;
    init_control_22[23] = 6.8705 * 9.81;

    forces.clear();
    forces["FR_foot"].push_back(init_control_22.segment(18, 3));
    forces["RL_foot"].push_back(init_control_22.segment(21, 3));

    init_control_22.segment(6, 12) =
      model
        .inverseDynamics(init.head(STATE_SIZE),
                         init.tail(CONTROL_SIZE),
                         Eigen::VectorXd::Zero(CONTROL_SIZE),
                         forces)
        .tail(12);

    addStandingPhase(STANDING_STEPS, init_control_4, model, init_controls);

    uint current_step = STANDING_STEPS;

    for (uint i = 0; i < REPEATS; ++i)
    {
        for (uint k = current_step; k < current_step + INPLACE_STEPS; ++k)
        {
            model.addConstraints<constraints::ContactConstraint>(k,
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

            init_controls.push_back(init_control_21);
        }

        current_step += INPLACE_STEPS;

        addStandingPhase(INPLACE_STEPS,
                         init_control_4,
                         model,
                         init_controls,
                         current_step);

        current_step += INPLACE_STEPS;

        for (uint k = current_step; k < current_step + INPLACE_STEPS; ++k)
        {
            model.addConstraints<constraints::ContactConstraint>(k,
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

            init_controls.push_back(init_control_22);
        }

        current_step += INPLACE_STEPS;

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

    for (uint k = 0; k <= HORIZON; ++k)
    {
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          base_pos,
          Eigen::Matrix3d::Identity(),
          "trunk",
          STATE_SIZE,
          CONTROL_SIZE,
          P_WGT,
          O_WGT);

        model.addCost<costs::SquaredJointAngleCost>(k,
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    A_WGT,
                                                    true);
        model.addCost<costs::SquaredJointVelocityCost>(k,
                                                       CONTROL_SIZE,
                                                       V_WGT,
                                                       true);

        if (k != HORIZON)
        {
            model.addCost<costs::SquaredControlCost>(k, CONTROL_SIZE, U_WGT);
            model.addCost<costs::SquaredForceCost>(k, CONTROL_SIZE, F_WGT);
        }
    }

    ssqp::model::Model::Config model_config;
    model_config.init_states   = { init };
    model_config.init_controls = init_controls;

    model.finalize(model_config);

    ssqp::solvers::ssqp::Params params;
    params.tolerance  = 1e-2;
    params.max_iter   = 500;
    params.use_filter = true;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-3;
    admm_params.max_iter      = 2500;
    admm_params.warm_start    = true;

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

    solver.write("examples/data/a1_inplace");

    return 0;
}