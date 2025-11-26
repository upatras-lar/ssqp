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
#define HORIZON      50
#define dt           0.01
// Trunk position weight
#define P_WGT 1000
// Orientation weight
#define R_WGT 100
// Velocity regularization weight
#define V_WGT 5e-4
// Control regularization weight
#define U_WGT 1e-5
// Force regularization weight
#define F_WGT 1e-5
// Whether we only use two feet to stand on
#define TWO_FEET true

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

    // Initial controls
    Eigen::VectorXd init_control = Eigen::VectorXd::Zero(24);

    // Initialize model
    ssqp::model::Model model("examples/urdf/a1/a1.urdf", HORIZON, dt, true);

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add cost for trunk to stay in base position
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          base_pos,
          Eigen::Matrix3d::Identity(),
          "trunk",
          STATE_SIZE,
          CONTROL_SIZE,
          P_WGT,
          R_WGT);

        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredVelocityCost>(k, CONTROL_SIZE, V_WGT);
        model.addCost<costs::SquaredControlCost>(k, CONTROL_SIZE, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, CONTROL_SIZE, F_WGT);

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

        if (!TWO_FEET)
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
        }

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

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    base_pos,
                                                    Eigen::Matrix3d::Identity(),
                                                    "trunk",
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    P_WGT,
                                                    R_WGT);

    // model.addConstraints<constraints::FramePositionBound>(HORIZON,
    //                                                       model,
    //                                                       "trunk",
    //                                                       base_pos,
    //                                                       base_pos,
    //                                                       STATE_SIZE,
    //                                                       CONTROL_SIZE);

    model.finalize({}, { init }, { init_control });

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 1e-1;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 6e-3;
    admm_params.rel_tolerance = 1e-3;

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

    /*
    std::cout << "Feet z-coordinate over horizon:" << std::endl;
    for (uint k = 0; k < HORIZON; ++k)
    {
        std::cout << "Stage " << k << std::endl;
        Eigen::Vector3d pos, vel;
        std::tie(pos, std::ignore) =
          model.forwardKinematics(solution.x.stage(k).head(STATE_SIZE),
                                  "FL_foot");
        vel = model.frameVelocity(solution.x.stage(k).head(STATE_SIZE),
                                  solution.x.stage(k).tail(CONTROL_SIZE),
                                  "FL_foot",
                                  true);
        std::cout << "FL_foot: " << std::endl;
        std::cout << pos.transpose() << " " << vel.transpose() << std::endl;

        std::tie(pos, std::ignore) =
          model.forwardKinematics(solution.x.stage(k).head(STATE_SIZE),
                                  "FR_foot");
        vel = model.frameVelocity(solution.x.stage(k).head(STATE_SIZE),
                                  solution.x.stage(k).tail(CONTROL_SIZE),
                                  "FR_foot",
                                  true);
        std::cout << "FR_foot: " << std::endl;
        std::cout << pos.transpose() << " " << vel.transpose() << std::endl;

        std::tie(pos, std::ignore) =
          model.forwardKinematics(solution.x.stage(k).head(STATE_SIZE),
                                  "RL_foot");
        vel = model.frameVelocity(solution.x.stage(k).head(STATE_SIZE),
                                  solution.x.stage(k).tail(CONTROL_SIZE),
                                  "RL_foot",
                                  true);
        std::cout << "RL_foot: " << std::endl;
        std::cout << pos.transpose() << " " << vel.transpose() << std::endl;

        std::tie(pos, std::ignore) =
          model.forwardKinematics(solution.x.stage(k).head(STATE_SIZE),
                                  "RR_foot");
        vel = model.frameVelocity(solution.x.stage(k).head(STATE_SIZE),
                                  solution.x.stage(k).tail(CONTROL_SIZE),
                                  "RR_foot",
                                  true);
        std::cout << "RR_foot: " << std::endl;
        std::cout << pos.transpose() << " " << vel.transpose() << std::endl;
    }
    */

    solver.write("examples/data/a1_standing");

    return 0;
}