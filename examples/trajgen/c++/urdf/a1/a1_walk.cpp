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

#define STATE_SIZE   19
#define CONTROL_SIZE 18
#define HORIZON      400
#define dt           0.01
// Velocity regularization weight
#define V_WGT 1e-2
// Control regularization weight
#define U_WGT 1e-3
// Force regularization weight
#define F_WGT 1e-4
// Position weight
#define T_WGT 1.0

#define AIR_STEPS    20
#define GROUND_STEPS 10

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Initial trunk position
    Eigen::Vector3d base_pos;
    base_pos << 0, 0, 0.3464;

    // Initial configuration
    Eigen::VectorXd init = Eigen::VectorXd::Zero(37);
    init.segment(0, 3)   = base_pos;
    init[6]              = 1.0;

    init.segment(7, 12) << 0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3,
      0.0, M_PI / 6, -M_PI / 3, 0.0, M_PI / 6, -M_PI / 3;

    // Initial control vector for 4 feet touching the ground
    // Each force is the total mass / 4, times g
    Eigen::VectorXd init_control = Eigen::VectorXd::Zero(30);
    init_control[20]             = 3.43525 * 9.81;
    init_control[23]             = 3.43525 * 9.81;
    init_control[26]             = 3.43525 * 9.81;
    init_control[29]             = 3.43525 * 9.81;

    // Equivalently, initial control vector for 2 feet touching the ground
    Eigen::VectorXd init_control_2 = Eigen::VectorXd::Zero(24);
    init_control_2[20]             = 6.8705 * 9.81;
    init_control_2[23]             = 6.8705 * 9.81;

    // Map mapping frame names to their force equivalents
    // Used for performing inverse dynamics, to gain initial torques
    std::unordered_map<std::string, std::vector<Eigen::Vector3d>> forces;
    forces["FL_foot"].push_back(init_control.segment(18, 3));
    forces["FR_foot"].push_back(init_control.segment(21, 3));
    forces["RL_foot"].push_back(init_control.segment(24, 3));
    forces["RR_foot"].push_back(init_control.segment(27, 3));

    // Initialize model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path  = "examples/assets/urdf/a1/a1.urdf";
    model_structure.free_flyer = true;

    // Select inverse dynamics mode
    model_structure.dynamics_mode = ssqp::model::DynamicsMode::Inverse;

    ssqp::model::Model model(HORIZON, dt, model_structure);

    // With the model created, we can use the inverseDynamics method to
    // calculate the initial torques, given the robot model and the initial
    // forces we apply in the contact frames
    init_control.segment(6, 12) =
      model
        .inverseDynamics(init.head(STATE_SIZE),
                         init.tail(CONTROL_SIZE),
                         Eigen::VectorXd::Zero(CONTROL_SIZE),
                         forces)
        .tail(12);

    std::vector<Eigen::VectorXd> init_controls;

    // Create a vector for desired positions per stage
    std::vector<Eigen::Vector3d> traj(HORIZON + 1);
    double                       target = 2.5;

    for (uint i = 0; i <= HORIZON; ++i)
    {
        // For each stage k, move an equal amount closer to the target
        traj[i] = (Eigen::Vector3d() << i * (target - base_pos[0]) / HORIZON,
                   0.0,
                   base_pos[2])
                    .finished();
    }

    uint MOVE_STEPS = 2 * GROUND_STEPS + AIR_STEPS;
    bool ls1        = false;

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add regularization cost for velocity, controls and forces
        model.addCost<costs::SquaredJointVelocityCost>(k,
                                                       CONTROL_SIZE,
                                                       V_WGT,
                                                       true);
        model.addCost<costs::SquaredControlCost>(k, CONTROL_SIZE, U_WGT);
        model.addCost<costs::SquaredForceCost>(k, CONTROL_SIZE, F_WGT);

        // Add cost for trunk to follow the desired trajectory
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          "trunk",
          traj[k],
          Eigen::Matrix3d::Identity(),
          STATE_SIZE,
          CONTROL_SIZE,
          T_WGT,
          T_WGT);

        uint rel_k = k % MOVE_STEPS;

        if (rel_k < GROUND_STEPS || rel_k > GROUND_STEPS + AIR_STEPS)
        {
            // If in ground part, add contact constraints for all feet
            init_controls.push_back(init_control);
            model.addConstraints<constraints::ContactConstraint>(
              k,
              model,
              ls1 ? "RR_foot" : "RL_foot",
              STATE_SIZE,
              CONTROL_SIZE,
              0.6);

            model.addConstraints<constraints::FrameLinearVelocityBound>(
              k,
              model,
              ls1 ? "RR_foot" : "RL_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(),
              STATE_SIZE,
              CONTROL_SIZE);

            model.addConstraints<constraints::ContactConstraint>(
              k,
              model,
              ls1 ? "FR_foot" : "FL_foot",
              STATE_SIZE,
              CONTROL_SIZE,
              0.6);

            model.addConstraints<constraints::FrameLinearVelocityBound>(
              k,
              model,
              ls1 ? "FR_foot" : "FL_foot",
              Eigen::Vector3d::Zero(),
              Eigen::Vector3d::Zero(),
              STATE_SIZE,
              CONTROL_SIZE);
        }
        else
        {
            // Else, skip adding for certain two
            init_controls.push_back(init_control_2);
        }

        // Toggle which two feet move next
        if (rel_k == 0)
        {
            ls1 = !ls1;
        }

        // Add contact constraints for remaining two feet regardless
        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             ls1 ? "FR_foot"
                                                                 : "FL_foot",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          ls1 ? "FR_foot" : "FL_foot",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);

        model.addConstraints<constraints::ContactConstraint>(k,
                                                             model,
                                                             ls1 ? "RL_foot"
                                                                 : "RR_foot",
                                                             STATE_SIZE,
                                                             CONTROL_SIZE,
                                                             0.6);

        model.addConstraints<constraints::FrameLinearVelocityBound>(
          k,
          model,
          ls1 ? "RL_foot" : "RR_foot",
          Eigen::Vector3d::Zero(),
          Eigen::Vector3d::Zero(),
          STATE_SIZE,
          CONTROL_SIZE);
    }

    // Add terminal costs
    model.addCost<costs::SquaredJointVelocityCost>(HORIZON,
                                                   CONTROL_SIZE,
                                                   V_WGT,
                                                   true);

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    "trunk",
                                                    traj[HORIZON],
                                                    Eigen::Matrix3d::Identity(),
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    T_WGT,
                                                    T_WGT);

    model.addConstraints<constraints::FrameLinearVelocityBound>(
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
    params.tolerance = 2e-2;
    // Edit filter size (line search parameter)
    params.filter_size = 100;

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

    solver.write("examples/trajgen/data/a1_walk.json");

    return 0;
}