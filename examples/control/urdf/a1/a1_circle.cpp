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
#define HORIZON      100
#define dt           0.01
// Trunk position weight
#define P_WGT 100
// Orientation weight
#define R_WGT 100
// Velocity regularization weight
#define V_WGT 1e-5
// Control regularization weight
#define U_WGT 1e-5
// Force regularization weight
#define F_WGT 1e-5

using namespace ssqp::model::stagewise::predefined;

std::vector<Eigen::Vector3d> make_trajectory(int H)
{
    const double z = 0.3464;
    const double r = 0.05;

    const Eigen::Vector3d start(0.0, 0.0, z);
    const Eigen::Vector3d entry(0.0, -r, z);

    // --- Lengths ---
    double L_seg   = r;
    double L_circ  = 2.0 * M_PI * r;
    double L_total = L_seg + L_circ;

    int N_seg  = std::max(1, int(std::floor((L_seg / L_total) * H)));
    int N_circ = H - N_seg;
    if (N_circ < 1) N_circ = 1;

    std::vector<Eigen::Vector3d> traj;
    traj.reserve(H);

    // --- Straight segment ---
    for (int k = 0; k < N_seg; ++k)
    {
        double s = double(k) / double(N_seg);
        traj.emplace_back(start + s * (entry - start));
    }

    // Make sure the straight portion ends exactly at entry
    traj.push_back(entry);

    // --- Circular portion ---
    for (int k = 0; k < N_circ; ++k)
    {
        double theta = 2.0 * M_PI * (double(k) / double(N_circ));
        double x     = r * std::sin(theta);
        double y     = -r * std::cos(theta);
        traj.emplace_back(x, y, z);
    }

    // Trim if one extra due to push_back(entry)
    if (int(traj.size()) > H) traj.resize(H);

    return traj;
}

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

    std::vector<Eigen::Vector3d> traj = make_trajectory(HORIZON);

    // Initialize model
    ssqp::model::Model model("examples/urdf/a1/a1.urdf", HORIZON, dt, true);
    model.printFramesAndJoints();

    for (uint k = 0; k < HORIZON; ++k)
    {
        // Add cost for trunk to stay in base position
        model.addCost<costs::FrameSquaredPoseErrorCost>(
          k,
          model,
          traj[k],
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
    }

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    traj[HORIZON],
                                                    Eigen::Matrix3d::Identity(),
                                                    "trunk",
                                                    STATE_SIZE,
                                                    CONTROL_SIZE,
                                                    P_WGT,
                                                    R_WGT);

    model.finalize({}, { init });

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 1e-1;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-4;
    admm_params.max_iter      = 100'000;

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

    solver.write("examples/data/a1_circle");

    return 0;
}