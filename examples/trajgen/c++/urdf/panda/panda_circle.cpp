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

#define STATE_SIZE 9
#define HORIZON    400
#define dt         0.01

// Position and rotation weights
#define P_WGT 10
#define R_WGT 1

// Joint velocity weight
#define V_WGT 1e-5
// Control weight
#define U_WGT 1e-5

// Circle properties:
// Center
#define cx 0.3
#define cy 0
#define cz 0.26
// Radius
#define cr 0.22

// Trajectory properties
// Distance to cover
#define DISTANCE 2 * M_PI
// Angular speed
#define ω DISTANCE / (HORIZON * dt)
// Initial phase
#define φ0 0

// Whether to constain movement in a rectangle within the circle
// @script-parameter
#define CONSTRAIN true

inline Eigen::Vector3d circle_to_square_point(const Eigen::Vector3d &p)
{
    double theta = std::atan2(p(2) - cz, p(1) - cy);
    double c = std::cos(theta), s = std::sin(theta);
    double a = cr / std::sqrt(2.0);
    double t = a / std::max(std::abs(c), std::abs(s));

    Eigen::Vector3d res;
    res(0) = cx;
    res(1) = cy + t * c;
    res(2) = cz + t * s;

    return res;
}

inline std::pair<Eigen::Vector3d, Eigen::Vector3d> square_point_bounds(
  const Eigen::Vector3d &p)
{
    Eigen::Vector3d lb, ub;

    lb(0) = -std::numeric_limits<double>::infinity();
    ub(0) = std::numeric_limits<double>::infinity();

    if (p(1) > cy)
    {
        lb(1) = -std::numeric_limits<double>::infinity();
        ub(1) = p(1);
    }
    else
    {
        lb(1) = p(1);
        ub(1) = std::numeric_limits<double>::infinity();
    }

    if (p(2) > cz)
    {
        lb(2) = -std::numeric_limits<double>::infinity();
        ub(2) = p(2);
    }
    else
    {
        lb(2) = p(2);
        ub(2) = std::numeric_limits<double>::infinity();
    }

    return { lb, ub };
}

using namespace ssqp::model::stagewise::predefined;

int main()
{
    // Cost trajectory to track
    std::vector<Eigen::Vector3d> traj(HORIZON + 1);

    for (uint k = 0; k <= HORIZON; ++k)
    {
        double φk = φ0 + ω * k * dt;
        traj[k] << cx, cy + cr * sin(φk), cz + cr * std::cos(φk);
    }

    Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
    Rd(1, 1)           = -1;
    Rd(2, 2)           = -1;

    // Define SSQP model
    ssqp::model::Model::Structure model_structure;
    model_structure.urdf_path = "examples/urdf/panda/panda.urdf";
    ssqp::model::Model model(HORIZON, dt, model_structure);

    Eigen::VectorXd init_state = Eigen::VectorXd::Zero(2 * STATE_SIZE);

    init_state.head(STATE_SIZE) << 0.0013, -0.9115, -0.0010, -2.6316, -0.0008,
      1.7431, 0.7862, 0.0, 0.0;

    for (uint k = 0; k < HORIZON; ++k)
    {
        Eigen::Matrix4d desired_pose        = Eigen::Matrix4d::Zero();
        desired_pose.topLeftCorner<3, 3>()  = Rd;
        desired_pose.topRightCorner<3, 1>() = traj[k];
        desired_pose(3, 3)                  = 1.0;

        // Pass an SE3 transformation matrix as a desired pose, generated with
        // the current translation for stage k and the normal rotation matrix
        model.addCost<costs::FrameSquaredPoseErrorCost>(k,
                                                        model,
                                                        desired_pose,
                                                        "panda_ee",
                                                        STATE_SIZE,
                                                        STATE_SIZE,
                                                        P_WGT,
                                                        R_WGT);

        // Add regularization costs
        model.addCost<costs::SquaredJointVelocityCost>(k, STATE_SIZE, V_WGT);
        model.addCost<costs::SquaredControlCost>(k, STATE_SIZE, U_WGT);

        if (CONSTRAIN)
        {
            // Constrain within rectangle
            Eigen::Vector3d lb, ub;
            std::tie(lb, ub) =
              square_point_bounds(circle_to_square_point(traj[k]));

            // Add a FramePositionBound for the panda_ee frame
            model.addConstraints<constraints::FramePositionBound>(k,
                                                                  model,
                                                                  "panda_ee",
                                                                  lb,
                                                                  ub,
                                                                  STATE_SIZE,
                                                                  STATE_SIZE);
        }
    }

    Eigen::Matrix4d desired_pose        = Eigen::Matrix4d::Zero();
    desired_pose.topLeftCorner<3, 3>()  = Rd;
    desired_pose.topRightCorner<3, 1>() = traj[HORIZON];
    desired_pose(3, 3)                  = 1.0;

    model.addCost<costs::FrameSquaredPoseErrorCost>(HORIZON,
                                                    model,
                                                    desired_pose,
                                                    "panda_ee",
                                                    STATE_SIZE,
                                                    STATE_SIZE,
                                                    P_WGT,
                                                    R_WGT);

    model.addCost<costs::SquaredJointVelocityCost>(HORIZON, STATE_SIZE, V_WGT);

    if (CONSTRAIN)
    {
        Eigen::Vector3d lb, ub;
        std::tie(lb, ub) =
          square_point_bounds(circle_to_square_point(traj[HORIZON]));

        model.addConstraints<constraints::FramePositionBound>(HORIZON,
                                                              model,
                                                              "panda_ee",
                                                              lb,
                                                              ub,
                                                              STATE_SIZE,
                                                              STATE_SIZE);
    }

    ssqp::model::Model::Config model_config;
    model_config.init_states = { init_state };
    model.finalize(model_config);

    // Use custom SSQP parameters
    ssqp::solvers::ssqp::Params params;
    // Disable loose success
    params.loose_success = false;
    // Edit the SSQP tolerance
    params.tolerance = 1e-2;

    // Use custom ADMM parameters
    ssqp::solvers::admm_qp::Params admm_params;
    // Change the maximum ADMM iterations
    admm_params.max_iter = 1000;
    // Edit the ADMM tolerance
    admm_params.abs_tolerance = 1e-3;
    admm_params.rel_tolerance = 1e-4;
    // Do not reset rho and y for consecutive ADMMs
    admm_params.reset_rho = false;
    admm_params.reset_y   = false;

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

    solver.write("examples/trajgen/data/panda_circle.json");

    return 0;
}