#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "model/Model.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#define STATE_SIZE 9
#define HORIZON    400
#define dt         0.01
#define P_WGT      10
#define R_WGT      1
#define V_WGT      1e-5
#define U_WGT      1e-5
#define TERM_WGT   1

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

#define CONSTRAIN false

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
    // model_structure.integration   = ssqp::model::ExplicitEuler;
    // model_structure.dynamics_mode = ssqp::model::DynamicsMode::Inverse;
    ssqp::model::Model model(HORIZON, dt, model_structure);

    Eigen::VectorXd init_state = Eigen::VectorXd::Zero(2 * STATE_SIZE);

    init_state.head(STATE_SIZE) << 0.0013, -0.9115, -0.0010, -2.6316, -0.0008,
      1.7431, 0.7862, 0.0, 0.0;

    LOGLN("Trajectory[0]: " << traj[0].transpose());
    Eigen::VectorXd pos;
    std::tie(pos, std::ignore) =
      model.forwardKinematics(init_state.head(STATE_SIZE), "panda_ee");
    LOGLN("Actual initial state: " << pos.transpose());

    // std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
    //   constraints(HORIZON + 1, nullptr);

    // model.printFramesAndJoints();
    // model.addForce(0, "panda_ee");
    //model.addForce(2, "panda_ee");

    for (uint k = 0; k < HORIZON; ++k)
    {
        // model.addCost<costs::FrameSquaredPositionErrorCost>(k,
        //                                                     model,
        //                                                     traj[k],
        //                                                     "panda_ee",
        //                                                     STATE_SIZE,
        //                                                     STATE_SIZE,
        //                                                     P_WGT);

        // model.addCost<costs::FrameSquaredRotationErrorCost>(k,
        //                                                     model,
        //                                                     Rd,
        //                                                     "panda_ee",
        //                                                     STATE_SIZE,
        //                                                     STATE_SIZE,
        //                                                     R_WGT);

        Eigen::Matrix4d desired_pose        = Eigen::Matrix4d::Zero();
        desired_pose.topLeftCorner<3, 3>()  = Rd;
        desired_pose.topRightCorner<3, 1>() = traj[k];
        desired_pose(3, 3)                  = 1.0;

        model.addCost<costs::FrameSquaredPoseErrorCost>(k,
                                                        model,
                                                        desired_pose,
                                                        "panda_ee",
                                                        STATE_SIZE,
                                                        STATE_SIZE,
                                                        P_WGT,
                                                        R_WGT);

        model.addCost<costs::SquaredControlCost>(k, STATE_SIZE, U_WGT);

        model.addCost<costs::SquaredJointVelocityCost>(k, STATE_SIZE, V_WGT);

        if (CONSTRAIN)
        {
            Eigen::Vector3d lb, ub;
            std::tie(lb, ub) = square_point_bounds(
              circle_to_square_point(circle_to_square_point(traj[k])));

            model.addConstraints<constraints::FramePositionBound>(k,
                                                                  model,
                                                                  "panda_ee",
                                                                  lb,
                                                                  ub,
                                                                  STATE_SIZE,
                                                                  STATE_SIZE);
        }
    }

    // model.addCost<costs::FrameSquaredPositionErrorCost>(HORIZON,
    //                                                     model,
    //                                                     traj[HORIZON],
    //                                                     "panda_ee",
    //                                                     STATE_SIZE,
    //                                                     STATE_SIZE,
    //                                                     TERM_WGT * P_WGT);

    // model.addCost<costs::FrameSquaredRotationErrorCost>(HORIZON,
    //                                                     model,
    //                                                     Rd,
    //                                                     "panda_ee",
    //                                                     STATE_SIZE,
    //                                                     STATE_SIZE,
    //                                                     TERM_WGT * R_WGT);

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
                                                    TERM_WGT * P_WGT,
                                                    TERM_WGT * R_WGT);

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

    ssqp::solvers::ssqp::Params params;
    params.loose_success = false;
    params.tolerance     = 1e-1;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance          = 1e-3;
    admm_params.rel_tolerance          = 1e-4;
    admm_params.adaptive_rho_tolerance = 5;
    admm_params.rho_update_iter_freq   = 25;
    admm_params.reset_rho              = false;
    admm_params.reset_y                = false;
    admm_params.max_iter               = 1000;

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

    solver.write("examples/trajgen/data/panda_circle");

    return 0;
}