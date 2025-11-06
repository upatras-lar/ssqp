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

#define USE_BOX true

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

Eigen::Matrix3d skew(Eigen::Vector3d v)
{
    Eigen::Matrix3d M;

    M(0, 0) = 0;
    M(0, 1) = -v[2];
    M(0, 2) = v[1];
    M(1, 0) = v[2];
    M(1, 1) = 0;
    M(1, 2) = -v[0];
    M(2, 0) = -v[1];
    M(2, 1) = v[0];
    M(2, 2) = 0;

    return M;
}

int main()
{
    // Cost trajectory to track
    std::vector<Eigen::Vector3d> traj(HORIZON + 1);

    LOGLN("Goal trajectory:");
    for (uint k = 0; k <= HORIZON; ++k)
    {
        double φk = φ0 + ω * k * dt;
        traj[k] << cx, cy + cr * sin(φk), cz + cr * std::cos(φk);
        LOGLN(circle_to_square_point(traj[k]).transpose());
    }

    Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
    Rd(1, 1)           = -1;
    Rd(2, 2)           = -1;

    Eigen::Matrix3d Rc = Eigen::Matrix3d::Identity();
    Rc(1, 1)           = -1;
    Rc(2, 2)           = -1;

    // Define SSQP model
    ssqp::model::Model model("examples/urdf/panda.urdf", HORIZON, dt);

    Eigen::VectorXd init_state = Eigen::VectorXd::Zero(2 * STATE_SIZE);

    init_state.head(STATE_SIZE) << 0.0013, -0.9115, -0.0010, -2.6316, -0.0008,
      1.7431, 0.7862, 0.0, 0.0;

    LOGLN("Trajectory[0]: " << traj[0].transpose());
    Eigen::VectorXd pos;
    std::tie(pos, std::ignore) =
      model.forwardKinematics(init_state.head(STATE_SIZE), "panda_ee");
    LOGLN("Actual initial state: " << pos.transpose());

    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints(HORIZON + 1, nullptr);

    for (uint k = 0; k < HORIZON; ++k)
    {
        model.cost(k) = std::make_shared<
          ssqp::model::stagewise::predefined::costs::FrameSquaredPoseErrorCost>(
          model,
          traj[k],
          Rd,
          "panda_ee",
          STATE_SIZE,
          STATE_SIZE,
          P_WGT,
          R_WGT,
          V_WGT,
          U_WGT);

        Eigen::Vector3d lb, ub;
        std::tie(lb, ub) = square_point_bounds(
          circle_to_square_point(circle_to_square_point(traj[k])));

        constraints[k] = std::make_shared<
          ssqp::model::stagewise::predefined::constraints::FramePositionBound>(
          model,
          "panda_ee",
          lb,
          ub,
          STATE_SIZE,
          STATE_SIZE);
    }

    Eigen::VectorXd lb_r(9), ub_r(9);
    lb_r = Eigen::Map<const Eigen::VectorXd>(Rc.data(), 9);
    ub_r = lb_r;

    auto op = [](const Eigen::VectorXd &x, const Eigen::VectorXd &y)
    {
        Eigen::Matrix3d X, Y;
        X = x.reshaped(3, 3);
        Y = y.reshaped(3, 3);

        return ssqp::pinocchio_wrapper::logSO3(X.transpose() * Y);
    };

    /*
    auto rot_constraint_eval =
      [model](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        Eigen::MatrixXd rot;
        std::tie(std::ignore, rot) =
          model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");

        return rot.reshaped(9, 1);
    };

    auto rot_constraint_grad = [model, Rc](const Eigen::VectorXd &x,
                                           const Eigen::VectorXd &u,
                                           Eigen::MatrixXd &dX,
                                           Eigen::MatrixXd &dU)
    {
        Eigen::VectorXd pos;
        Eigen::MatrixXd rot;
        std::tie(pos, rot) =
          model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");
        Eigen::MatrixXd Jfk =
          model.frameJacobian(x.head(STATE_SIZE), "panda_ee");

        dX = Eigen::MatrixXd::Zero(3, 2 * STATE_SIZE);
        dX.leftCols(STATE_SIZE) =
          ssqp::pinocchio_wrapper::JlogSO3(Rc.transpose() * rot) *
          Rc.transpose() * Jfk.bottomRows(3);

        dU = Eigen::MatrixXd::Zero(3, STATE_SIZE);
    };

    constraints[HORIZON / 2] =
      ssqp::model::stagewise::factory::merge_constraints(
        constraints[HORIZON / 2],
        ssqp::model::stagewise::factory::make_constraints(9,
                                                          3,
                                                          lb_r,
                                                          ub_r,
                                                          rot_constraint_eval,
                                                          rot_constraint_grad,
                                                          op));
    */

    model.cost(HORIZON) = std::make_shared<
      ssqp::model::stagewise::predefined::costs::FrameSquaredPoseErrorCost>(
      model,
      traj[HORIZON],
      Rd,
      "panda_ee",
      STATE_SIZE,
      STATE_SIZE,
      TERM_WGT * P_WGT,
      TERM_WGT * R_WGT,
      TERM_WGT * V_WGT,
      0);

    Eigen::Vector3d lb, ub;
    std::tie(lb, ub) =
      square_point_bounds(circle_to_square_point(traj[HORIZON]));

    constraints[HORIZON] = std::make_shared<
      ssqp::model::stagewise::predefined::constraints::FramePositionBound>(
      model,
      "panda_ee",
      lb,
      ub,
      STATE_SIZE);

    if (USE_BOX)
    {
        model.finalize(constraints, { init_state });
    }
    else
    {
        model.finalize({}, { init_state });
    }

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 3e-2;

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
    LOGLN("Trajectories: expected vs actual (per step)");

    for (uint k = 0; k <= HORIZON; ++k)
    {
        LOGLN("Step " << k << ":");
        LOGLN("Goal: " << traj[k].transpose());
        std::tie(pos, std::ignore) =
          model.forwardKinematics(solution.x.stage(HORIZON).head(STATE_SIZE),
                                  "panda_ee");
        LOGLN("Actual: " << pos.transpose());
    }
    */

    solver.write("visualizer/data/panda_circle");

    return 0;
}