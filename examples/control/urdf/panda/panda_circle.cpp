#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define STATE_SIZE 9
#define HORIZON    400
#define dt         0.01
#define P_WGT      10
#define R_WGT      0.5
#define V_WGT      1e-4
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

#define USE_BOX false

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

inline Eigen::Vector3d logMapSO3(const Eigen::Matrix3d &R)
{
    const double cos_theta = (R.trace() - 1.0) * 0.5;
    double       theta     = std::acos(std::clamp(cos_theta, -1.0, 1.0));

    Eigen::Vector3d omega;
    if (theta < 1e-8)
    {
        // Near zero rotation: first-order approximation
        omega = 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2),
                                      R(0, 2) - R(2, 0),
                                      R(1, 0) - R(0, 1));
    }
    else
    {
        double sin_theta = std::sin(theta);
        omega = (0.5 * theta / sin_theta) * Eigen::Vector3d(R(2, 1) - R(1, 2),
                                                            R(0, 2) - R(2, 0),
                                                            R(1, 0) - R(0, 1));
    }
    return omega;
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

    // Define cost functions
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageCost>> costs(
      HORIZON);

    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints(HORIZON + 1);

    // TODO: move these to PinocchioModel
    for (uint k = 0; k < HORIZON; ++k)
    {
        Eigen::Vector3d traj_k = traj[k];

        auto cost_eval = [model, traj_k, Rd](const Eigen::VectorXd &x,
                                             const Eigen::VectorXd &u)
        {
            Eigen::VectorXd pos;
            Eigen::MatrixXd rot;

            std::tie(pos, rot) =
              model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");

            Eigen::Vector<double, 1> res;

            res(0) = P_WGT * (pos - traj_k).squaredNorm() +
                     R_WGT * logMapSO3(Rd.transpose() * rot).squaredNorm() +
                     V_WGT * x.tail(STATE_SIZE).squaredNorm() +
                     U_WGT * u.squaredNorm();
            return res;
        };

        // TODO: rotation gradient incorrect
        auto cost_grad = [model, traj_k, Rd](const Eigen::VectorXd &x,
                                             const Eigen::VectorXd &u,
                                             Eigen::VectorXd &dx,
                                             Eigen::VectorXd &du)
        {
            Eigen::VectorXd pos;
            Eigen::MatrixXd rot;
            std::tie(pos, rot) =
              model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");
            Eigen::MatrixXd Jfk =
              model.frameJacobian(x.head(STATE_SIZE), "panda_ee");

            dx = Eigen::VectorXd::Zero(2 * STATE_SIZE);
            dx.head(STATE_SIZE) =
              2 * P_WGT * Jfk.topRows(3).transpose() * (pos - traj_k) +
              2 * R_WGT * Jfk.bottomRows(3).transpose() * Rd *
                logMapSO3(Rd.transpose() * rot);
            dx.tail(STATE_SIZE) = 2 * V_WGT * x.tail(STATE_SIZE);

            du = 2 * U_WGT * u;
        };

        auto cost_hess_xx = [model](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Hxx)
        {
            Eigen::MatrixXd Jfk =
              model.frameJacobian(x.head(STATE_SIZE), "panda_ee");

            Hxx = Eigen::MatrixXd::Zero(2 * STATE_SIZE, 2 * STATE_SIZE);
            Hxx.topLeftCorner(STATE_SIZE, STATE_SIZE) =
              2 * P_WGT * Jfk.topRows(3).transpose() * Jfk.topRows(3) +
              2 * R_WGT * Jfk.bottomRows(3).transpose() * Jfk.bottomRows(3);
            Hxx.bottomRightCorner(STATE_SIZE, STATE_SIZE) =
              2 * V_WGT * Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
        };

        auto cost_hess_uu = [model](const Eigen::VectorXd &x,
                                    const Eigen::VectorXd &u,
                                    Eigen::MatrixXd &Huu)
        {
            Huu = Eigen::MatrixXd::Zero(u.size(), u.size());
            Huu.diagonal().setConstant(2 * U_WGT);
        };

        costs[k] = ssqp::model::stagewise::factory::make_cost(
          cost_eval,
          cost_grad,
          cost_hess_xx,
          ssqp::model::ZeroHessian_xu,
          cost_hess_uu);

        Eigen::Vector3d box_k = circle_to_square_point(traj_k);

        Eigen::Vector2d lb, ub;
        if (box_k(1) > cy)
        {
            lb(0) = -std::numeric_limits<double>::infinity();
            ub(0) = box_k(1);
        }
        else
        {
            lb(0) = box_k(1);
            ub(0) = std::numeric_limits<double>::infinity();
        }

        if (box_k(2) > cz)
        {
            lb(1) = -std::numeric_limits<double>::infinity();
            ub(1) = box_k(2);
        }
        else
        {
            lb(1) = box_k(2);
            ub(1) = std::numeric_limits<double>::infinity();
        }

        auto constraint_eval =
          [model](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
        {
            Eigen::VectorXd pos;
            std::tie(pos, std::ignore) =
              model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");

            return (Eigen::Vector2d() << pos(1), pos(2)).finished();
        };

        auto constraint_grad = [model](const Eigen::VectorXd &x,
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

            dX                      = Eigen::MatrixXd::Zero(2, x.size());
            dX.leftCols(STATE_SIZE) = Jfk.block(1, 0, 2, Jfk.cols());

            dU = Eigen::MatrixXd::Zero(2, STATE_SIZE);
        };

        constraints[k] =
          ssqp::model::stagewise::factory::make_constraints(2,
                                                            lb,
                                                            ub,
                                                            constraint_eval,
                                                            constraint_grad);
    }

    Eigen::Vector3d traj_k = traj[HORIZON];

    auto term_cost_eval =
      [model, traj_k, Rd](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        Eigen::VectorXd pos;
        Eigen::MatrixXd rot;

        std::tie(pos, rot) =
          model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");

        Eigen::Vector<double, 1> res;

        res(0) =
          TERM_WGT * (P_WGT * (pos - traj_k).squaredNorm() +
                      R_WGT * logMapSO3(Rd.transpose() * rot).squaredNorm() +
                      V_WGT * x.tail(STATE_SIZE).squaredNorm());
        return res;
    };

    auto term_cost_grad = [model, traj_k, Rd](const Eigen::VectorXd &x,
                                              const Eigen::VectorXd &u,
                                              Eigen::VectorXd &dx,
                                              Eigen::VectorXd &du)
    {
        Eigen::VectorXd pos;
        Eigen::MatrixXd rot;
        std::tie(pos, rot) =
          model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");
        Eigen::MatrixXd Jfk =
          model.frameJacobian(x.head(STATE_SIZE), "panda_ee");

        dx = Eigen::VectorXd::Zero(2 * STATE_SIZE);
        dx.head(STATE_SIZE) =
          2 * P_WGT * Jfk.topRows(3).transpose() * (pos - traj_k) +
          2 * R_WGT * Jfk.bottomRows(3).transpose() * Rd *
            logMapSO3(Rd.transpose() * rot);
        dx.tail(STATE_SIZE)  = 2 * V_WGT * x.tail(STATE_SIZE);
        dx                  *= TERM_WGT;
    };

    auto term_cost_hess_xx = [model](const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &u,
                                     Eigen::MatrixXd &Hxx)
    {
        Eigen::MatrixXd Jfk =
          model.frameJacobian(x.head(STATE_SIZE), "panda_ee");

        Hxx = Eigen::MatrixXd::Zero(2 * STATE_SIZE, 2 * STATE_SIZE);
        Hxx.topLeftCorner(STATE_SIZE, STATE_SIZE) =
          2 * P_WGT * Jfk.topRows(3).transpose() * Jfk.topRows(3) +
          2 * R_WGT * Jfk.bottomRows(3).transpose() * Jfk.bottomRows(3);
        Hxx.bottomRightCorner(STATE_SIZE, STATE_SIZE) =
          Eigen::MatrixXd::Identity(STATE_SIZE, STATE_SIZE);
        Hxx *= TERM_WGT;
    };

    auto terminal_cost =
      ssqp::model::stagewise::factory::make_cost(term_cost_eval,
                                                 term_cost_grad,
                                                 term_cost_hess_xx,
                                                 ssqp::model::ZeroHessian_xu,
                                                 ssqp::model::ZeroHessian_uu);

    Eigen::Vector3d box_k = circle_to_square_point(traj_k);

    Eigen::Vector2d lb, ub;
    if (box_k(1) > cy)
    {
        lb(0) = -std::numeric_limits<double>::infinity();
        ub(0) = box_k(1);
    }
    else
    {
        lb(0) = box_k(1);
        ub(0) = std::numeric_limits<double>::infinity();
    }

    if (box_k(2) > cz)
    {
        lb(1) = -std::numeric_limits<double>::infinity();
        ub(1) = box_k(2);
    }
    else
    {
        lb(1) = box_k(2);
        ub(1) = std::numeric_limits<double>::infinity();
    }

    auto constraint_eval =
      [model](const Eigen::VectorXd &x, const Eigen::VectorXd &u)
    {
        Eigen::VectorXd pos;
        std::tie(pos, std::ignore) =
          model.forwardKinematics(x.head(STATE_SIZE), "panda_ee");

        return (Eigen::Vector2d() << pos(1), pos(2)).finished();
    };

    auto constraint_grad = [model](const Eigen::VectorXd &x,
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

        dX                      = Eigen::MatrixXd::Zero(2, x.size());
        dX.leftCols(STATE_SIZE) = Jfk.block(1, 0, 2, Jfk.cols());

        dU = Eigen::MatrixXd::Zero(2, 1);
    };

    constraints[HORIZON] =
      ssqp::model::stagewise::factory::make_constraints(2,
                                                        lb,
                                                        ub,
                                                        constraint_eval,
                                                        constraint_grad);

    if (USE_BOX)
    {
        model.finalize(costs, terminal_cost, constraints, { init_state });
    }
    else
    {
        model.finalize(costs, terminal_cost, {}, { init_state });
    }

    ssqp::solvers::ssqp::Params params;
    // params.loose_success = false;
    params.tolerance = 1e-2;
    // params.loose_success_tolerance = 1e-3;
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

    solver.write("visualizer/data/panda_circle");

    return 0;
}