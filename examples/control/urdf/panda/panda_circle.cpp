#include <iostream>
#include <cmath>
#include <vector>

#include "solvers/ssqp.hpp"
#include "model/Model.hpp"

#define STATE_SIZE 9
#define HORIZON    100
#define dt         0.01
#define P_WGT      1
#define R_WGT      0
#define V_WGT      0
#define U_WGT      1e-5

// Circle properties:
// Center
#define cx 0
#define cy 0
#define cz 0.48
// Radius
#define cr 0.3
// Angular speed
#define ω 2 * M_PI / (HORIZON * dt)
// Initial phase
#define φ0 0

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

    for (uint k = 0; k <= HORIZON; ++k)
    {
        double φk = φ0 + ω * k * dt;
        traj[k] << cx + cr * std::cos(φk), cy + cr * sin(φk), cz;
    }

    Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
    Rd(1, 1)           = -1;
    Rd(2, 2)           = -1;

    // Define SSQP model
    ssqp::model::Model model("examples/urdf/panda.urdf", HORIZON, dt);

    Eigen::VectorXd init_state = Eigen::VectorXd::Zero(2 * STATE_SIZE);
    // init_state.head(STATE_SIZE) =
    //   model.inverseKinematics(traj[0],
    //                           Eigen::Matrix3d::Identity(),
    //                           "panda_joint7");
    init_state.head(STATE_SIZE) << 0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785,
      0.0, 0.0;

    DEBUG_LOGLN("Initial state: " << init_state.transpose());

    // Define cost functions
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageCost>> costs(
      HORIZON + 1);

    for (uint k = 0; k <= HORIZON; ++k)
    {
        Eigen::Vector3d traj_k = traj[k];

        ssqp::model::EvaluationFunction cost_eval =
          [model, traj_k, Rd](const Eigen::VectorXd &x,
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

        ssqp::model::GradientFunction cost_grad =
          [model, traj_k, Rd](const Eigen::VectorXd &x,
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
              2 * R_WGT * Jfk.bottomRows(3).transpose() *
                logMapSO3(Rd.transpose() * rot);
            dx.tail(STATE_SIZE) = 2 * V_WGT * x.tail(STATE_SIZE);

            du = 2 * U_WGT * u;
        };

        ssqp::model::HessianFunction cost_hess_xx =
          [model](const Eigen::VectorXd &x,
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
        };

        ssqp::model::HessianFunction cost_hess_uu =
          [model](const Eigen::VectorXd &x,
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
    }

    model.finalize(costs, costs[HORIZON], {}, { init_state });

    ssqp::solvers::ssqp::Params params;
    params.tolerance = 0.1;
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

    solver.write("visualizer/data/panda_circle");

    return 0;
}