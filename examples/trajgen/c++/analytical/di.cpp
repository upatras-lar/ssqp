#include <iostream>
#include <cmath>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp

#include "model/Model.hpp"               // ssqp::model::Model
#include "model/stagewise/StageCost.hpp" // ssqp::model::stagewise::StageCost
#include "model/stagewise/StageDynamics.hpp" // ssqp::model::stagewise::StageDynamics
#include "model/stagewise/StageConstraints.hpp" // ssqp::model::stagewise::StageConstraints

#define HORIZON 50
#define dt      0.01

class Dynamics : public ssqp::model::stagewise::StageDynamics
{
    uint output_size() const override
    {
        return 2;
    }

    uint state_size() const override
    {
        return 2;
    }

    uint control_size() const override
    {
        return 1;
    }

    ssqp::VecX evaluate(const ssqp::VecX &x, const ssqp::VecX &u) const override
    {
        double vn = x[1] + dt * u[0];
        double pn = x[0] + dt * vn;

        return (Eigen::Vector2d() << pn, vn).finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix2d::Zero();
        dU = Eigen::Matrix<double, 2, 1>::Zero();

        dX(0, 0) = 1;
        dX(0, 1) = dt;
        dX(1, 1) = 1;

        dU(1, 0) = dt;
    }

    void hessprod_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

class Cost : public ssqp::model::stagewise::StageCost
{
    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>() << 0.5 * (x[0] - 1.0) * (x[0] - 1.0))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {

        dx = (Eigen::Vector2d() << (x(0) - 1.0), 0.0).finished();
        du = Eigen::VectorXd::Zero(1);
    }

    void hessian_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx       = Eigen::Matrix2d::Zero();
        Hxx(0, 0) = 1.0;
    }

    void hessian_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessian_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = (Eigen::Matrix<double, 1, 1>() << 0.002).finished();
    }
};

class Constraint : public ssqp::model::stagewise::StageConstraints
{
    uint output_size() const override
    {
        return 1;
    }

    ssqp::VecX lower_bound() const override
    {
        return (Eigen::Vector<double, 1>() << -20.0).finished();
    }

    ssqp::VecX upper_bound() const override
    {
        return (Eigen::Vector<double, 1>() << 30.0).finished();
    }

    ssqp::VecX evaluate(const ssqp::VecX &x, const ssqp::VecX &u) const override
    {
        Eigen::VectorXd v = Eigen::Vector<double, 1>::Zero();

        v(0) = u(0);

        return v;
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix<double, 1, 2>::Zero();
        dU = Eigen::Matrix<double, 1, 1>::Zero();

        dU(0, 0) = 1;
    }

    void hessprod_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

int main()
{
    std::shared_ptr<ssqp::model::stagewise::StageCost> cost =
      std::make_shared<Cost>();

    std::vector<std::shared_ptr<ssqp::model::stagewise::StageDynamics>>
      dynamics;
    dynamics.push_back(std::make_shared<Dynamics>());

    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints;

    for (uint k = 0; k <= HORIZON; ++k)
    {
        constraints.push_back(std::make_shared<Constraint>());
    }

    ssqp::model::Model::Config config;
    config.stage_cost    = cost;
    config.terminal_cost = cost;
    config.dynamics      = dynamics;
    config.constraints   = constraints;

    ssqp::model::Model model(HORIZON, dt, config);

    ssqp::solvers::ssqp::Params ssqp_params;
    ssqp_params.loose_success  = true;
    ssqp_params.qp_solver_type = ssqp::solvers::ssqp::QPSolverType::HPIPM;

    ssqp::solvers::ssqp        solver(model, ssqp_params);
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << ssqp::solvers::to_string(state)
              << std::endl;

    if (state == ssqp::solvers::SolverState::SUCCESS ||
        state == ssqp::solvers::SolverState::LOOSE_SUCCESS)
    {
        // Output solver solution
        std::cout << "x: " << std::endl << solution.x << std::endl;
        std::cout << "u: " << std::endl << solution.u << std::endl;
    }

    return 0;
}
