#include <iostream>

#include "solvers/ssqp.hpp"

#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageConstraints.hpp"

#define HORIZON 10

/* 1D point mass
 * state x = [p v] (position, velocity)
 * control u = [a] (acceleration)
 *
 * Start state: [0 0]
 * Target state: [1 x]

 * Constraints:
 *  a) -10 <= a <= 10 (control bounds)
 *
 * Horizon = 10, dt = 0.1
 * Dynamics:
 *  1) p(k+1) = p(k) + dt*v(k) + 0.5*dt^2*a(k)
 *  2) v(k+1) = v(k) + dt*a(k)
 * Cost:
 *  sum_{k=0}{H-1} 10e-5*a(k)^2 + (p(H) - 1)^2
 * Constraints:
 *  1) a(k) + 10 >= 0
 *  2) 10 - a(k) >= 0
 */

class TerminalCost : public ssqp::model::stagewise::StageCost
{
  public:
    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>() << u(0) * u(0) * 0.001 +
                                                2 * x(1) * x(1) +
                                                (x(0) - 0.4) * (x(0) - 0.4))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {
        dx = (Eigen::Vector2d() << 2 * (x(0) - 0.4), 4 * x(1)).finished();
        du = (Eigen::VectorXd(1) << 0.002 * u(0)).finished();
    }

    void hessian_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx       = Eigen::Matrix2d::Zero();
        Hxx(0, 0) = 2;
        Hxx(1, 1) = 4;
    }

    void hessian_xu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessian_uu(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = (Eigen::Matrix<double, 2, 1>() << 0.002).finished();
    }
};

class Cost : public ssqp::model::stagewise::StageCost
{
  public:
    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>()
                << u(0) * u(0) * 0.001 + 2 * x(1) * x(1))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {

        dx = (Eigen::Vector2d() << 2 * (x(0) - 0.4), 4 * x(1)).finished();
        du = (Eigen::VectorXd(1) << 0.002 * u(0)).finished();
    }

    void hessian_xx(const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx       = Eigen::Matrix2d::Zero();
        Hxx(1, 1) = 4;
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

class TerminalConstraints : public ssqp::model::stagewise::StageConstraints
{
    uint output_size() const override
    {
        return 1;
    }

    ssqp::VecX lower_bound() const override
    {
        return (Eigen::Vector<double, 1>() << 12).finished();
    }

    ssqp::VecX upper_bound() const override
    {
        return (Eigen::Vector<double, 1>() << 12).finished();
    }

    ssqp::VecX evaluate(const ssqp::VecX &x, const ssqp::VecX &u) const override
    {
        Eigen::VectorXd v = Eigen::Vector<double, 1>::Zero();

        v(0) = x(0);

        return v;
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix<double, 1, 2>::Zero();
        dU = Eigen::Matrix<double, 1, 1>::Zero();

        dX(0, 0) = 1;
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

class Constraints : public ssqp::model::stagewise::StageConstraints
{
    uint output_size() const override
    {
        return 1;
    }

    ssqp::VecX lower_bound() const override
    {
        return (Eigen::Vector<double, 1>() << -27).finished();
    }

    ssqp::VecX upper_bound() const override
    {
        return (Eigen::Vector<double, 1>() << 27).finished();
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
        return (Eigen::Vector2d() << x(0) + 0.1 * x(1) + 0.005 * u(0),
                x(1) + 0.1 * u(0))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix2d::Zero();
        dU = Eigen::Matrix<double, 2, 1>::Zero();

        dX(0, 0) = 1;
        dX(0, 1) = 0.1;
        dX(1, 1) = 1;

        dU(0, 0) = 0.005;
        dU(1, 0) = 0.1;
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

int main()
{
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageCost>> costs(
      HORIZON + 1);
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageDynamics>>
      dynamics(HORIZON + 1);
    std::vector<std::shared_ptr<ssqp::model::stagewise::StageConstraints>>
      constraints(HORIZON + 1);

    for (uint k = 0; k < HORIZON; ++k)
    {
        costs[k]       = std::make_shared<Cost>();
        dynamics[k]    = std::make_shared<Dynamics>();
        constraints[k] = std::make_shared<Constraints>();
    }

    costs[HORIZON]       = std::make_shared<TerminalCost>();
    dynamics[HORIZON]    = std::make_shared<Dynamics>();
    constraints[HORIZON] = std::make_shared<TerminalConstraints>();

    ssqp::model::Model model(costs, dynamics, constraints);

    ssqp::solvers::ssqp        solver(model);
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

    solver.write("examples/data/point_mass");

    return 0;
}