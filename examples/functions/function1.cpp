#include <iostream>

#include "solvers/csqp.hpp"

/* Function example 1
 * Goal: min f([x y]) = (x - 1)^2 + (y - 2.5)^2
 * subject to:
 * x - 2y + 2 >= 0
 * -x - 2y + 6 >= 0
 * -x + 2x + 2 >= 0
 * x^2 + y^2 - 25 = 0
 */

class Function : public ssqp::cost::AbstractCost
{
  public:
    double evaluate(const uint k, const ssqp::VecX &x,
                    const ssqp::VecX &u) const override
    {
        return (x(0) - 1) * (x(0) - 1) + (u(0) - 2.5) * (u(0) - 2.5);
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::VecX &dx, ssqp::VecX &du) const override
    {
        dx = (Eigen::VectorXd(1) << 2 * (x(0) - 1)).finished();
        du = (Eigen::VectorXd(1) << 2 * (u(0) - 2.5)).finished();
    }

    void hessian_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx = (Eigen::MatrixXd(1, 1) << 2).finished();
    }

    void hessian_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessian_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = (Eigen::MatrixXd(1, 1) << 2).finished();
    }
};

class InequalityConstraints : public ssqp::constraint::AbstractConstraint
{
    int output_size() const override
    {
        return 3;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        return (Eigen::Vector3d() << x(0) - 2 * (u(0)) + 2,
                -x(0) - 2 * u(0) + 6,
                -x(0) + 2 * u(0) + 2)
          .finished();
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = (Eigen::MatrixXd(3, 1) << 1, -1, -1).finished();
        dU = (Eigen::MatrixXd(3, 1) << -2, -2, 2).finished();
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

class EqualityConstraints : public ssqp::dynamics::AbstractDynamics
{
    int output_size() const override
    {
        return 1;
    }

    int control_size() const override
    {
        return 1;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        return (Eigen::VectorXd(1) << x(0) * x(0) + u(0) * u(0) - 25)
          .finished();
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = (Eigen::MatrixXd(1, 1) << 2 * x(0)).finished();
        dU = (Eigen::MatrixXd(1, 1) << 2 * u(0)).finished();
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxx) const override
    {
        Pxx = (Eigen::MatrixXd(1, 1) << 2 * lambda(0)).finished();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Puu) const override
    {
        Puu = (Eigen::MatrixXd(1, 1) << 2 * lambda(0)).finished();
    }
};

int main()
{
    ssqp::solvers::OCP ocp;
    ocp.costs       = std::make_shared<Function>();
    ocp.constraints = std::make_shared<InequalityConstraints>();
    ocp.dynamics    = std::make_shared<EqualityConstraints>();

    ssqp::solvers::csqp  solver(ocp);
    ssqp::solvers::Guess solution = solver.solve();

    std::cout << "x: " << solution.x << std::endl;
    std::cout << "y: " << solution.u << std::endl;
}