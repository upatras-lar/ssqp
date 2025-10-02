#include <iostream>

#include "solvers/csqp.hpp"

#include "model/cost/AbstractCost.hpp"
#include "model/constraint/AbstractConstraint.hpp"
#include "model/dynamics/AbstractDynamics.hpp"

#define HORIZON 1

class Cost : public ssqp::cost::AbstractCost
{
  public:
    double evaluate(const uint k, const ssqp::VecX &x,
                    const ssqp::VecX &u) const override
    {
        if (k == 0) return -u(0) * u(0) - u(1) * u(1);
        else return 0;
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::VecX &dx, ssqp::VecX &du) const override
    {
        dx = Eigen::Vector<double, 1>::Zero();
        du = Eigen::Vector<double, 2>::Zero();

        if (k == 0)
            du =
              (Eigen::Vector<double, 2>() << -2 * u(0), -2 * u(1)).finished();
    }

    void hessian_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessian_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 1, 2>::Zero();
    }

    void hessian_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = Eigen::Matrix2d::Zero();
        if (k == 0)
        {
            Huu(0, 0) = -2;
            Huu(1, 1) = -2;
        }
    }
};

class Constraints : public ssqp::constraint::AbstractConstraint
{
    int output_size() const override
    {
        return 5;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        Eigen::VectorXd v = Eigen::Vector<double, 5>::Zero();

        if (k == 0)
        {
            v(0) = 0.035 * u(0) + 0.2 * u(1) - 20;
            v(1) = -1.23 * u(0) - 3.1 * u(1) + 500;
            v(2) = u(0);
            v(3) = u(1);
            v(4) = u(0) - 250;
        }

        return v;
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix<double, 5, 1>::Zero();
        dU = Eigen::Matrix<double, 5, 2>::Zero();

        if (k == 0)
        {
            dU(0, 0) = 0.035;
            dU(0, 1) = 0.2;
            dU(1, 0) = -1.23;
            dU(1, 1) = -3.1;
            dU(2, 0) = 1;
            dU(3, 1) = 1;
            dU(4, 0) = 1;
        }
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 1, 2>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix2d::Zero();
    }
};

class Dynamics : public ssqp::dynamics::AbstractDynamics
{
    int output_size() const override
    {
        return 1;
    }

    int control_size() const override
    {
        return 2;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>() << x(0)).finished();
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = (Eigen::Matrix<double, 1, 1>() << 1).finished();
        dU = Eigen::Matrix<double, 1, 2>::Zero();
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix<double, 1, 1>::Zero();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 1, 2>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix2d::Zero();
    }
};

int main()
{
    ssqp::OCP ocp;
    ocp.costs       = std::make_shared<Cost>();
    ocp.constraints = std::make_shared<Constraints>();
    ocp.dynamics    = std::make_shared<Dynamics>();
    ocp.horizon     = HORIZON;

    ssqp::solvers::csqp        solver(ocp);
    ssqp::solvers::csqp::Guess solution;
    ssqp::solvers::SolverState state = solver.solve(solution);

    std::cout << "Solver state: " << state << std::endl;

    std::cout << "x: " << solution.x << std::endl;
    std::cout << "u: " << solution.u << std::endl;

    return 0;
}