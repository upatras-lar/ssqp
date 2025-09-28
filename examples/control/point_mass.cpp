#include <iostream>

#include "solvers/csqp.hpp"

/* 1D point mass
 * state x = [p v] (position, velocity)
 * control u = [a] (acceleration)
 *
 * Start state: [0 0]
 * Target state: [1 0]
 * Constraints:
 *  a) -1 <= a <= 1 (control bounds)
 *  b) T = 1 (fixed final time)
 *
 * Assuming horizon = 10, dt = 1 / 10
 * Dynamics:
 *  1) p(k+1) = p(k) + dt*v(k)
 *  2) v(k+1) = v(k) + dt*a(k)
 * Cost:
 *  sum_{k=0}{H-1} a(k)^2*dt + (p(H) - 1)^2
 * Constraints:
 *  1) a(k) + 1 >= 0
 *  2) 1 - a(k) >= 0
 */

class Cost : public ssqp::cost::AbstractCost
{
  public:
    double evaluate(const uint k, const ssqp::VecX &x,
                    const ssqp::VecX &u) const override
    {
        // a(k)^2 * dt
        // TODO: u weight issue (larger -> blows up)
        return u(0) * u(0) * 0.1 + 20 * (x(0) - 1) * (x(0) - 1);
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::VecX &dx, ssqp::VecX &du) const override
    {
        dx = (Eigen::Vector2d() << 40 * (x(0) - 1), 0).finished();
        // TODO: If this is 0, convergence succeeds
        du = (Eigen::VectorXd(1) << 0.2 * u(0)).finished();
    }

    void hessian_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxx) const override
    {
        Hxx       = Eigen::Matrix2d::Zero();
        Hxx(0, 0) = 40;
    }

    void hessian_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Hxu) const override
    {
        Hxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessian_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                    ssqp::MatX &Huu) const override
    {
        Huu = (Eigen::Matrix<double, 1, 1>() << 0.2).finished();
    }
};

class Constraints : public ssqp::constraint::AbstractConstraint
{
    int output_size() const override
    {
        return 2;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        Eigen::VectorXd v = Eigen::Vector<double, 2>::Zero();
        v(0)              = u(0) + 1000;
        v(1)              = 1000 - u(0);
        return v;
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix<double, 2, 2>::Zero();
        dU = Eigen::Matrix<double, 2, 1>::Zero();

        dU(0, 0) = 1;
        dU(1, 0) = -1;
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &mu, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

class Dynamics : public ssqp::dynamics::AbstractDynamics
{
    int output_size() const override
    {
        return 2;
    }

    int control_size() const override
    {
        return 1;
    }

    ssqp::VecX evaluate(const uint k, const ssqp::VecX &x,
                        const ssqp::VecX &u) const override
    {
        return (Eigen::Vector2d() << x(0) + 0.1 * x(1), x(1) + 0.1 * u(0))
          .finished();
    }

    void gradient(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                  ssqp::MatX &dX, ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix2d::Zero();
        dU = Eigen::Matrix<double, 2, 1>::Zero();

        dX(0, 0) = 1;
        dX(0, 1) = 0.1;
        dX(1, 1) = 1;

        dU(1, 0) = 0.1;
    }

    void hessprod_xx(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxx) const override
    {
        Pxx = Eigen::Matrix2d::Zero();
    }

    void hessprod_xu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Pxu) const override
    {
        Pxu = Eigen::Matrix<double, 2, 1>::Zero();
    }

    void hessprod_uu(const uint k, const ssqp::VecX &x, const ssqp::VecX &u,
                     const ssqp::VecX &lambda, ssqp::MatX &Puu) const override
    {
        Puu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

int main()
{
    ssqp::solvers::OCP ocp;
    ocp.costs       = std::make_shared<Cost>();
    ocp.constraints = std::make_shared<Constraints>();
    ocp.dynamics    = std::make_shared<Dynamics>();

    ssqp::solvers::csqp  solver(ocp);
    ssqp::solvers::Guess solution = solver.solve();

    std::cout << "x: " << solution.x << std::endl;
    std::cout << "u: " << solution.u << std::endl;

    return 0;
}