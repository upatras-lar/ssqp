#include <iostream>

#include "solvers/ssqp.hpp" // ssqp::solvers::ssqp

#include "model/Model.hpp"               // ssqp::model::Model
#include "model/stagewise/StageCost.hpp" // ssqp::model::stagewise::StageCost
#include "model/stagewise/StageDynamics.hpp" // ssqp::model::stagewise::StageDynamics
#include "model/stagewise/StageConstraints.hpp" // ssqp::model::stagewise::StageConstraints

#define HORIZON 10
#define dt      0.1

/* 1D point mass
 * state x = [p v] (position, velocity)
 * control u = [a] (acceleration)
 *
 * Start state: [0 0]
 * Target state: [0.4 x]
 * 
 * Horizon = 10, dt = 0.1
 * Dynamics:
 *  1) p(k+1) = p(k) + dt*v(k) + 0.5*dt^2*a(k)
 *  2) v(k+1) = v(k) + dt*a(k)
 * Cost:
 *  sum_{k=0}{H-1} 1e-3*a(k)^2 + 2*v(k)^2 + (p(H) - 0.4)^2
 * Constraints:
 *  a) -27 <= a(k) <= 27 (control bounds)
 *  b) 0.4 <= p(H) <= 0.4
 */

class TerminalCost : public ssqp::model::stagewise::StageCost
{
    /* Manually defined custom cost class for terminal cost
     * Class must provide implementations for:
     *  - evaluate
     *  - gradient
     *  - hessian_xx
     *  - hessian_xu
     *  - hessian_uu
     */

    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>()
                << 2 * x(1) * x(1) + (x(0) - 0.4) * (x(0) - 0.4))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {
        dx = (Eigen::Vector2d() << 2 * (x(0) - 0.4), 4 * x(1)).finished();
        du = Eigen::VectorXd::Zero(1);
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
        Huu = Eigen::Matrix<double, 1, 1>::Zero();
    }
};

class Cost : public ssqp::model::stagewise::StageCost
{
    /* Similarly, we manually define a cost class for non-terminal stages
     * Same as above, but without the terminal position term
     */

    Eigen::VectorXd evaluate(const ssqp::VecX &x,
                             const ssqp::VecX &u) const override
    {
        return (Eigen::Vector<double, 1>()
                << 1e-3 * u(0) * u(0) + 2 * x(1) * x(1))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::VecX &dx,
                  ssqp::VecX &du) const override
    {
        dx = (Eigen::Vector2d() << 0.0, 4 * x(1)).finished();
        du = (Eigen::VectorXd(1) << 2 * 1e-3 * u(0)).finished();
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
    /* Manually defined custom constraints class for terminal constraints
     * Class must provide implementations for:
     *  - output_size
     *  - lower_bound
     *  - upper_bound
     *  - evaluate
     *  - gradient
     *  - hessprod_xx
     *  - hessprod_xu
     *  - hessprod_uu
     * Here, we provide the same value for the lower and upper bounds (0.4)
     * As such, we make this an equality constraint on the desired variable (p(H))
     */

    uint output_size() const override
    {
        return 1;
    }

    const ssqp::VecX &lower_bound() const override
    {
        return _lb;
    }

    const ssqp::VecX &upper_bound() const override
    {
        return _ub;
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

    const ssqp::VecX _lb = (Eigen::Vector<double, 1>() << 0.4).finished();
    const ssqp::VecX _ub = (Eigen::Vector<double, 1>() << 0.4).finished();
};

class Constraints : public ssqp::model::stagewise::StageConstraints
{
    /* Similarly, we manually define a constraints class for non-terminal constraints
     * Here, we constrain the acceleration value (-27 <= a(k) <= 27)
     */

    uint output_size() const override
    {
        return 1;
    }

    const ssqp::VecX &lower_bound() const override
    {
        return _lb;
    }

    const ssqp::VecX &upper_bound() const override
    {
        return _ub;
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

    const ssqp::VecX _lb = (Eigen::Vector<double, 1>() << -27).finished();
    const ssqp::VecX _ub = (Eigen::Vector<double, 1>() << 27).finished();
};

class Dynamics : public ssqp::model::stagewise::StageDynamics
{
    /* Manually defined custom dynamics class
     * Class must provide implementations for:
     *  - output_size
     *  - state_size
     *  - control_size
     *  - evaluate
     *  - gradient
     *  - hessprod_xx
     *  - hessprod_xu
     *  - hessprod_uu
     * Since we are not using a URDF to initialize the SSQP model, the dynamics
     * must be defined analytically
     */

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
        return (Eigen::Vector2d() << x(0) + dt * x(1) + 0.5 * dt * dt * u(0),
                x(1) + dt * u(0))
          .finished();
    }

    void gradient(const ssqp::VecX &x, const ssqp::VecX &u, ssqp::MatX &dX,
                  ssqp::MatX &dU) const override
    {
        dX = Eigen::Matrix2d::Zero();
        dU = Eigen::Matrix<double, 2, 1>::Zero();

        dX(0, 0) = 1;
        dX(0, 1) = dt;
        dX(1, 1) = 1;

        dU(0, 0) = 0.5 * dt * dt;
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

int main()
{
    // A cost, dynamics and (optionally) constraints must be passed for each stage
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

    // Pass generated vectors directly to the model config
    ssqp::model::Model::Config model_config;
    model_config.costs         = costs;
    model_config.terminal_cost = costs[HORIZON];
    model_config.dynamics      = dynamics;
    model_config.constraints   = constraints;

    // Initialize the model with our HORIZON, dt and config
    ssqp::model::Model model(HORIZON, dt, model_config);

    // Create an SSQP solver object based on the model
    // HPIPM solves this QP much faster than the default ADMM backend
    ssqp::solvers::ssqp::Params params;
    params.qp_solver_type = ssqp::solvers::ssqp::QPSolverType::HPIPM;
    ssqp::solvers::ssqp solver(model, params);
    // Create a guess object using the model's dimensions and linear dimensions
    ssqp::solvers::ssqp::Guess solution(model.dims(), model.linear_dims());
    // Solve
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