#ifndef SSQP_SOLVERS_SSQP_HPP
#define SSQP_SOLVERS_SSQP_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <variant>
#include <deque>
#include <chrono>

#include "types.hpp"
#include "sqp_step/step.hpp"
#include "sqp_step/filter.hpp"
#include "sqp_step/merit.hpp"
#include "model/Model.hpp"
#include "solvers/admm_qp.hpp"

#include "external/nlohmann/json.hpp"

namespace ssqp
{
namespace tests
{
class FDGradFullTrajectoryCost;
class FDGradFullTrajectoryStageGap;
class FDGradFullTrajectoryConstraintViolation;
}

namespace solvers
{
class ssqp
{
  public:
    struct Params
    {
        // General SSQP parameters

        // Maximum number of SSQP iterations
        uint max_iter = 10'000;
        // (Initial) Tolerance for lagrangian jacobian norms
        double tolerance = 1e-3;
        // Final tolerance for lagrangian jacobian norms
        // This value will be reached after `tolerance_update_steps` ssqp steps
        // Use -1.0 to ignore
        double tolerance_final = -1.0;
        // Number of steps to go from `tolerance` to `tolerance_final`
        // Updates are done linearly per step
        // Ignored if `tolerance_final` is -1.0
        uint tolerance_update_steps = 0;

        // Loose success parameters

        // Whether to enable loose success
        // Loose success will terminate if there are no updates for
        // `loose_success_stale_iters` that are larger than
        // `loose_success_tolerance`
        bool loose_success = false;
        // Tolerance for change in cost jacobian norms
        double loose_success_tolerance = tolerance / 10;
        // Number of iterations for stale cost jacobian norms
        uint loose_success_stale_iters = 5;

        // Line search parameters

        // Which type of line search to use
        // Choose between:
        //  - Filter line search
        //  - Merit function line search
        bool use_filter = true;

        // Filter line-search parameters

        // Size of filter history
        uint filter_size = 10;

        // Merit function line-search parameters

        // Weight for stage gap term
        // NOTE: Set to 0 in Inverse Dynamics mode
        double merit_mu_gamma = 1.0;
        // Weight for constraint violation term
        double merit_mu_c = 1.0;
        // Factor for gradient of merit function
        double merit_armijo_beta = 1e-4;
        // β-term for approximating infinity norm for stage gap and constraint
        // violation gradients using the softmax approximation
        const double merit_grad_softmax_beta = 10.0;

        // Factor to divide α (step size) by each line search step
        double inv_alpha_scale_factor = 2.0;
        // Minimum α (step size) value
        double min_alpha = 1e-4;

        // Whether to update with min_alpha if it is reached
        // Automatically also skips regularization
        bool accept_worse_update = false;
        // Whether to skip regularization step if line search fails
        // Will throw an exception instead
        bool skip_regularization = false;

        static Params Default()
        {
            return {};
        }
    };

    struct Guess
    {
        // Vector sequences indexed over time horizon for state and control
        // guesses n
        FlatVec x, u;
        // Equivalent vector sequences for lagrange multiplier guesses n
        FlatVec lambda, mu;

        Guess(const model::Dimensions &dims, const model::Dimensions &lin_dims);
    };

    struct IterationConvergenceData
    {
        // QP iterations
        uint qp_iters;
        // Step size
        double step_size;
        // Lagrangian norms
        double norm_dx, norm_du;
    };

    ssqp(model::Model &model, const Params &params = Params::Default(),
         const admm_qp::Params &admm_params = admm_qp::Params::Default());

    const model::Model &model() const;

    SolverState solve(Guess &guess);

    bool write(const std::string &filename);

  protected:
    Params          _params;
    admm_qp::Params _admm_params;
    model::Model   &_model;
    Guess           _guess;
    QPData          _qp;

    FlatVec _prev_cost_dx, _prev_cost_du;
    FlatVec _cost_dx, _cost_du;

    uint _iter;
    uint _stale_iters;

    double _tolerance_init;

    double                                _total_duration;
    std::vector<IterationConvergenceData> _iteration_convergence_data;

    template<bool withGradients>
    void _setup_qp();

    SolverState _check_termination();

    // Trajectory cost
    double _lf(const FlatVec &x, const FlatVec &u) const;
    void   _lf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                    FlatVec &du) const;
    friend class tests::FDGradFullTrajectoryCost;

    // Gap violation
    double _gammaf(const FlatVec &x, const FlatVec &u) const;
    double _gammaf_softmax(const FlatVec &x, const FlatVec &u) const;
    void   _gammaf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                        FlatVec &du) const;
    friend class tests::FDGradFullTrajectoryStageGap;

    // Constraint violation
    double _cf(const FlatVec &x, const FlatVec &u) const;
    double _cf_softmax(const FlatVec &x, const FlatVec &u) const;
    void   _cf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                    FlatVec &du) const;

    friend class tests::FDGradFullTrajectoryConstraintViolation;

    std::deque<std::tuple<double, double, double>> _filter_memory;
};
}
}

#include "src/solvers/ssqp.tpp"

#endif
