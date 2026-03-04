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
#include "solvers/hpipm_qp.hpp"

#include "nlohmann/json.hpp"

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

/* SSQP class
 * Defines an SSQP problem solver based on a given SSQP model and the desired
 * parameters.
 * 
 * SSQP is a Structure-exploiting Sequential Quadratic Programming solver.
 * The solver starts with an initial guess (either provided, or zero) for the
 * states (x) and controls (u) of an optimization problem over a time horizon H.
 * 
 * At each step, SSQP constructs a Quadratic Program at the current SSQP guess
 * with a quadratic cost function and linearized constraints. This is done using
 * the SSQP model, a model of the system being optimized which provides the
 * necessary methods to linearize at each guess, like gradients and hessians.
 * 
 * The solver then utilizes a QP solver to produce a solution to the constructed
 * QP, which can act as an update step for the current SSQP guess. Based on this
 * solution, a line search method is used to update the current guess.
 * 
 * Termination criteria are then checked for the new SSQP guess. The process is
 * repeated until convergence is reached.
 */
class ssqp
{
  public:
    enum LineSearchMethod
    {
        Filter,
        Merit
    };

    enum QPSolverType
    {
        ADMM,
        HPIPM
    };

    struct Params
    {
        Params() = default;
        Params(const std::string &filename);

        // General SSQP parameters

        // Maximum number of SSQP iterations
        uint max_iter = 10'000;
        // (Initial) Tolerance for lagrangian jacobian norms
        double tolerance = 1e-3;
        // Constraint satisfaction tolerance
        // If positive, defines a separate tolerance for constraints than the
        // general tolerance used for dx/du norms
        double constraint_tolerance = -1.0;
        // Final tolerance for lagrangian jacobian norms
        // This value will be reached after `tolerance_update_steps` SSQP steps
        // Use -1.0 to ignore
        double tolerance_final = -1.0;
        // Number of steps to go from `tolerance` to `tolerance_final`
        // Updates are done linearly per step
        // Ignored if `tolerance_final` is -1.0
        uint tolerance_update_steps = 0;
        // Whether to terminate after constraints are satisfied, regardless of
        // lagrangian jacobian norms
        bool stop_after_constraint_satisfaction = false;

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
        LineSearchMethod line_search_method = LineSearchMethod::Filter;

        // Filter line-search parameters

        // Size of filter history
        uint filter_size = 10;

        // Merit function line-search parameters

        // Weight for stage gap term
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

        // Whether to calculate full stage costs and violations at each step
        // This is usually skipped once a violation is detected to speed up
        // termination criteria checking, but can be enabled for logging
        // purposes
        static constexpr bool calculate_stagewise_convergence = true;

        // Which solver to use to solve the QPs at each step
        QPSolverType qp_solver_type = QPSolverType::ADMM;

        // Parameters for ADMM (applicable when using the ADMM solver)
        admm_qp::Params admm_params;

        // Parameters for HPIPM (applicable when using the HPIPM solver)
        hpipm_qp::Params hpipm_params;

        static Params Default()
        {
            return {};
        }

        void print() const;

        bool write(const std::string &filename) const;
        void write(nlohmann::ordered_json &j) const;
    };

    /* Guess struct
     * Stores data related to an SSQP guess.
     * This includes the solution vectors $\boldsymbol{x}$ and $\boldsymbol{u}$, as well as the lagrange
     * multipliers $\boldsymbol{\lambda}$ for equality and $\boldsymbol{\mu}$ for inequality constraints
     */
    struct Guess
    {
        // Vector sequences indexed over time horizon for state and control
        // guesses n
        FlatVec x, u;
        // Equivalent vector sequences for lagrange multiplier guesses n
        FlatVec lambda, mu;

        // Initialize a guess based on the model non-linear and linear dimensions
        Guess(const model::Dimensions &dims, const model::Dimensions &lin_dims);
    };

    // Data related to convergence
    struct IterationConvergenceData
    {
        // QP iterations
        uint qp_iters;
        // QP solver runtime
        double qp_duration;
        // SQP iteration duration
        double iter_duration;
        // Step size
        double step_size;
        // Lagrangian norms
        double norm_dx, norm_du;
        // Total trajectory cost
        double trajectory_cost;
        // Total trajectory stage gap
        double trajectory_stage_gap;
        // Total trajectory constraint violation
        double trajectory_constraint_violation;
        // First stage with a stage gap/constraint violation
        int first_violation_stage;
        // Violation at first violation stage
        double violation;
        // Stagewise data
        std::vector<double> costs;
        std::vector<double> gap_violations;
        std::vector<double> constraint_violations;
    };

    struct ConvergenceData
    {
        // Data per iteration
        std::vector<IterationConvergenceData> iteration_data;

        // Final SQP data
        // Total SSQP runtime
        double total_duration;
        // Total SSQP iterations
        uint iterations;
        // Whether termination occurred due to loose success
        bool loose_success = false;
        // Whether termination occurred due to constraint success
        bool constraint_success = false;
        // Whether termination occurred due to max iterations being reached
        bool max_iter_reached = false;
        // Total trajectory cost
        double trajectory_cost;
        // Total trajectory stage gap
        double trajectory_stage_gap;
        // Total trajectory constraint violation
        double trajectory_constraint_violation;
    };

    // Initialize an SSQP solver based on the given model and SSQP and ADMM
    // parameters
    ssqp(model::Model &model, const Params &params = Params::Default());

    // Same constructor, but assumes model is already finalized
    ssqp(const model::Model &model, const Params &params = Params::Default());

    const model::Model &model() const;

    // Solve the defined problem using SSQP
    SolverState solve(Guess &guess);

    // Write the current solution to the output file with the provided name
    bool write(const std::string &filename);

    const ConvergenceData convergenceData() const;

  protected:
    Params              _params;
    const model::Model &_model;
    Guess               _guess;
    QPData              _qp;

    FlatVec _prev_cost_dx, _prev_cost_du;
    FlatVec _cost_dx, _cost_du;

    uint _iter;
    uint _stale_iters;

    double _tolerance_init;

    ConvergenceData _convergence_data;

    template<bool withGradients>
    void _setup_qp();

    template<bool fullStagewiseConvergence>
    SolverState _check_termination();

    void _printIterationHeader() const;
    void _printIterationInfo() const;

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
