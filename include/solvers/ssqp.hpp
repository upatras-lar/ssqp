#ifndef SSQP_SOLVERS_SSQP_HPP
#define SSQP_SOLVERS_SSQP_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <variant>
#include <deque>
#include <chrono>
#include <memory>

#include "types.hpp"
#include "sqp_step/step.hpp"
#include "sqp_step/filter.hpp"
#include "sqp_step/merit.hpp"
#include "model/Model.hpp"
#include "solvers/admm_qp.hpp"
#include "solvers/hpipm_qp.hpp"
#include "solvers/piqp_qp.hpp"

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
auto to_compile_time(bool b);

/** Load a previously generated SSQP trajectory from a file
 * Input:
 *  - `filepath`: The path for the trajectory file output by `ssqp::write`
 * 
 * Returns: a pair of `FlatVec` for the full state and control vectors
 */
std::pair<FlatVec, FlatVec> loadTrajectory(const std::string &filepath);

/** SSQP class
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
        ADMM  = 0,
        HPIPM = 1,
        PIQP  = 2,
    };

    static std::string QPSolverTypeName(QPSolverType type)
    {
        switch (type)
        {
        case ADMM:
        default:
            return "ADMM";
        case HPIPM:
            return "HPIPM";
        case PIQP:
            return "PIQP";
        }
    }

    struct Params
    {
        Params() = default;
        Params(const std::string &filepath);

        // General SSQP parameters

        // Maximum number of SSQP iterations
        uint max_iter = 10'000;
        // Maximum solve time (in seconds)
        // Iterations expected to end past it are skipped
        // Use -1.0 to ignore
        double max_time = -1.0;
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
        // Whether to stop when the QP solver reports an infeasible QP
        // Otherwise ADMM skips infeasibility detection and runs as usual
        bool stop_on_qp_infeasibility = true;
        // Whether to skip the termination check after the last iteration
        // max_iter allows, which saves its gradient evaluations
        // A solve that converges on that iteration then reports reaching
        // max_iter
        bool skip_last_termination_check = false;

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

        // Parameters for PIQP (applicable when using the PIQP solver)
        piqp_qp::Params piqp_params;

        static Params Default()
        {
            return {};
        }

        void print() const;

        bool write(const std::string &filepath) const;
        void write(nlohmann::ordered_json &j) const;
    };

    /** Guess struct
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
        // QP solver runtime (in μs)
        double qp_duration_us;
        // SQP iteration duration (in μs)
        double iter_duration_us;
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

        void print() const;
    };

    struct ConvergenceData
    {
        // Data per iteration
        std::vector<IterationConvergenceData> iteration_data;

        // Final SQP data
        // Total SSQP runtime (in μs)
        double total_duration_us;
        // Total SSQP iterations
        uint iterations;
        // Whether termination occurred due to loose success
        bool loose_success = false;
        // Whether termination occurred due to constraint success
        bool constraint_success = false;
        // Whether termination occurred due to max iterations being reached
        bool max_iter_reached = false;
        // Whether termination occurred due to the time limit being reached
        bool time_limit_reached = false;
        // Total trajectory cost
        double trajectory_cost;
        // Total trajectory stage gap
        double trajectory_stage_gap;
        // Total trajectory constraint violation
        double trajectory_constraint_violation;

        void print() const;
    };

    // Initialize an SSQP solver based on the given model and SSQP and ADMM
    // parameters
    ssqp(model::Model &model, const Params &params = Params::Default());

    // Same constructor, but assumes model is already finalized
    ssqp(const model::Model &model, const Params &params = Params::Default());

    const model::Model &model() const;

    // Reset the internal guess from the model's current initial states and
    // controls.
    void reinitialize();

    // Solve the defined problem using SSQP
    SolverState solve(Guess &guess);

    // Write the current solution to the output file with the provided name
    bool write(const std::string &filepath);

    // Get the SSQP solver's convergence data (after solving)
    const ConvergenceData convergenceData() const;

    // Set the reduced SSQP output flag
    void setReducedOutput(const bool reduced);

    // Feedback gain K (nu-by-nx) of the last QP at stage k, giving the change
    // du = K dx of the stage's controls for a change dx of its state, in the
    // tangent space of the dynamics (see state_right_minus)
    MatX feedbackGain(const uint k) const;

  protected:
    Params                    _params;
    const model::Model       &_model;
    Guess                     _guess;
    QPData                    _qp;
    std::unique_ptr<QPSolver> _qp_solver;

    FlatVec _prev_cost_dx, _prev_cost_du;
    FlatVec _cost_dx, _cost_du;

    // Scratch trajectory overwritten by the line search each iteration
    FlatVec _candidate_x, _candidate_u;

    // Scratch Hessian-block buffers for _setup_qp, one per shape so reuse
    // doesn't thrash between shapes within a stage
    MatX _tmp_hess_xx, _tmp_hess_xu, _tmp_hess_uu;

    // Scratch XML Jacobian-transform buffers for _setup_qp (nv/nu are
    // whole-model constants, not per-stage)
    MatX _tmp_jac_x, _tmp_jac_u, _tmp_jac_r_inv, _tmp_jac_r_inv_act;

    // Scratch manifold-difference buffers for _gammaf_grad
    mutable MatX _tmp_jr_inv_wrt_x, _tmp_jr_inv_wrt_f;
    mutable VecX _tmp_gammaf_w;

    // Per-stage dynamics/constraints evaluate() outputs at the current guess
    // Filled in by _check_termination and reused by the following _setup_qp
    // call, since both run at the same guess
    std::vector<VecX> _cached_f, _cached_c;

    uint _iter;
    uint _stale_iters;

    double _tolerance_init;

    ConvergenceData _convergence_data;

    bool _reduced_output;

    // firstSetup: whether this is the QP setup before the solve loop starts,
    // with no prior _check_termination call to reuse dynamics/constraints
    // evaluate() outputs and gradients from
    template<bool firstSetup, bool XMLModel = true>
    void _setup_qp();

    template<bool fullStagewiseConvergence>
    SolverState _check_termination();

    // Whether another iteration as long as the current one would exceed
    // max_time
    bool _out_of_time(
      const std::chrono::high_resolution_clock::time_point &start,
      const std::chrono::high_resolution_clock::time_point &iter_start) const;

    // Throws if any constraint has a lower bound above its upper bound
    // Bounds may change between solves, so this runs at the start of each one
    void _check_bounds() const;

    // Returns the current guess of an unconverged solve
    SolverState _finish_unconverged(
      SolverState state, uint iterations, Guess &guess,
      const std::chrono::high_resolution_clock::time_point &start);

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
    // reuse_cached_f: whether (x, u) is _guess.x/_guess.u at its current
    // value, letting the per-stage dynamics evaluate() be read from
    // _cached_f instead of recomputed
    void _gammaf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx,
                      FlatVec &du, bool reuse_cached_f = false) const;
    friend class tests::FDGradFullTrajectoryStageGap;

    // Constraint violation
    double _cf(const FlatVec &x, const FlatVec &u) const;
    double _cf_softmax(const FlatVec &x, const FlatVec &u) const;
    // reuse_cached_c: whether (x, u) is _guess.x/_guess.u at its current
    // value, letting the per-stage constraints evaluate() be read from
    // _cached_c instead of recomputed
    void _cf_grad(const FlatVec &x, const FlatVec &u, FlatVec &dx, FlatVec &du,
                  bool reuse_cached_c = false) const;

    friend class tests::FDGradFullTrajectoryConstraintViolation;

    std::deque<std::tuple<double, double, double>> _filter_memory;

    // (cost, stage gap, constraint violation) at the current guess, kept from
    // the line search's accepted point so that it is evaluated once
    std::tuple<double, double, double> _current_metrics;

    // Temporaries to prevent aliasing for Eigen operations
    struct
    {
        VecX x1;
        VecX u1;
    } _tmp;

    const VecX _emptyVec = VecX::Zero(0);
};
}
}

#include "src/solvers/ssqp.tpp"

#endif
