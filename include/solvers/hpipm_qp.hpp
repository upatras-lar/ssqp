#ifndef SSQP_SOLVERS_HPIPM_QP_HPP
#define SSQP_SOLVERS_HPIPM_QP_HPP

#include "solvers/qp.hpp"

#include "nlohmann/json.hpp"

namespace ssqp
{
namespace solvers
{

class hpipm_data;

enum HPIPMMode
{
    ROBUST,
    SPEED,
    BALANCE
};

class hpipm_qp : public QPSolver
{
  public:
    struct Params
    {
        Params() = default;
        Params(const nlohmann::ordered_json &hpipmj);

        // HPIPM mode
        HPIPMMode mode = ROBUST;
        // Maximum iterations
        int max_iter = 100;

        // Minimum step size
        double alpha_min = 1e-8;
        // Initial barrier parameter
        double mu0 = 1e+2;

        // Tolerances
        double tol_stat = 1e-8;
        double tol_eq   = 1e-8;
        double tol_ineq = 1e-8;
        double tol_comp = 1e-8;

        // Regularization term
        double reg_prim = 1e-12;

        // Warm start flag
        bool warm_start = false;
        // Prediction correction flag
        bool prediction_correction = true;
        // Square root riccati algorithm flag
        bool square_root_riccati = false;
        // Whether to use different step for primal and dual variables
        bool split_step = false;

        static Params Default()
        {
            return {};
        }

        void print() const;

        void write(nlohmann::ordered_json &j) const;
    };

    hpipm_qp(const QPData &qp, const Params &params = Params::Default());
    ~hpipm_qp();

    hpipm_qp(const hpipm_qp &)            = delete;
    hpipm_qp &operator=(const hpipm_qp &) = delete;

    void        set_stage_sizes();
    void        updateDimensions() override;
    void        reset() override;
    void        increaseRegularization() override;
    SolverState solve(QPSolver::Solution &sol) override;
    uint        iterations() const override;

    // Riccati gain of the last interior point iteration, where the barrier
    // stiffens the active constraints
    MatX feedbackGain(const uint k) const override;

  protected:
    // Boxes on the entries of a stage's states or controls
    // Every row on the same entry joins one box, whose bounds are the
    // tightest of the rows
    struct Boxes
    {
        // The entry of each box, and the rows that bound it, with box j's rows
        // at rows[start[j]] up to rows[start[j + 1]]
        std::vector<int>  idx;
        std::vector<uint> rows, start;

        // The row giving each side of each box, or -1 when none does
        std::vector<int> lb_row, ub_row;

        // Copies of the bounds, since infinite bounds must be replaced, but
        // the actual QP must not be altered
        VecX                lb, ub;
        std::vector<double> lb_mask, ub_mask;

        // Whether the entries are fixed to zero, as the initial state is
        bool fixed = false;

        // Makes a box for each of the n entries with control or state box
        // rows, or for every entry when they are fixed
        void layout(const std::vector<model::BoxRow> &box_rows,
                    const bool control, const uint n);

        // Sets the bounds from the rows' bounds, starting from [0, 0] when the
        // entries are fixed and from no bounds otherwise
        void merge(const VecX &cl, const VecX &cu);

        // Writes the multipliers of the boxes into the rows that gave their
        // bounds
        void multipliers(const double *lam_lb, const double *lam_ub,
                         VecXRef mu) const;
    };

    // A stage's constraint rows, split into boxes and general rows
    struct Stage
    {
        Boxes x, u;

        // The rows passed as general constraints, with HPIPM's C and D
        // holding their rows of the QP's D and E
        std::vector<uint>   general_rows;
        MatX                C, D;
        VecX                lg, ug;
        std::vector<double> lg_mask, ug_mask;
    };

    // Splits the rows by the box layout in _box_rows and creates the HPIPM
    // structures for it
    void _build();
    void _free();

    // Splits the rows of stage k into boxes and general rows, with a box on
    // every state entry when the state is fixed
    void _split_rows(const uint k, const uint nu, const bool fixed_state);

    // Uploads the constraints of stage k, where E is null at the terminal
    // stage since it has no controls
    void _upload_constraints(const uint k, const MatX &D, const MatX *E,
                             const VecX &cl, const VecX &cu);

    // Reads the multipliers of stage k back into the rows they belong to
    void _get_multipliers(const uint k, Solution &sol) const;

    const QPData &_qp;
    Params        _params;
    Solution      _solution;

    double       _reg;
    const double _reg_max = 1e8;

    // The box layout the HPIPM structures were created for
    std::vector<std::vector<model::BoxRow>> _box_rows;
    std::vector<Stage>                      _stages;

    // Holds the multipliers HPIPM returns for a stage's lower and upper
    // sides, sized for the stage with the most constraints of one kind
    mutable VecX _lam;

    // HPIPM stores the state/control cross term as nu-by-nx, whereas QPData
    // stores it as nx-by-nu. Keep the transposed upload buffers persistent.
    std::vector<MatX> _S;

    int  _iter;
    int  _status;
    bool _qp_uploaded = false;
    bool _solved      = false;

    // HPIPM structures
    std::shared_ptr<hpipm_data> _hpipm_data;

    // Memory arenas (HPIPM uses custom allocators)
    void *_hpipm_dim_mem = nullptr;
    void *_hpipm_qp_mem  = nullptr;
    void *_hpipm_sol_mem = nullptr;
    void *_hpipm_arg_mem = nullptr;
    void *_hpipm_ws_mem  = nullptr;
};
}
}

#endif
