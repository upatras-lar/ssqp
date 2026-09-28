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
        bool square_root_riccati = true;
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
    void        reset() override;
    void        increaseRegularization() override;
    SolverState solve(QPSolver::Solution &sol) override;
    uint        iterations() const override;

    // Riccati gain of the last interior point iteration, where the barrier
    // stiffens the active constraints
    MatX feedbackGain(const uint k) const override;

  protected:
    const QPData &_qp;
    Params        _params;
    Solution      _solution;

    double       _reg;
    const double _reg_max = 1e8;

    // Copies of lower/upper constraint vectors
    // Required since infinity bounds must be replaced, but the actual QP
    // must not be altered
    std::vector<VecX> _cl, _cu;

    // HPIPM stores the state/control cross term as nu-by-nx, whereas QPData
    // stores it as nx-by-nu. Keep the transposed upload buffers persistent.
    std::vector<MatX> _S;

    // Masks to ignore open constraints
    std::vector<std::vector<double>> _cl_masks;
    std::vector<std::vector<double>> _cu_masks;

    int  _iter;
    int  _status;
    bool _qp_uploaded = false;
    bool _solved      = false;

    // HPIPM structures
    std::shared_ptr<hpipm_data> _hpipm_data;

    // Memory arenas (HPIPM uses custom allocators)
    void *_hpipm_dim_mem;
    void *_hpipm_qp_mem;
    void *_hpipm_sol_mem;
    void *_hpipm_arg_mem;
    void *_hpipm_ws_mem;
};
}
}

#endif
