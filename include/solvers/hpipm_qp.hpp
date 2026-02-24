#ifndef SSQP_SOLVERS_HPIPM_QP_HPP
#define SSQP_SOLVERS_HPIPM_QP_HPP

extern "C"
{
#include <hpipm_d_ocp_qp.h>
#include <hpipm_d_ocp_qp_dim.h>
#include <hpipm_d_ocp_qp_ipm.h>
#include <hpipm_d_ocp_qp_sol.h>
#include <hpipm_d_ocp_qp_utils.h>
}

#include "solvers/qp.hpp"

namespace ssqp
{
namespace solvers
{

class hpipm_qp : public QPSolver
{
  public:
    struct Params
    {
        // HPIPM mode
        hpipm_mode mode = ROBUST;
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

    // Masks to ignore open constraints
    std::vector<std::vector<double>> _cl_masks;
    std::vector<std::vector<double>> _cu_masks;

    int _iter;
    int _status;

    // HPIPM structures
    d_ocp_qp_dim     _hpipm_qp_dim;
    d_ocp_qp         _hpipm_qp;
    d_ocp_qp_sol     _hpipm_qp_sol;
    d_ocp_qp_ipm_arg _hpipm_qp_arg;
    d_ocp_qp_ipm_ws  _hpipm_qp_ws;

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
