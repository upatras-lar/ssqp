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
        int max_iter = 1'000;

        double tol_stat = 1e-3;
        double tol_eq   = 1e-3;
        double tol_ineq = 1e-3;
        double tol_comp = 1e-3;

        bool warm_start = true;

        static Params Default()
        {
            return {};
        }
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
