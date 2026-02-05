#ifndef SSQP_SOLVERS_ADMM_QP_HPP
#define SSQP_SOLVERS_ADMM_QP_HPP

#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"

namespace ssqp
{
namespace solvers
{
class admm_qp
{
  public:
    struct Params
    {
        // Parameters
        double init_rho = 0.1;
        double alpha    = 1.6;
        double sigma    = 1e-6;

        // Rho update
        bool   rho_update             = true;
        double rho_min                = 1e-6;
        double rho_max                = 1e8;
        double adaptive_rho_tolerance = 5;
        uint   rho_update_iter_freq   = 25;

        // Termination parameters
        uint   max_iter               = 1'000;
        double abs_tolerance          = 1e-3; // Absolute tolerance
        double abs_tolerance_final    = -1.0;
        double rel_tolerance          = 1e-4; // Relative tolerance
        double rel_tolerance_final    = -1.0;
        uint   tolerance_update_steps = 0;
        double primal_infs_tolerance  = 1e-6; // Primal infeasibility tolerance

        bool warm_start = true;
        bool reset_rho  = true;
        bool reset_y    = true;

        static Params Default()
        {
            return {};
        }
    };

    struct Solution
    {
        FlatVec           dx, du;
        std::vector<MatX> P;
        std::vector<VecX> p;
        FlatVec           z, y;

        Solution(const model::Dimensions &dims);
    };

    admm_qp(const QPData &qp, const Params &params = Params::Default());

    void reset();

    void increaseRegularization();

    SolverState solve(Solution &sol);

    uint iterations() const;

  protected:
    const QPData &_qp;
    Params        _params;
    Solution      _solution;

    FlatVec      _rho, _rho_inv;
    double       _rho_common;
    const double _rho_min = 1e-6;
    bool         _rho_changed;

    // Norms of residuals
    double _norm_primal, _norm_dual;
    double _norm_primal_rel, _norm_dual_rel;

    // Primal infeasibility criteria
    double _primal_infs_norm, _primal_infs_support;
    double _primal_infs_tol_scaled;

    uint _iter;

    uint _reg_incr;

    double _abs_tolerance_init;
    double _rel_tolerance_init;

    // Calculates residual norms
    // r_prim = Dx - z
    // r_dual = Q(z - z_prev)
    void _calculate_residuals(const FlatVec &x_prev, const FlatVec &u_prev,
                              const FlatVec &z_prev, const FlatVec &y_prev);

    SolverState _check_termination() const;

    void _update_rho();

    void _step(const FlatVec &dx_hat, const FlatVec &du_hat);
};
}
}

#endif
