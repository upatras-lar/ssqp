#ifndef SSQP_ADMM_QP_HPP
#define SSQP_ADMM_QP_HPP

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
        const double init_rho = 0.1;
        const double alpha    = 1.6;
        const double sigma    = 1e-6;

        // Rho update
        const double rho_min_                = 1e-6;
        const double rho_max_                = 1e3;
        const double adaptive_rho_tolerance_ = 5;
        const uint   rho_update_iter_freq    = 25;

        // Termination parameters
        const uint   max_iter      = 10'000;
        const double abs_tolerance = 1e-3; // Absolute tolerance
        const double rel_tolerance = 1e-4; // Relative tolerance
        const double primal_infs_tolerance =
          1e-6; // Primal infeasibility tolerance

        bool warm_start = true;

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

    SolverState solve(Solution &sol);

  protected:
    QPData   _qp;
    Params   _params;
    Solution _solution;

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
