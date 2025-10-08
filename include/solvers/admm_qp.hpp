#ifndef SSQP_ADMM_QP_HPP
#define SSQP_ADMM_QP_HPP

#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "ssqp.hpp"

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
        const uint   max_iter      = 100'000;
        const double abs_tolerance = 1e-6; // Absolute tolerance
        const double rel_tolerance = 1e-6; // Relative tolerance
        const double primal_infeasibility_tolerance =
          1e-9; // Primal infeasibility tolerance

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
    };

    admm_qp(const QPData &qp, const ssqp::Guess &guess,
            const Params &params = Params::Default());

    SolverState solve(Solution &sol);

  protected:
    QPData      _qp;
    ssqp::Guess _guess;
    Params      _params;

    FlatVec      _rho, _rho_inv;
    double       _rho_common;
    const double _rho_min = 1e-6;

    // Norms of residuals
    double _norm_primal = 0, _norm_dual = 0;
    double _norm_primal_rel = 0, _norm_dual_rel = 0;

    uint _iter;

    Solution _solution;

    // Calculates residuals and their relative parts
    // r_prim = Dx - z
    // r_dual = Q(z - z_prev)
    // Relative parts include dX and dU variable segments
    void _calculate_residuals(const FlatVec &z_prev);

    SolverState _check_termination() const;

    void _update_rho();

    void _step(const FlatVec &dx_hat, const FlatVec &du_hat);
};
}
}

#endif
