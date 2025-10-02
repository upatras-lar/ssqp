#include <cassert>
#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "csqp.hpp"
#include "model/constraint/AbstractConstraint.hpp"
#include "model/cost/AbstractCost.hpp"
#include "model/dynamics/AbstractDynamics.hpp"

#ifndef SSQP_ADMM_QP_HPP
#define SSQP_ADMM_QP_HPP

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
        const uint rho_scale_factor     = 2;
        const uint rho_incr_factor      = 5;
        const uint rho_decr_factor      = 5;
        const uint rho_update_iter_freq = 25;

        // Termination parameters
        const uint   max_iter      = 100'000;
        const double abs_tolerance = 1e-4; // Absolute tolerance
        const double rel_tolerance = 1e-3; // Relative tolerance
        const double primal_infeasibility_tolerance =
          1e-6; // Primal infeasibility tolerance

        static Params Default()
        {
            return {};
        }
    };

    struct Solution
    {
        MatX              dX, dU;
        std::vector<MatX> P;
        std::vector<VecX> p;
        MatX              z, y;
    };

    admm_qp(const QPData &qp, const csqp::Guess &guess,
            const Params &params = Params::Default());

    SolverState solve(Solution &sol);

  protected:
    QPData      _qp;
    csqp::Guess _guess;
    Params      _params;

    std::vector<DMatX> _rho, _rho_inv;
    const double       _rho_min = 1e-6;

    // Relative parts of residuals
    MatX _Dx, _rQTz, _rQTzp;
    // Residuals
    MatX _r_prim, _r_dual;

    uint _iter;

    Solution _solution;

    // Calculates residuals and their relative parts
    // r_prim = Dx - z
    // r_dual = Q(z - z_prev)
    // Relative parts include dX and dU variable segments
    void _calculate_residuals(const MatX &z_prev);

    SolverState _check_termination() const;

    void _update_rho();

    void _step(const MatX &dX_hat, const MatX &dU_hat);
};
}
}

#endif
