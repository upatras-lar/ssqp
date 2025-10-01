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
        double       rho                  = 0.1;
        const double alpha                = 1.6;
        const double sigma                = 1e-6;
        const uint   rho_scale_factor     = 2;
        const uint   rho_incr_factor      = 5;
        const uint   rho_decr_factor      = 5;
        const uint   rho_update_iter_freq = 25;
        const double tolerance_abs        = 1e-4;
        const double tolerance_rel        = 1e-3;

        static Params Default()
        {
            return {};
        }
    };

    struct Solution
    {
        MatX              dX, dU;
        MatX              dX_hat;
        std::vector<MatX> P;
        std::vector<VecX> p;
        MatX              z, y;
    };

    admm_qp(const QPData &qp, const csqp::Guess &guess,
            const Params &params = Params::Default());

    Solution solve();

  protected:
    QPData      _qp;
    csqp::Guess _guess;
    Params      _params;

    MatX _r_prim, _r_dual;

    uint _iter;

    Solution _solution;

    void _step(const MatX &dX_hat, const MatX &dU_hat, MatX &dX, MatX &dU,
               MatX &y, MatX &z, MatX &z_hat);
};
}
}

#endif
