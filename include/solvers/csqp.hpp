#include <iostream>
#include <vector>
#include <functional>

#include "types.hpp"

#ifndef SSQP_CSQP_HPP
#define SSQP_CSQP_HPP

namespace ssqp
{
namespace solvers
{
class csqp
{
  public:
    struct Params
    {
        const uint max_iter = 10'000;
        // Tolerance for lagrangian jacobian norms
        const double tolerance = 1e-4;
        // Tolerance for change in cost jacobian norms
        const double loose_success_tolerance = 1e-5;
        // Number of iterations for stale cost jacobian norms
        const uint loose_success_stale_iters = 5;

        // Whether to warm start ADMM with an unconstrained LQR solution
        bool warm_start_admm = true;

        static Params Default()
        {
            return {};
        }
    };

    struct Guess
    {
        // Vector sequences indexed over time horizon for state and control
        // guesses
        // n
        // Matrix x is of dimensions nx x (H + 1) (starting at x_0 and ending at
        // x_H)
        // Matrix u is of dimensions nu x H (starting at u_0 and ending at
        // u_(H-1))
        MatX x, u;
        // Equivalent vector sequences for lagrange multiplier guesses n
        // Matrix lambda is of dimensions nx x H (starting at 1 and ending at H)
        // Matrix mu is of dimensions nc x (H + 1) (starting at 0 and ending at
        // H) (nc: number of constraints)
        MatX lambda, mu;
    };

    csqp(const OCP &ocp, const Params &params = Params::Default());

    SolverState solve(Guess &guess);

  protected:
    const uint _horizon;
    const uint _nx, _nu, _nc;

    const std::vector<std::function<double(const MatX &, const MatX &)>>
      _filter_ls_costs;

    Params _params;
    OCP    _ocp;
    Guess  _guess;

    MatX _prev_cost_dx, _prev_cost_du;
    MatX _cost_dx, _cost_du;

    uint _iter;
    uint _stale_iters;

    QPData _setup_qp();

    SolverState _check_termination();

    // Trajectory cost
    double _lf(const MatX &x, const MatX &u) const;

    // Gap violation
    double _gammaf(const MatX &x, const MatX &u) const;

    // Constraint violation
    double _cf(const MatX &x, const MatX &u) const;
};
}
}

#endif
