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
        const uint   max_iter  = 10'000;
        const double tolerance = 1e-4;

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
    const int _horizon;
    const int _nx, _nu, _nc;

    const std::vector<std::function<double(const MatX &, const MatX &)>>
      _filter_ls_costs;

    Params _params;
    OCP    _ocp;
    Guess  _guess;

    uint _iter;

    QPData _setup_qp();

    SolverState _check_termination() const;

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
