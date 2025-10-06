#ifndef SSQP_SSQP_HPP
#define SSQP_SSQP_HPP

#include <iostream>
#include <vector>
#include <functional>

#include "types.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace solvers
{
class ssqp
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
        FlatVec x, u;
        // Equivalent vector sequences for lagrange multiplier guesses n
        FlatVec lambda, mu;
    };

    ssqp(const model::Model &model, const Params &params = Params::Default());

    SolverState solve(Guess &guess);

  protected:
    const std::vector<std::function<double(const FlatVec &, const FlatVec &)>>
      _filter_ls_costs;

    Params       _params;
    model::Model _model;
    Guess        _guess;

    FlatVec _prev_cost_dx, _prev_cost_du;
    FlatVec _cost_dx, _cost_du;

    uint _iter;
    uint _stale_iters;

    QPData _setup_qp();

    SolverState _check_termination();

    // Trajectory cost
    double _lf(const FlatVec &x, const FlatVec &u) const;

    // Gap violation
    double _gammaf(const FlatVec &x, const FlatVec &u) const;

    // Constraint violation
    double _cf(const FlatVec &x, const FlatVec &u) const;
};
}
}

#endif
