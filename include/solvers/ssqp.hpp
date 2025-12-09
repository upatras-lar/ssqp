#ifndef SSQP_SOLVERS_SSQP_HPP
#define SSQP_SOLVERS_SSQP_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <variant>
#include <deque>

#include "types.hpp"
#include "sqp_step/step.hpp"
#include "sqp_step/filter.hpp"
#include "sqp_step/merit.hpp"
#include "model/Model.hpp"
#include "solvers/admm_qp.hpp"

#include "external/nlohmann/json.hpp"

namespace ssqp
{
namespace solvers
{
class ssqp
{
  public:
    struct Params
    {
        uint max_iter = 10'000;
        // Tolerance for lagrangian jacobian norms
        double tolerance = 1e-3;

        // Loose success
        bool loose_success = true;
        // Tolerance for change in cost jacobian norms
        double loose_success_tolerance = tolerance / 10;
        // Number of iterations for stale cost jacobian norms
        uint loose_success_stale_iters = 5;

        // Line-search parameters
        bool use_filter  = true;
        uint filter_size = 10;

        double merit_mu_gamma = 1.0;
        double merit_mu_c     = 1.0;

        double inv_alpha_scale_factor = 2.0;
        double min_alpha              = 1e-4;

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

        Guess(const model::Dimensions &dims, const model::Dimensions &lin_dims);
    };

    ssqp(model::Model &model, const Params &params = Params::Default(),
         const admm_qp::Params &admm_params = admm_qp::Params::Default());

    SolverState solve(Guess &guess);

    bool write(const std::string &filename);

  protected:
    Params          _params;
    admm_qp::Params _admm_params;
    model::Model    _model;
    Guess           _guess;

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

    std::deque<std::tuple<double, double, double>> _filter_memory;
};
}
}

#endif
