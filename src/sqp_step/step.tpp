#include "sqp_step/step.hpp"

namespace ssqp
{
namespace step
{
template<typename SelectionMethod>
std::tuple<solvers::FlatVec, solvers::FlatVec, double> step(
  const SelectionMethod &selection, solvers::FlatVec &x, solvers::FlatVec &u,
  solvers::FlatVec &dX, solvers::FlatVec &dU,
  const model::DynamicsContainer &dynamics, double inv_alpha_scale_factor,
  double min_alpha)
{
    double alpha = 1.0;

    step::Point a(x, u);
    step::Point offset(dX, dU);
    step::Point b(a.dynamics_add(alpha * offset, dynamics));

    while (!selection(a, b) && alpha >= min_alpha)
    {
        alpha /= inv_alpha_scale_factor;
        b      = a.dynamics_add(alpha * offset, dynamics);
    }

    return { b.x, b.u, alpha };
}
}
}