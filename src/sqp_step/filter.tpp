#include "sqp_step/filter.hpp"

namespace ssqp
{
namespace step
{
template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
bool costFilter(const FilterFunction1 &ff1, const FilterFunction2 &ff2,
                const FilterFunction3 &ff3, const Point &previous,
                const Point &next)
{

    return ff1(next.x, next.u) < ff1(previous.x, previous.u) ||
           ff2(next.x, next.u) < ff2(previous.x, previous.u) ||
           ff3(next.x, next.u) < ff3(previous.x, previous.u);
}

template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
std::tuple<solvers::FlatVec, solvers::FlatVec, double> filter(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec &dU, const FilterFunction1 &ff1, const FilterFunction2 &ff2,
  const FilterFunction3 &ff3, const model::DynamicsContainer &dynamics,
  double inv_alpha_scale_factor, double min_alpha)
{
    double alpha = 1.0;

    step::Point a(x, u);
    step::Point offset(dX, dU);
    step::Point b(a.dynamics_add(alpha * offset, dynamics));

    while (!costFilter(ff1, ff2, ff3, a, b) && alpha >= min_alpha)
    {
        alpha /= inv_alpha_scale_factor;
        b      = a.dynamics_add(alpha * offset, dynamics);
    }

    return { b.x, b.u, alpha };
}

}
}