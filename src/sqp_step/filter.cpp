#include "sqp_step/filter.hpp"

namespace ssqp
{
namespace step
{
bool costFilter(const std::vector<std::function<double(
                  const solvers::FlatVec &, const solvers::FlatVec &)>> &costs,
                const Point &previous, const Point &next)
{
    for (const auto &f : costs)
    {
        double fp = f(previous.x, previous.u);
        double fn = f(next.x, next.u);

        if (fn < fp)
        {
            return true;
        }
    }

    return false;
}

std::tuple<solvers::FlatVec, solvers::FlatVec, double> filter(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec &dU,
  const std::vector<
    std::function<double(const solvers::FlatVec &, const solvers::FlatVec &)>>
       costs,
  uint inv_alpha_scale_factor, double min_alpha)
{
    double alpha = 1.0;

    step::Point a(x, u);
    step::Point offset(dX, dU);
    step::Point b(a + alpha * offset);

    while (!costFilter(costs, a, b) && alpha >= min_alpha)
    {
        alpha /= inv_alpha_scale_factor;
        b      = a + alpha * offset;
    }

    return { b.x, b.u, alpha };
}

}
}