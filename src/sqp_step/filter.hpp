#include "types.hpp"
#include "sqp_step/types.hpp"

#ifndef SSQP_FILTER_HPP
#define SSQP_FILTER_HPP

namespace ssqp
{
namespace step
{

bool costFilter(
  const std::vector<std::function<double(const MatX &, const MatX &)>> &costs,
  const Point &smaller, const Point &greater)
{

    for (const auto &f : costs)
    {
        double fs = f(smaller.x, smaller.u);
        double fg = f(greater.x, greater.u);

        if (fs < fg)
        {
            return true;
        }
    }

    return false;
}

std::pair<MatX, MatX> filter(
  MatX &x, MatX &u, MatX &dX, MatX &dU,
  const std::vector<std::function<double(const MatX &, const MatX &)>> costs,
  int inv_alpha_scale_factor = 2, double min_alpha = 1e-5)
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

    return { b.x, b.u };
}
}
}

#endif