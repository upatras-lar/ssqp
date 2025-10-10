#ifndef SSQP_FILTER_HPP
#define SSQP_FILTER_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{

bool costFilter(const std::vector<FilterFunction> &costs, const Point &previous,
                const Point &next);

std::tuple<solvers::FlatVec, solvers::FlatVec, double> filter(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec &dU, const std::vector<FilterFunction> costs,
  uint inv_alpha_scale_factor = 2, double min_alpha = 1e-5);
}
}

#endif