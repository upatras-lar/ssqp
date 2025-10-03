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
  const Point &previous, const Point &next);

std::tuple<MatX, MatX, double> filter(
  MatX &x, MatX &u, MatX &dX, MatX &dU,
  const std::vector<std::function<double(const MatX &, const MatX &)>> costs,
  int inv_alpha_scale_factor = 2, double min_alpha = 1e-5);
}
}

#endif