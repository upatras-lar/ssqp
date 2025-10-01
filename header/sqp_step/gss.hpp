#include "types.hpp"
#include "sqp_step/types.hpp"

#ifndef SSQP_GSS_HPP
#define SSQP_GSS_HPP

namespace ssqp
{
namespace step
{
bool costSum(
  const std::vector<std::function<double(const MatX &, const MatX &)>> &costs,
  const std::vector<double> &weights, const Point &smaller,
  const Point &greater);

// Golden section search
std::pair<MatX, MatX> gss(
  MatX &x, MatX &u, MatX &dX, MatX &dU,
  const std::vector<std::function<double(const MatX &, const MatX &)>> costs,
  const std::vector<double> &weights, double tolerance = 1e-5);
}
}

#endif