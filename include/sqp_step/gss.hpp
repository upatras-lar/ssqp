#ifndef SSQP_GSS_HPP
#define SSQP_GSS_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
bool costSum(const std::vector<std::function<double(
               const solvers::FlatVec &, const solvers::FlatVec &)>> &costs,
             const std::vector<double> &weights, const Point &smaller,
             const Point &greater);

// Golden section search
std::pair<solvers::FlatVec, solvers::FlatVec> gss(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec                                                   &dU,
  const std::vector<std::function<double(const solvers::FlatVec &,
                                         const solvers::FlatVec &)>> &costs,
  const std::vector<double> &weights, double tolerance = 1e-5);
}
}

#endif