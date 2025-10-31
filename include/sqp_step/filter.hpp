#ifndef SSQP_FILTER_HPP
#define SSQP_FILTER_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{

template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
bool costFilter(const FilterFunction1 &ff1, const FilterFunction2 &ff2,
                const FilterFunction3 &ff3, const Point &previous,
                const Point &next);

template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
std::tuple<solvers::FlatVec, solvers::FlatVec, double> filter(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec &dU, const FilterFunction1 &ff1, const FilterFunction2 &ff2,
  const FilterFunction3 &ff3, double inv_alpha_scale_factor = 2,
  double min_alpha = 1e-5);
}
}

#include "src/sqp_step/filter.tpp"

#endif