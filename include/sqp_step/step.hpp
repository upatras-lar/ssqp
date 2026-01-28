#ifndef SSQP_SQP_STEP_STEP_HPP
#define SSQP_SQP_STEP_STEP_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
template<typename SelectionMethod>
std::tuple<solvers::FlatVec, solvers::FlatVec, double> step(
  const SelectionMethod &selection, solvers::FlatVec &x, solvers::FlatVec &u,
  const solvers::FlatVec &dX, const solvers::FlatVec &dU,
  const model::DynamicsContainer &dynamics, double inv_alpha_scale_factor,
  double min_alpha);
}
}

#include "src/sqp_step/step.tpp"

#endif