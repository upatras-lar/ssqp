#ifndef SSQP_SQP_STEP_MERIT_HPP
#define SSQP_SQP_STEP_MERIT_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction>
bool merit(const double mf_previous, const double directional_derivative,
           const Point &next, const double alpha, const MeritFunction &mf,
           const double armijo_beta = 1e-4);
}
}

#include "src/sqp_step/merit.tpp"

#endif