#ifndef SSQP_SQP_STEP_MERIT_HPP
#define SSQP_SQP_STEP_MERIT_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction, typename MeritGradFunction>
bool merit(const Point &previous, const Point &next, const Point &offset,
           const double alpha, const MeritFunction &mf,
           const MeritGradFunction &dmf);
}
}

#include "src/sqp_step/merit.tpp"

#endif