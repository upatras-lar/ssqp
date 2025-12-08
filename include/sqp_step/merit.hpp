#ifndef SSQP_SQP_STEP_MERIT_HPP
#define SSQP_SQP_STEP_MERIT_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction>
bool merit(const Point &previous, const Point &next, const MeritFunction &mf);
}
}

#include "src/sqp_step/merit.tpp"

#endif