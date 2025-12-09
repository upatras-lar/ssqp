#ifndef SSQP_SQP_STEP_FILTER_HPP
#define SSQP_SQP_STEP_FILTER_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
bool filter(const Point &previous, const Point &next,
            const std::deque<std::tuple<double, double, double>> filter_memory,
            const FilterFunction1 &ff1, const FilterFunction2 &ff2,
            const FilterFunction3 &ff3);
}
}

#include "src/sqp_step/filter.tpp"

#endif