#include "sqp_step/filter.hpp"

namespace ssqp
{
namespace step
{
template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
bool filter(const Point &previous, const Point &next,
            const FilterFunction1 &ff1, const FilterFunction2 &ff2,
            const FilterFunction3 &ff3)
{

    return ff1(next.x, next.u) < ff1(previous.x, previous.u) ||
           ff2(next.x, next.u) < ff2(previous.x, previous.u) ||
           ff3(next.x, next.u) < ff3(previous.x, previous.u);
}

}
}