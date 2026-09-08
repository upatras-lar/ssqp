#include "sqp_step/filter.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace step
{
template<typename FilterFunction1, typename FilterFunction2,
         typename FilterFunction3>
bool filter(const std::tuple<double, double, double> &previous_metrics,
            const Point                              &next,
            const std::deque<std::tuple<double, double, double>> &filter_memory,
            const FilterFunction1 &ff1, const FilterFunction2 &ff2,
            const FilterFunction3              &ff3,
            std::tuple<double, double, double> &next_metrics)
{
    ZoneScopedN("SQP Step: filter");

    double v1 = ff1(next.x, next.u);
    double v2 = ff2(next.x, next.u);
    double v3 = ff3(next.x, next.u);

    next_metrics = { v1, v2, v3 };

    for (uint i = 0; i < filter_memory.size(); ++i)
    {
        if (v1 > std::get<0>(filter_memory[i]) &&
            v2 > std::get<1>(filter_memory[i]) &&
            v3 > std::get<2>(filter_memory[i]))
        {
            return false;
        }
    }

    return v1 <= std::get<0>(previous_metrics) ||
           v2 <= std::get<1>(previous_metrics) ||
           v3 <= std::get<2>(previous_metrics);
}

}
}