#include "sqp_step/merit.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction>
bool merit(const double mf_previous, const double directional_derivative,
           const Point &next, const double alpha, const MeritFunction &mf,
           const double armijo_beta)
{
    ZoneScopedN("SQP step: merit");

    return mf(next.x, next.u) <
           mf_previous + armijo_beta * alpha * directional_derivative;
}
}
}