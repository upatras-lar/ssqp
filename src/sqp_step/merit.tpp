#include "sqp_step/merit.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction, typename MeritGradFunction>
bool merit(const Point &previous, const Point &next, const Point &offset,
           const double alpha, const MeritFunction &mf,
           const MeritGradFunction &dmf, const double armijo_beta)
{
    ZoneScopedN("SQP step: merit");

    solvers::FlatVec dx(offset.x, false), du(offset.u, false);
    dmf(previous.x, previous.u, dx, du);

    return mf(next.x, next.u) < mf(previous.x, previous.u) +
                                  armijo_beta * alpha *
                                    (dx.vec.transpose() * offset.x.vec +
                                     du.vec.transpose() * offset.u.vec)[0];
}
}
}