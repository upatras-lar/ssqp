#include "sqp_step/merit.hpp"

namespace ssqp
{
namespace step
{
template<typename MeritFunction>
bool merit(const Point &previous, const Point &next, const MeritFunction &mf)
{
    return mf(next.x, next.u) < mf(previous.x, previous.u);
}
}
}