#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
Point::Point(const solvers::FlatVec &x, const solvers::FlatVec &u) : x(x), u(u)
{
}

Point &Point::operator=(const Point &p)
{
    x = p.x;
    u = p.u;
    return *this;
}

Point Point::dynamics_add(const Point                    &p,
                          const model::DynamicsContainer &dynamics) const
{
    assert(x.stages() == p.x.stages() && u.stages() == p.u.stages() &&
           "FlatVec x+y: stage mismatch");
    solvers::FlatVec xn(x, false), un(u, false);

    for (uint k = 0; k < x.stages(); ++k)
    {
        xn.stage(k) = dynamics[k]->state_right_plus(x.stage(k), p.x.stage(k));
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        un.stage(k) = u.stage(k) + p.u.stage(k);
    }

    return Point(xn, un);
}

double Point::infNorm()
{
    double norm1 = x.vec.lpNorm<Eigen::Infinity>();
    double norm2 = u.vec.lpNorm<Eigen::Infinity>();

    return std::max(norm1, norm2);
}

Point operator*(double val, const Point &p)
{
    return Point(val * p.x, val * p.u);
}
}
}