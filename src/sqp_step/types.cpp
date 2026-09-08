#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
Point::Point(const solvers::FlatVec &x, const solvers::FlatVec &u, bool copy)
  : x(x, copy)
  , u(u, copy)
{
}

Point &Point::operator=(const Point &p)
{
    x = p.x;
    u = p.u;
    return *this;
}

void Point::dynamics_add_scaled(double alpha, const Point &offset,
                                const model::DynamicsContainer &dynamics,
                                Point &scaled_offset, Point &out) const
{
    assert(x.stages() == offset.x.stages() && u.stages() == offset.u.stages() &&
           "Point dynamics_add_scaled: stage mismatch");

    scaled_offset.x.vec.noalias() = alpha * offset.x.vec;
    scaled_offset.u.vec.noalias() = alpha * offset.u.vec;

    for (uint k = 0; k < x.stages(); ++k)
    {
        out.x.stage(k) =
          dynamics[k]->state_right_plus(x.stage(k), scaled_offset.x.stage(k));
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        out.u.stage(k) = u.stage(k) + scaled_offset.u.stage(k);
    }
}

double Point::infNorm()
{
    double norm1 = x.vec.lpNorm<Eigen::Infinity>();
    double norm2 = u.vec.lpNorm<Eigen::Infinity>();

    return std::max(norm1, norm2);
}
}
}