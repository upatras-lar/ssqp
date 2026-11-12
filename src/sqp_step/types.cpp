#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
void Point::dynamics_add_scaled(double alpha, const Point &offset,
                                const model::DynamicsContainer &dynamics,
                                solvers::FlatVec               &scaled_dx,
                                solvers::FlatVec               &scaled_du,
                                solvers::FlatVec               &x_out,
                                solvers::FlatVec               &u_out) const
{
    assert(x.stages() == offset.x.stages() && u.stages() == offset.u.stages() &&
           "Point dynamics_add_scaled: stage mismatch");

    scaled_dx.vec.noalias() = alpha * offset.x.vec;
    scaled_du.vec.noalias() = alpha * offset.u.vec;

    for (uint k = 0; k < x.stages(); ++k)
    {
        dynamics[k]->state_right_plus_into(x.stage(k),
                                           scaled_dx.stage(k),
                                           x_out.stage(k));
    }

    for (uint k = 0; k < u.stages(); ++k)
    {
        u_out.stage(k) = u.stage(k) + scaled_du.stage(k);
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