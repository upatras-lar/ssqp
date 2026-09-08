#include "sqp_step/step.hpp"

#include "tracy/Tracy.hpp"

namespace ssqp
{
namespace step
{
template<typename SelectionMethod>
double step(const SelectionMethod &selection, const solvers::FlatVec &x,
            const solvers::FlatVec &u, const solvers::FlatVec &dX,
            const solvers::FlatVec         &dU,
            const model::DynamicsContainer &dynamics,
            double inv_alpha_scale_factor, double min_alpha,
            solvers::FlatVec &x_out, solvers::FlatVec &u_out)
{
    ZoneScopedN("SQP Step: step");

    double alpha = 1.0;

    step::Point a(x, u);
    step::Point offset(dX, dU);
    step::Point b(x, u, false);

    a.dynamics_add_scaled(alpha, offset, dynamics, b);

    while (!selection(a, b, offset, alpha) && alpha >= min_alpha)
    {
        alpha /= inv_alpha_scale_factor;
        a.dynamics_add_scaled(alpha, offset, dynamics, b);
    }

    x_out = b.x;
    u_out = b.u;

    return alpha;
}
}
}