#ifndef SSQP_SQP_STEP_STEP_HPP
#define SSQP_SQP_STEP_STEP_HPP

#include "types.hpp"
#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
// `scaled_dX`/`scaled_dU` are scratch sized like `dX`/`dU`, and
// `x_out`/`u_out` must be sized like `x`/`u`
template<typename SelectionMethod>
double step(const SelectionMethod &selection, const solvers::FlatVec &x,
            const solvers::FlatVec &u, const solvers::FlatVec &dX,
            const solvers::FlatVec         &dU,
            const model::DynamicsContainer &dynamics,
            double inv_alpha_scale_factor, double min_alpha,
            solvers::FlatVec &scaled_dX, solvers::FlatVec &scaled_dU,
            solvers::FlatVec &x_out, solvers::FlatVec &u_out);
}
}

#include "src/sqp_step/step.tpp"

#endif