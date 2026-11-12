#ifndef SSQP_SQP_STEP_TYPES_HPP
#define SSQP_SQP_STEP_TYPES_HPP

#include <vector>
#include <functional>
#include <cassert>
#include <memory>

#include "include/types.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace step
{

// A trajectory, or a step of one, viewed in place
struct Point
{
    const solvers::FlatVec &x, &u;

    // Writes (this ⊕ alpha * offset) into `x_out`/`u_out`, using the
    // dynamics' manifold right-plus for the state part
    // `scaled_dx`/`scaled_du` are scratch storage for alpha * offset, scaled
    // as a single whole-vector pass rather than per-stage so it stays
    // vectorized
    // Does not allocate, the outputs and the scratch must already be sized
    // like `this` and `offset`
    void dynamics_add_scaled(double alpha, const Point &offset,
                             const model::DynamicsContainer &dynamics,
                             solvers::FlatVec               &scaled_dx,
                             solvers::FlatVec               &scaled_du,
                             solvers::FlatVec               &x_out,
                             solvers::FlatVec               &u_out) const;

    double infNorm();
};
}
}
#endif