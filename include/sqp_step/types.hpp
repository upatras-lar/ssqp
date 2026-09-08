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

struct Point
{
    solvers::FlatVec x, u;

    // If copy is true, x/u are copied from the given vectors
    // Otherwise, a zero-initialized Point of the same size is constructed
    Point(const solvers::FlatVec &x, const solvers::FlatVec &u,
          bool copy = true);

    Point &operator=(const Point &p);

    // Writes (this ⊕ alpha * offset) into `out`, using the dynamics' manifold
    // right-plus for the state part
    // `scaled_offset` is scratch storage for alpha * offset, scaled as a
    // single whole-vector pass rather than per-stage so it stays vectorized
    // Does not allocate, `out`/`scaled_offset` must already be sized like
    // `this`/`offset`
    void dynamics_add_scaled(double alpha, const Point &offset,
                             const model::DynamicsContainer &dynamics,
                             Point &scaled_offset, Point &out) const;

    double infNorm();
};
}
}
#endif