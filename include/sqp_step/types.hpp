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

    Point(const solvers::FlatVec &x, const solvers::FlatVec &u);

    Point &operator=(const Point &p);

    Point dynamics_add(const Point                    &p,
                       const model::DynamicsContainer &dynamics) const;

    double infNorm();

    friend Point operator*(double val, const Point &p);
};

Point operator*(double val, const Point &p);
}
}
#endif