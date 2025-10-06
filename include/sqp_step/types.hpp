#ifndef SSQP_STEP_TYPES_HPP
#define SSQP_STEP_TYPES_HPP

#include <vector>
#include <functional>
#include <cassert>

#include "include/types.hpp"

namespace ssqp
{
namespace step
{
struct Point
{
    solvers::FlatVec x, u;

    Point(const solvers::FlatVec &x, const solvers::FlatVec &u);

    Point &operator=(const Point &p);

    Point operator+(const Point &p) const;

    Point operator-(const Point &p) const;

    friend Point operator*(double val, const Point &p);

    Point operator/(double val);

    Point &operator+=(const Point &p);

    double infNorm();
};

Point operator*(double val, const Point &p);
}
}
#endif