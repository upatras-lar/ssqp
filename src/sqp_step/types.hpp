#include <vector>
#include <functional>
#include <cassert>

#include "src/types.hpp"

#ifndef SSQP_STEP_TYPES_HPP
#define SSQP_STEP_TYPES_HPP
namespace ssqp
{
namespace step
{
struct Point
{
    MatX x, u;

    Point(const MatX &x, const MatX &u) : x(x), u(u) {}

    Point &operator=(const Point &p)
    {
        x = p.x;
        u = p.u;
        return *this;
    }

    Point operator+(const Point &p) const
    {
        return Point(x + p.x, u + p.u);
    }

    Point operator-(const Point &p) const
    {
        return Point(x - p.x, u - p.u);
    }

    friend Point operator*(double val, const Point &p);

    Point operator/(double val)
    {
        return Point(x / val, u / val);
    }

    Point &operator+=(const Point &p)
    {
        x += p.x;
        u += p.u;

        return *this;
    }

    double infNorm()
    {
        double norm1 = x.lpNorm<Eigen::Infinity>();
        double norm2 = u.lpNorm<Eigen::Infinity>();

        return std::max(norm1, norm2);
    }
};

Point operator*(double val, const Point &p)
{
    return Point(val * p.x, val * p.u);
}

}
}
#endif