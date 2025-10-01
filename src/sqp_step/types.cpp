#include "sqp_step/types.hpp"

namespace ssqp
{
namespace step
{
Point::Point(const MatX &x, const MatX &u) : x(x), u(u) {}

Point &Point::operator=(const Point &p)
{
    x = p.x;
    u = p.u;
    return *this;
}

Point Point::operator+(const Point &p) const
{
    return Point(x + p.x, u + p.u);
}

Point Point::operator-(const Point &p) const
{
    return Point(x - p.x, u - p.u);
}

Point Point::operator/(double val)
{
    return Point(x / val, u / val);
}

Point &Point::operator+=(const Point &p)
{
    x += p.x;
    u += p.u;

    return *this;
}

double Point::infNorm()
{
    double norm1 = x.lpNorm<Eigen::Infinity>();
    double norm2 = u.lpNorm<Eigen::Infinity>();

    return std::max(norm1, norm2);
}

Point operator*(double val, const Point &p)
{
    return Point(val * p.x, val * p.u);
}
}
}