#include "sqp_step/gss.hpp"

namespace ssqp
{
namespace step
{
bool costSum(const std::vector<FilterFunction> &costs,
             const std::vector<double> &weights, const Point &smaller,
             const Point &greater)
{
    assert(costs.size() == weights.size() &&
           "Cost and weight vector size mismatch");
    double smaller_sum = 0.0;
    double greater_sum = 0.0;

    for (uint i = 0; i < costs.size(); ++i)
    {
        smaller_sum += weights[i] * costs[i](smaller.x, smaller.u);
        greater_sum += weights[i] * costs[i](greater.x, greater.u);
    }

    return smaller_sum < greater_sum;
}

std::pair<solvers::FlatVec, solvers::FlatVec> gss(
  solvers::FlatVec &x, solvers::FlatVec &u, solvers::FlatVec &dX,
  solvers::FlatVec &dU, const std::vector<FilterFunction> &costs,
  const std::vector<double> &weights, double tolerance)
{
    static const double invphi = (std::sqrt(5) - 1) / 2;

    Point a(x, u);
    Point b(x + dX, u + dU);

    while ((b - a).infNorm() > tolerance)
    {
        Point c = b - invphi * (b - a);
        Point d = a + invphi * (b - a);

        if (costSum(costs, weights, c, d))
        {
            b = d;
            d = c;
            c = b - invphi * (b - a);
        }
        else
        {
            a = c;
            c = d;
            d = a + invphi * (b - a);
        }
    }

    a = (a + b) / 2;
    return { a.x, a.u };
}
}
}