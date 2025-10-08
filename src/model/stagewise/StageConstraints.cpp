#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
bool StageConstraints::is_satisfied(const VecX &x, const VecX &u,
                                    const double tolerance) const
{
    VecX c = evaluate(x, u);

    return (c - lower_bound()).cwiseMin(0).lpNorm<Eigen::Infinity>() <
             tolerance &&
           (c - upper_bound()).cwiseMax(0).lpNorm<Eigen::Infinity>() <
             tolerance;
}

VecX StageConstraints::lower_bound() const
{
    VecX l(output_size());
    l.setConstant(-std::numeric_limits<double>::infinity());

    return l;
}

VecX StageConstraints::upper_bound() const
{
    VecX u(output_size());
    u.setConstant(std::numeric_limits<double>::infinity());

    return u;
}
}
}
}