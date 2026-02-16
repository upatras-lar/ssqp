#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
uint StageConstraints::required_forces() const
{
    return 0;
}

std::string StageConstraints::force_frame(const uint i) const
{
    return "";
}

void StageConstraints::set_forces(const std::vector<Force> &forces) {}

double StageConstraints::violation(const VecX &x, const VecX &u) const
{
    VecX c = evaluate(x, u);

    return std::max(
      right_minus(c, lower_bound()).cwiseMin(0).lpNorm<Eigen::Infinity>(),
      right_minus(c, upper_bound()).cwiseMax(0).lpNorm<Eigen::Infinity>());
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

uint StageConstraints::constraint_diff_size() const
{
    return output_size();
}

VecX StageConstraints::right_minus(const VecX &x, const VecX &y) const
{
    return x - y;
}
}
}
}