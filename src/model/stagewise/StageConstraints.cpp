#include "model/stagewise/StageConstraints.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
uint StageConstraints::TOTAL = 0;

StageConstraints::StageConstraints()
  : Function("Constraints" + std::to_string(StageConstraints::TOTAL++))
{
}

StageConstraints::StageConstraints(const std::string &name)
  : Function(name.empty()
               ? "Constraints" + std::to_string(StageConstraints::TOTAL++)
               : name)
{
}

void StageConstraints::set_enabled(const bool enabled)
{
    _enabled = enabled;
}

bool StageConstraints::enabled() const
{
    return _enabled;
}

uint StageConstraints::required_forces() const
{
    return 0;
}

std::string StageConstraints::force_frame(const uint i) const
{
    return "";
}

void StageConstraints::set_forces(
  const std::vector<std::shared_ptr<Force>> &forces)
{
}

const VecX &StageConstraints::lower_bound() const
{
    if (static_cast<uint>(_default_lb.size()) != output_size())
    {
        _default_lb.setConstant(output_size(),
                                -std::numeric_limits<double>::infinity());
    }

    return _default_lb;
}

const VecX &StageConstraints::upper_bound() const
{
    if (static_cast<uint>(_default_ub.size()) != output_size())
    {
        _default_ub.setConstant(output_size(),
                                std::numeric_limits<double>::infinity());
    }

    return _default_ub;
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