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

std::vector<uint> StageConstraints::state_only_rows() const
{
    return {};
}

uint StageConstraints::constraint_diff_size() const
{
    return output_size();
}

VecX StageConstraints::right_minus(const VecX &x, const VecX &y) const
{
    return x - y;
}

void StageConstraints::add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                                    MatXRef dU) const
{
    gradient(x, u, _add_scratch_dX, _add_scratch_dU);
    dX += _add_scratch_dX;
    dU += _add_scratch_dU;
}

void StageConstraints::add_hessprod_xx(const VecX &x, const VecX &u,
                                       const VecX &mu, MatXRef Pxx) const
{
    hessprod_xx(x, u, mu, _add_scratch_xx);
    Pxx += _add_scratch_xx;
}

void StageConstraints::add_hessprod_xu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatXRef Pxu) const
{
    hessprod_xu(x, u, mu, _add_scratch_xu);
    Pxu += _add_scratch_xu;
}

void StageConstraints::add_hessprod_uu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatXRef Puu) const
{
    hessprod_uu(x, u, mu, _add_scratch_uu);
    Puu += _add_scratch_uu;
}
}
}
}