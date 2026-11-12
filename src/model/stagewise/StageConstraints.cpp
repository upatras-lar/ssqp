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

void StageConstraints::state_only_rows(std::vector<uint> &out) const
{
    out.clear();
}

void StageConstraints::box_rows(std::vector<BoxRow> &out) const
{
    out.clear();
}

uint StageConstraints::constraint_diff_size() const
{
    return output_size();
}

void StageConstraints::right_minus(ConstVecXRef x, ConstVecXRef y,
                                   VecX &out) const
{
    out = x - y;
}

void StageConstraints::right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                        VecXRef out) const
{
    right_minus(x, y, _right_minus_scratch);
    out = _right_minus_scratch;
}

void StageConstraints::add_gradient(ConstVecXRef x, ConstVecXRef u, MatXRef dX,
                                    MatXRef dU) const
{
    gradient(x, u, _add_scratch_dX, _add_scratch_dU);
    dX += _add_scratch_dX;
    dU += _add_scratch_dU;
}

void StageConstraints::add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                       ConstVecXRef mu, MatXRef Pxx) const
{
    hessprod_xx(x, u, mu, _add_scratch_xx);
    Pxx += _add_scratch_xx;
}

void StageConstraints::add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                       ConstVecXRef mu, MatXRef Pxu) const
{
    hessprod_xu(x, u, mu, _add_scratch_xu);
    Pxu += _add_scratch_xu;
}

void StageConstraints::add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                       ConstVecXRef mu, MatXRef Puu) const
{
    hessprod_uu(x, u, mu, _add_scratch_uu);
    Puu += _add_scratch_uu;
}
}
}
}