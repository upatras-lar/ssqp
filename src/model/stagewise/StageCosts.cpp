#include "model/stagewise/StageCost.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
uint StageCost::TOTAL = 0;

StageCost::StageCost() : Function("Cost" + std::to_string(StageCost::TOTAL++))
{
}

StageCost::StageCost(const std::string &name)
  : Function(name.empty() ? "Cost" + std::to_string(StageCost::TOTAL++) : name)
{
}

void StageCost::set_enabled(const bool enabled)
{
    _enabled = enabled;
}

bool StageCost::enabled() const
{
    return _enabled;
}

void StageCost::set_stage_forces(
  const std::vector<std::shared_ptr<Force>> &forces,
  const std::vector<std::shared_ptr<Force>> &wrenches)
{
}

void StageCost::add_gradient(ConstVecXRef x, ConstVecXRef u, VecXRef dx,
                             VecXRef du) const
{
    gradient(x, u, _add_scratch_dx, _add_scratch_du);
    dx += _add_scratch_dx;
    du += _add_scratch_du;
}

void StageCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Hxx) const
{
    hessian_xx(x, u, _add_scratch_xx);
    Hxx += _add_scratch_xx;
}

void StageCost::add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Hxu) const
{
    hessian_xu(x, u, _add_scratch_xu);
    Hxu += _add_scratch_xu;
}

void StageCost::add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                               MatXRef Huu) const
{
    hessian_uu(x, u, _add_scratch_uu);
    Huu += _add_scratch_uu;
}
}
}
}