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
}
}
}