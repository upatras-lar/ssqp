#include "model/stagewise/StageCost.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
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