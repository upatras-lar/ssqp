#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
void Function::set_enabled(const bool enabled)
{
    _enabled = !enabled;
}

bool Function::enabled() const
{
    return _enabled;
}
}
}
}