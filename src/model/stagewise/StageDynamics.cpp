#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
uint StageDynamics::state_diff_size() const
{
    return state_size();
}

VecX StageDynamics::init_state() const
{
    return VecX::Zero(state_size());
}

void StageDynamics::set_init_state(const VecX &x) {}

VecX StageDynamics::state_right_plus(const VecX &x, const VecX &dx) const
{
    return x + dx;
}

VecX StageDynamics::state_right_minus(const VecX &x, const VecX &y) const
{
    return x - y;
}
}
}
}