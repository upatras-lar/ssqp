#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
uint StageDynamics::TOTAL = 0;

StageDynamics::StageDynamics()
  : Function("Dynamics" + std::to_string(StageDynamics::TOTAL++))
{
}

StageDynamics::StageDynamics(const std::string &name)
  : Function(name.empty() ? "Dynamics" + std::to_string(StageDynamics::TOTAL++)
                          : name)
{
}

uint StageDynamics::state_diff_size() const
{
    return state_size();
}

VecX StageDynamics::init_state() const
{
    return _init_state.size() ? _init_state : VecX::Zero(state_size());
}

void StageDynamics::set_init_state(const VecX &init_state)
{
    assert(init_state.size() == state_size() &&
           "FunctionStageDynamics: nx must match init_state.size()");
    _init_state = init_state;
}

VecX StageDynamics::init_control() const
{
    return _init_control.size() ? _init_control : VecX::Zero(control_size());
}

void StageDynamics::set_init_control(const VecX &init_control)
{
    assert(init_control.size() == control_size() &&
           "FunctionStageDynamics: nu must match init_control.size()");
    _init_control = init_control;
}

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