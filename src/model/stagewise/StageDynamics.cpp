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

const VecX &StageDynamics::init_state() const
{
    if (_init_state.size())
    {
        return _init_state;
    }

    if (static_cast<uint>(_default_init_state.size()) != state_size())
    {
        _default_init_state.setZero(state_size());
    }

    return _default_init_state;
}

void StageDynamics::set_init_state(const VecX &init_state)
{
    assert(init_state.size() == state_size() &&
           "FunctionStageDynamics: nx must match init_state.size()");
    _init_state = init_state;
}

const VecX &StageDynamics::init_control() const
{
    if (_init_control.size())
    {
        return _init_control;
    }

    if (static_cast<uint>(_default_init_control.size()) != control_size())
    {
        _default_init_control.setZero(control_size());
    }

    return _default_init_control;
}

void StageDynamics::set_init_control(const VecX &init_control)
{
    assert(init_control.size() == control_size() &&
           "FunctionStageDynamics: nu must match init_control.size()");
    _init_control = init_control;
}

void StageDynamics::state_right_plus(ConstVecXRef x, ConstVecXRef dx,
                                     VecX &out) const
{
    out = x + dx;
}

void StageDynamics::state_right_plus_into(ConstVecXRef x, ConstVecXRef dx,
                                          VecXRef out) const
{
    state_right_plus(x, dx, _right_plus_scratch);
    out = _right_plus_scratch;
}

void StageDynamics::state_right_minus(ConstVecXRef x, ConstVecXRef y,
                                      VecX &out) const
{
    out = x - y;
}

void StageDynamics::state_right_minus_into(ConstVecXRef x, ConstVecXRef y,
                                           VecXRef out) const
{
    state_right_minus(x, y, _right_minus_scratch);
    out = _right_minus_scratch;
}

void StageDynamics::add_hessprod_xx(ConstVecXRef x, ConstVecXRef u,
                                    ConstVecXRef lambda, MatXRef Pxx) const
{
    hessprod_xx(x, u, lambda, _add_scratch_xx);
    Pxx += _add_scratch_xx;
}

void StageDynamics::add_hessprod_xu(ConstVecXRef x, ConstVecXRef u,
                                    ConstVecXRef lambda, MatXRef Pxu) const
{
    hessprod_xu(x, u, lambda, _add_scratch_xu);
    Pxu += _add_scratch_xu;
}

void StageDynamics::add_hessprod_uu(ConstVecXRef x, ConstVecXRef u,
                                    ConstVecXRef lambda, MatXRef Puu) const
{
    hessprod_uu(x, u, lambda, _add_scratch_uu);
    Puu += _add_scratch_uu;
}

uint StageDynamics::parameter_size() const
{
    return 0;
}

VecX StageDynamics::parameters() const
{
    return VecX::Zero(0);
}

MatX StageDynamics::parameter_gradient(ConstVecXRef x, ConstVecXRef u) const
{
    return MatX::Zero(state_diff_size(), 0);
}

void StageDynamics::update_parameters(const VecX &theta) {}
}
}
}