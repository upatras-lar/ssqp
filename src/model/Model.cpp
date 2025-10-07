#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
Model::Model(
  const std::vector<std::shared_ptr<const stagewise::StageCost>>     &costs,
  const std::vector<std::shared_ptr<const stagewise::StageDynamics>> &dynamics,
  const std::vector<std::shared_ptr<const stagewise::StageConstraints>>
    &constraints)
  : _costs(costs)
  , _dynamics(dynamics)
  , _constraints(constraints)
{
    _dims.horizon = costs.size() - 1;

    assert(_dims.horizon == dynamics.size() - 1 &&
           _dims.horizon == constraints.size() - 1 &&
           "Horizons of stage functions must match");

    _dims.stage_state.resize(_dims.horizon + 1);
    _dims.stage_control.resize(_dims.horizon);
    _dims.stage_constraint.resize(_dims.horizon + 1);

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        _dims.stage_state[k]      = _costs[k]->state_size();
        _dims.stage_control[k]    = _costs[k]->control_size();
        _dims.stage_constraint[k] = _constraints[k]->output_size();

        assert(_dynamics[k]->state_size() == _dims.stage_state[k] &&
               "Dynamics state dimension mismatch");
        assert(_dynamics[k]->control_size() == _dims.stage_control[k] &&
               "Dynamics control dimension mismatch");

        if (k > 0)
        {
            assert(_dynamics[k - 1]->output_size() == _dims.stage_state[k] &&
                   "Dynamics function stage k output and state k+1 dimension "
                   "mismatch");
        }

        if (_constraints[k] != nullptr)
        {
            assert(_constraints[k]->state_size() == _dims.stage_state[k] &&
                   "Constraints state dimension mismatch");
            assert(_constraints[k]->control_size() == _dims.stage_control[k] &&
                   "Constraints control dimension mismatch");
        }

        _dims.state      += _dims.stage_state[k];
        _dims.control    += _dims.stage_control[k];
        _dims.constraint += _dims.stage_constraint[k];
    }

    _dims.stage_state[_dims.horizon] = _costs[_dims.horizon]->state_size();
    _dims.stage_constraint[_dims.horizon] =
      _constraints[_dims.horizon]->output_size();

    assert(_dynamics[_dims.horizon]->state_size() ==
             _dims.stage_state[_dims.horizon] &&
           "Dynamics state dimension mismatch");

    if (_constraints[_dims.horizon] != nullptr)
    {
        assert(_constraints[_dims.horizon]->state_size() ==
                 _dims.stage_state[_dims.horizon] &&
               "Constraints state dimension mismatch");
    }

    _dims.state      += _dims.stage_state[_dims.horizon];
    _dims.constraint += _dims.stage_constraint[_dims.horizon];
}

const uint Model::horizon() const
{
    return _dims.horizon;
}

const uint Model::total_state_size() const
{
    return _dims.state;
}

const uint Model::total_control_size() const
{
    return _dims.control;
}

const uint Model::total_constraint_size() const
{
    return _dims.constraint;
}

const std::vector<uint> Model::state_dims() const
{
    return _dims.stage_state;
}

const std::vector<uint> Model::control_dims() const
{
    return _dims.stage_control;
}

const std::vector<uint> Model::constraint_dims() const
{
    return _dims.stage_constraint;
}

const Dimensions Model::dims() const
{
    return _dims;
}

const Container<const stagewise::StageCost> Model::costs() const
{
    return _costs;
}

const Container<const stagewise::StageDynamics> Model::dynamics() const
{
    return _dynamics;
}

const Container<const stagewise::StageConstraints> Model::constraints() const
{
    return _constraints;
}
}
}