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
  , _constraints(
      constraints.size() == 1 && constraints[0] == nullptr
        ? std::vector<std::shared_ptr<const stagewise::StageConstraints>>(
            costs.size(), nullptr)
        : constraints)
{
    _setup();
}

void Model::_setup()
{
    _dims.horizon     = _costs.size() - 1;
    _lin_dims.horizon = _dims.horizon;

    assert(_dims.horizon == _dynamics.size() - 1 &&
           (_dims.horizon == _constraints.size() - 1) &&
           "Horizons of stage functions must match");

    _dims.state      = 0;
    _dims.control    = 0;
    _dims.constraint = 0;

    _lin_dims.state      = 0;
    _lin_dims.control    = 0;
    _lin_dims.constraint = 0;

    _dims.stage_state.resize(_dims.horizon + 1);
    _dims.stage_control.resize(_dims.horizon);
    _dims.stage_constraint.resize(_dims.horizon + 1);
    _dims.stage_sc.resize(_dims.horizon + 1);

    _lin_dims.stage_state.resize(_lin_dims.horizon + 1);
    _lin_dims.stage_control.resize(_lin_dims.horizon);
    _lin_dims.stage_constraint.resize(_lin_dims.horizon + 1);
    _lin_dims.stage_sc.resize(_dims.horizon + 1);

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        _dims.stage_state[k]   = _dynamics[k]->state_size();
        _dims.stage_control[k] = _dynamics[k]->control_size();
        _dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->output_size() : 0;
        _dims.stage_sc[k] = _dims.stage_state[k] + _dims.stage_control[k];

        _lin_dims.stage_state[k] = _dynamics[k]->state_diff_size();
        // TODO: control and constraint manifold?
        _lin_dims.stage_control[k]    = _dims.stage_control[k];
        _lin_dims.stage_constraint[k] = _dims.stage_constraint[k];
        _lin_dims.stage_sc[k] =
          _lin_dims.stage_state[k] + _lin_dims.stage_control[k];

        if (k > 0)
        {
            assert(_dynamics[k - 1]->output_size() == _dims.stage_state[k] &&
                   "Dynamics function stage k output and state k+1 dimension "
                   "mismatch");
        }

        _dims.state      += _dims.stage_state[k];
        _dims.control    += _dims.stage_control[k];
        _dims.constraint += _dims.stage_constraint[k];

        _lin_dims.state      += _lin_dims.stage_state[k];
        _lin_dims.control    += _lin_dims.stage_control[k];
        _lin_dims.constraint += _lin_dims.stage_constraint[k];
    }

    _dims.stage_state[_dims.horizon] = _dynamics[_dims.horizon]->state_size();
    _dims.stage_constraint[_dims.horizon] =
      (_constraints[_dims.horizon] != nullptr)
        ? _constraints[_dims.horizon]->output_size()
        : 0;
    _dims.stage_sc[_dims.horizon] = _dims.stage_state[_dims.horizon];

    _lin_dims.stage_state[_lin_dims.horizon] =
      _dynamics[_dims.horizon]->state_diff_size();
    _lin_dims.stage_constraint[_lin_dims.horizon] =
      (_constraints[_lin_dims.horizon] != nullptr)
        ? _constraints[_lin_dims.horizon]->output_size()
        : 0;
    _lin_dims.stage_sc[_lin_dims.horizon] =
      _lin_dims.stage_state[_lin_dims.horizon];

    _dims.state      += _dims.stage_state[_dims.horizon];
    _dims.constraint += _dims.stage_constraint[_dims.horizon];

    _lin_dims.state      += _lin_dims.stage_state[_lin_dims.horizon];
    _lin_dims.constraint += _lin_dims.stage_constraint[_lin_dims.horizon];
}

const uint Model::horizon() const
{
    return _dims.horizon;
}

const Dimensions Model::dims() const
{
    return _dims;
}

const Dimensions Model::linear_dims() const
{
    return _lin_dims;
}

const CostsContainer Model::costs() const
{
    return _costs;
}

const DynamicsContainer Model::dynamics() const
{
    return _dynamics;
}

const ConstraintsContainer Model::constraints() const
{
    return _constraints;
}
}
}