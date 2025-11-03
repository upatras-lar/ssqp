#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
Model::Model(
  const std::vector<std::shared_ptr<stagewise::StageCost>>        &costs,
  const std::vector<std::shared_ptr<stagewise::StageDynamics>>    &dynamics,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints)
  : _impl(pinocchio_wrapper::makeImpl())
  , _costs(costs)
  , _dynamics(dynamics)
  , _constraints(constraints.empty()
                   ? std::vector<std::shared_ptr<stagewise::StageConstraints>>(
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
        assert(_dynamics[k] != nullptr && _costs[k] != nullptr &&
               "All stage dynamics and costs must be defined");
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

    assert(_dynamics[_dims.horizon] != nullptr &&
           _costs[_dims.horizon] != nullptr &&
           "All stage dynamics and costs must be defined");
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

    _finalized = true;
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

std::shared_ptr<stagewise::StageCost> &Model::cost(const uint k)
{
    return _costs[k];
}

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
  : Model(urdf_path, horizon, dt, init_states)
{
    finalize(cost, terminal_cost, constraints);
}

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
  const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
  : Model(urdf_path, horizon, dt, init_states)
{
    finalize(costs, terminal_cost, constraints);
}

Model::Model(const std::string &urdf_path, const uint horizon, const double dt,
             const std::vector<VecX> &init_states)
  : _impl(pinocchio_wrapper::makeImpl())
  , _costs(horizon + 1)
  , _dynamics(horizon + 1)
  , _constraints(horizon + 1)
{
    pinocchio_wrapper::buildModelFromUrdf(urdf_path, _impl);

    int nq, nv;
    std::tie(nq, nv) = pinocchio_wrapper::modelDimensions(_impl);

    ManifoldOperators ops;
    ops.right_plus = [this, nq, nv](const VecX &x, const VecX &dx)
    {
        VecX out(nq + nv);

        out.head(nq) =
          pinocchio_wrapper::integrate(_impl, x.head(nq), dx.head(nv));
        out.tail(nv) = x.tail(nv) + dx.tail(nv);

        return out;
    };

    ops.right_minus = [this, nq, nv](const VecX &x, const VecX &y)
    {
        VecX out(2 * nv);

        out.head(nv) =
          pinocchio_wrapper::difference(_impl, y.head(nq), x.head(nq));
        out.tail(nv) = x.tail(nv) - y.tail(nv);

        return out;
    };

    auto dyn_eval = [this, nq, nv, dt](const VecX &x, const VecX &u)
    {
        VecX ddq = pinocchio_wrapper::aba(_impl, x.head(nq), x.tail(nv), u);

        VecX out(nq + nv);
        out.head(nq) =
          pinocchio_wrapper::integrate(_impl, x.head(nq), x.tail(nv) * dt);
        out.tail(nv) = x.tail(nv) + ddq * dt;

        return out;
    };

    auto dyn_grad =
      [this, nq, nv, dt](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q   = x.head(nq);
        VecX v   = x.tail(nv);
        VecX tau = u;

        Eigen::MatrixXd ddq_dq, ddq_dv, Minv;

        pinocchio_wrapper::computeABADerivatives(_impl,
                                                 q,
                                                 v,
                                                 tau,
                                                 ddq_dq,
                                                 ddq_dv,
                                                 Minv);

        MatX Jint_q = MatX::Zero(nv, nv);
        MatX Jint_v = MatX::Zero(nv, nv);

        pinocchio_wrapper::dIntegrate(_impl, q, v * dt, Jint_q, 0);
        pinocchio_wrapper::dIntegrate(_impl, q, v * dt, Jint_v, 1);

        dX                           = MatX::Zero(2 * nv, 2 * nv);
        dX.topLeftCorner(nv, nv)     = Jint_q;
        dX.topRightCorner(nv, nv)    = Jint_v * dt;
        dX.bottomLeftCorner(nv, nv)  = ddq_dq * dt;
        dX.bottomRightCorner(nv, nv) = MatX::Identity(nv, nv) + ddq_dv * dt;

        dU                = MatX::Zero(2 * nv, nv);
        dU.bottomRows(nv) = Minv * dt;
    };

    std::shared_ptr<stagewise::StageDynamics> dynamics = nullptr;

    if (init_states.empty())
    {
        dynamics = stagewise::factory::make_dynamics(nq + nv,
                                                     nv,
                                                     nq + nv,
                                                     2 * nv,
                                                     VecX::Zero(nq + nv),
                                                     ops,
                                                     dyn_eval,
                                                     dyn_grad);
    }
    else if (init_states.size() == 1)
    {
        dynamics = stagewise::factory::make_dynamics(nq + nv,
                                                     nv,
                                                     nq + nv,
                                                     2 * nv,
                                                     init_states[0],
                                                     ops,
                                                     dyn_eval,
                                                     dyn_grad);
    }
    else
    {
        assert(init_states.size() == horizon &&
               "init_states.size() must match horizon");
    }

    // TODO: linearize bounds
    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelBounds(_impl);

    auto innate_constraint_eval = [this, nv](const VecX &x, const VecX &u)
    {
        VecX c(3 * nv);
        c.head(nv)        = x.head(nv);
        c.segment(nv, nv) = x.tail(nv);
        c.tail(nv)        = u;

        return c;
    };

    auto innate_constraint_grad =
      [this, nv](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(3 * nv, 2 * nv);

        dU = MatX(3 * nv, nv);
        dU.block(0, 0, 2 * nv, nv).setZero();
        dU.block(2 * nv, 0, nv, nv) = MatX::Identity(nv, nv);
    };

    std::shared_ptr<stagewise::StageConstraints> innate_constraints =
      stagewise::factory::make_constraints(3 * nv,
                                           lb,
                                           ub,
                                           innate_constraint_eval,
                                           innate_constraint_grad);

    auto terminal_innate_constraint_eval =
      [this, nv](const VecX &x, const VecX &u)
    {
        VecX c(2 * nv);
        c.head(nv) = x.head(nv);
        c.tail(nv) = x.tail(nv);

        return c;
    };

    auto terminal_innate_constraint_grad =
      [this, nv](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        dX = MatX::Identity(2 * nv, 2 * nv);
        dU = MatX::Zero(2 * nv, 1);
    };

    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints =
      stagewise::factory::make_constraints(2 * nv,
                                           lb.head(2 * nv),
                                           ub.head(2 * nv),
                                           terminal_innate_constraint_eval,
                                           terminal_innate_constraint_grad);

    for (uint k = 0; k <= horizon; ++k)
    {
        _dynamics[k]    = dynamics != nullptr
                            ? dynamics
                            : stagewise::factory::make_dynamics(nq + nv,
                                                             nv,
                                                             nq + nv,
                                                             2 * nv,
                                                             init_states[k],
                                                             ops,
                                                             dyn_eval,
                                                             dyn_grad);
        _constraints[k] = innate_constraints;
    }

    _constraints[horizon] = terminal_innate_constraints;
}

const std::pair<VecX, MatX> Model::forwardKinematics(
  const VecX &x, const std::string &frame) const
{
    return pinocchio_wrapper::forwardKinematics(_impl, x, frame);
}

const MatX Model::frameJacobian(const VecX &x, const std::string &frame) const
{
    VecX p_ee;
    std::tie(p_ee, std::ignore) =
      pinocchio_wrapper::forwardKinematics(_impl, x, frame);

    int nv;
    std::tie(std::ignore, nv) = pinocchio_wrapper::modelDimensions(_impl);

    MatX J6(6, nv);
    pinocchio_wrapper::frameJacobian(_impl, x, frame, J6);

    return J6;
}

void Model::finalize(
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    finalize(
      std::vector<std::shared_ptr<stagewise::StageCost>>(_dynamics.size() - 1,
                                                         cost),
      terminal_cost,
      constraints,
      init_states);
}

void Model::finalize(
  const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
  const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    int horizon = _dynamics.size() - 1;

    if (!init_states.empty())
    {
        if (init_states.size() == 1)
        {
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[0]);
            }
        }
        else
        {
            assert(init_states.size() == horizon &&
                   "init_states.size() must match horizon");
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[k]);
            }
        }
    }

    assert(costs.size() == horizon && "Costs size mismatch");

    bool extra_constraints = !constraints.empty();
    if (extra_constraints)
    {
        assert(constraints.size() == horizon + 1 &&
               "Constraints size mismatch");
    }

    if (extra_constraints)
    {
        for (uint k = 0; k < horizon; ++k)
        {
            _costs[k] = costs[k];
            if (constraints[k] != nullptr)
            {
                _constraints[k] =
                  stagewise::factory::merge_constraints(_constraints[k],
                                                        constraints[k]);
            }
        }
        _costs[horizon] = terminal_cost;
        if (constraints[horizon] != nullptr)
        {
            _constraints[horizon] =
              stagewise::factory::merge_constraints(_constraints[horizon],
                                                    constraints[horizon]);
        }
    }
    else
    {
        for (uint k = 0; k < horizon; ++k)
        {
            _costs[k] = costs[k];
        }
        _costs[horizon] = terminal_cost;
    }

    _setup();
}

void Model::finalize(
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    int horizon = _dynamics.size() - 1;

    if (!init_states.empty())
    {
        if (init_states.size() == 1)
        {
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[0]);
            }
        }
        else
        {
            assert(init_states.size() == horizon &&
                   "init_states.size() must match horizon");
            for (uint k = 0; k < horizon; ++k)
            {
                _dynamics[k]->set_init_state(init_states[k]);
            }
        }
    }

    bool extra_constraints = !constraints.empty();
    if (extra_constraints)
    {
        assert(constraints.size() == horizon + 1 &&
               "Constraints size mismatch");
        for (uint k = 0; k < horizon; ++k)
        {
            if (constraints[k] != nullptr)
            {
                _constraints[k] =
                  stagewise::factory::merge_constraints(_constraints[k],
                                                        constraints[k]);
            }
        }
        if (constraints[horizon] != nullptr)
        {
            _constraints[horizon] =
              stagewise::factory::merge_constraints(_constraints[horizon],
                                                    constraints[horizon]);
        }
    }

    _setup();
}

}
}