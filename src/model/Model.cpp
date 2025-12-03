#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
Model::Model(uint horizon, double dt, const Model::Structure &structure)
  : _dt(dt)
  , _terrain(structure.terrain)
  , _free_flyer(structure.free_flyer)
  , _costs(horizon + 1)
  , _constraints(horizon + 1)
  , _dynamics(horizon + 1) //H+1 to keep init_states for final stage
  , _forces(horizon + 1)
  , _integration(structure.integration)
{
    _dims.horizon     = horizon;
    _lin_dims.horizon = horizon;

    _dims.state      = 0;
    _dims.control    = 0;
    _dims.constraint = 0;

    _lin_dims.state      = 0;
    _lin_dims.control    = 0;
    _lin_dims.constraint = 0;

    _dims.stage_state.resize(horizon + 1);
    _dims.stage_control.resize(horizon);
    _dims.stage_constraint.resize(horizon + 1);
    _dims.nq.resize(horizon + 1, 0);
    _dims.nv.resize(horizon + 1, 0);
    _dims.njoints.resize(horizon + 1, 0);

    _lin_dims.stage_state.resize(_lin_dims.horizon + 1);
    _lin_dims.stage_control.resize(_lin_dims.horizon);
    _lin_dims.stage_constraint.resize(_lin_dims.horizon + 1);

    if (!structure.urdf_path.empty())
    {
        _impl = pinocchio_wrapper::makeImpl();
        pinocchio_wrapper::buildModelFromUrdf(structure.urdf_path,
                                              _impl,
                                              _free_flyer);
        uint nq, nv, njoints;
        std::tie(nq, nv) = pinocchio_wrapper::modelDimensions(_impl);
        njoints          = pinocchio_wrapper::getTotalActuatedJoints(_impl);

        std::fill(_dims.nq.begin(), _dims.nq.end(), nq);
        std::fill(_dims.nv.begin(), _dims.nv.end(), nv);
        std::fill(_dims.njoints.begin(), _dims.njoints.end(), njoints);
    }

    if (_terrain == nullptr)
    {
        _terrain = terrain::make_terrain<terrain::predefined::FlatTerrain>();
    }
}

Model::Model(uint horizon, double dt, const Config &config) : Model(horizon, dt)
{
    finalize(config);
}

Model::Model(uint horizon, double dt, const Structure &structure,
             const Config &config)
  : Model(horizon, dt, structure)
{
    finalize(config);
}

void Model::finalize(const Config &config)
{
    if (_finalized)
    {
        throw std::runtime_error("Model is already finalized");
    }

    // Add innate constraints if model comes from URDF
    if (_impl != nullptr)
    {
        _addURDFInnateConstraints();
    }

    // Setup dynamics
    assert(_impl == nullptr ^ config.dynamics.empty() &&
           "Expected either a URDF or custom dynamics");

    // Build dynamics from URDF
    if (config.dynamics.empty())
    {
        _buildURDFDynamics(config);
    }
    // Use custom dynamics
    else
    {
        _dynamics = config.dynamics;
    }

    // Setup costs
    assert((config.stage_cost != nullptr || config.costs.empty() ||
            config.costs.size() == _dims.horizon + 1) &&
           "Costs size mismatch");

    assert((config.constraints.empty() ||
            config.constraints.size() == _dims.horizon + 1) &&
           "Constraints size mismatch");

    if (config.stage_cost != nullptr)
    {
        for (uint k = 0; k < _dims.horizon; ++k)
        {
            addCost(k, config.stage_cost);
        }
    }

    if (!config.costs.empty())
    {
        for (uint k = 0; k < _dims.horizon; ++k)
        {
            if (config.costs[k] != nullptr)
            {
                addCost(k, config.costs[k]);
            }
        }
    }

    if (config.terminal_cost != nullptr)
    {
        addCost(_dims.horizon, config.terminal_cost);
    }

    // Setup constraints
    if (!config.constraints.empty())
    {
        for (uint k = 0; k < _dims.horizon; ++k)
        {
            if (config.constraints[k] != nullptr)
            {
                addConstraints(k, config.constraints[k]);
            }
        }

        if (config.constraints[_dims.horizon] != nullptr)
        {
            addConstraints(_dims.horizon, config.constraints[_dims.horizon]);
        }
    }

    // Setup stagewise dimension data
    for (uint k = 0; k < _dims.horizon; ++k)
    {
        assert(_dynamics[k] != nullptr && _costs[k] != nullptr &&
               "All stage dynamics and costs must be defined");
        _dims.stage_state[k]   = _dynamics[k]->state_size();
        _dims.stage_control[k] = _dynamics[k]->control_size();
        _dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->output_size() : 0;

        _lin_dims.stage_state[k]   = _dynamics[k]->state_diff_size();
        _lin_dims.stage_control[k] = _dims.stage_control[k];
        _lin_dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->constraint_diff_size()
                                       : 0;

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

    _lin_dims.stage_state[_lin_dims.horizon] =
      _dynamics[_dims.horizon]->state_diff_size();
    _lin_dims.stage_constraint[_lin_dims.horizon] =
      (_constraints[_lin_dims.horizon] != nullptr)
        ? _constraints[_lin_dims.horizon]->constraint_diff_size()
        : 0;

    _dims.state      += _dims.stage_state[_dims.horizon];
    _dims.constraint += _dims.stage_constraint[_dims.horizon];

    _lin_dims.state      += _lin_dims.stage_state[_lin_dims.horizon];
    _lin_dims.constraint += _lin_dims.stage_constraint[_lin_dims.horizon];

    _finalized = true;
}

void Model::_addURDFInnateConstraints()
{
    std::shared_ptr<stagewise::StageConstraints> innate_constraints;
    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints;

    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelBounds(_impl, _free_flyer);

    auto zero_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]); };

    auto zero_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _dims.nv[0], u.size()); };

    auto zero_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(u.size(), u.size()); };

    if (_free_flyer)
    {
        auto innate_constraint_eval = [this](const VecX &x, const VecX &u)
        {
            VecX c(3 * _dims.njoints[0]);
            c.head(_dims.njoints[0]) = x.segment(7, _dims.njoints[0]);
            c.segment(_dims.njoints[0], _dims.njoints[0]) =
              x.tail(_dims.njoints[0]);
            c.tail(_dims.njoints[0]) = u.segment(6, _dims.njoints[0]);

            return c;
        };

        auto innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(3 * _dims.njoints[0], 2 * _dims.nv[0]);
            dX.block(0, 6, _dims.njoints[0], _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
            dX.block(_dims.njoints[0],
                     _dims.nv[0] + 6,
                     _dims.njoints[0],
                     _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);

            dU = MatX::Zero(3 * _dims.njoints[0], u.size());
            dU.block(2 * _dims.njoints[0],
                     6,
                     _dims.njoints[0],
                     _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
        };

        innate_constraints =
          stagewise::factory::make_constraints(3 * _dims.njoints[0],
                                               3 * _dims.njoints[0],
                                               lb,
                                               ub,
                                               innate_constraint_eval,
                                               innate_constraint_grad,
                                               DefaultRightMinus,
                                               zero_hessprod_xx,
                                               zero_hessprod_xu,
                                               zero_hessprod_uu);

        auto terminal_innate_constraint_eval =
          [this](const VecX &x, const VecX &u)
        {
            VecX c(2 * _dims.njoints[0]);
            c.head(_dims.njoints[0]) = x.segment(7, _dims.njoints[0]);
            c.segment(_dims.njoints[0], _dims.njoints[0]) =
              x.tail(_dims.njoints[0]);

            return c;
        };

        auto terminal_innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(2 * _dims.njoints[0], 2 * _dims.nv[0]);
            dX.block(0, 6, _dims.njoints[0], _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
            dX.block(_dims.njoints[0],
                     _dims.nv[0] + 6,
                     _dims.njoints[0],
                     _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);

            dU = MatX::Zero(2 * _dims.njoints[0], 1);
        };

        terminal_innate_constraints =
          stagewise::factory::make_constraints(2 * _dims.njoints[0],
                                               2 * _dims.njoints[0],
                                               lb.head(2 * _dims.njoints[0]),
                                               ub.head(2 * _dims.njoints[0]),
                                               terminal_innate_constraint_eval,
                                               terminal_innate_constraint_grad,
                                               DefaultRightMinus,
                                               zero_hessprod_xx,
                                               zero_hessprod_xu,
                                               zero_hessprod_uu);
    }
    else
    {
        auto innate_constraint_eval = [this](const VecX &x, const VecX &u)
        {
            VecX c(3 * _dims.njoints[0]);
            c.head(2 * _dims.njoints[0]) = x;
            c.tail(_dims.njoints[0])     = u.head(_dims.njoints[0]);

            return c;
        };

        auto innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(3 * _dims.njoints[0], 2 * _dims.nv[0]);
            dX.block(0, 0, 2 * _dims.njoints[0], 2 * _dims.njoints[0]) =
              MatX::Identity(2 * _dims.njoints[0], 2 * _dims.njoints[0]);

            dU = MatX::Zero(3 * _dims.njoints[0], u.size());
            dU.block(2 * _dims.njoints[0],
                     0,
                     _dims.njoints[0],
                     _dims.njoints[0]) =
              MatX::Identity(_dims.njoints[0], _dims.njoints[0]);
        };

        innate_constraints =
          stagewise::factory::make_constraints(3 * _dims.njoints[0],
                                               3 * _dims.njoints[0],
                                               lb,
                                               ub,
                                               innate_constraint_eval,
                                               innate_constraint_grad,
                                               DefaultRightMinus,
                                               zero_hessprod_xx,
                                               zero_hessprod_xu,
                                               zero_hessprod_uu);

        auto terminal_innate_constraint_eval =
          [this](const VecX &x, const VecX &u)
        {
            VecX c(2 * _dims.njoints[0]);
            c.head(2 * _dims.njoints[0]) = x;

            return c;
        };

        auto terminal_innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(2 * _dims.njoints[0], 2 * _dims.nv[0]);
            dX.block(0, 0, 2 * _dims.njoints[0], 2 * _dims.njoints[0]) =
              MatX::Identity(2 * _dims.njoints[0], 2 * _dims.njoints[0]);

            dU = MatX::Zero(2 * _dims.njoints[0], 1);
        };

        terminal_innate_constraints =
          stagewise::factory::make_constraints(2 * _dims.njoints[0],
                                               2 * _dims.njoints[0],
                                               lb.head(2 * _dims.njoints[0]),
                                               ub.head(2 * _dims.njoints[0]),
                                               terminal_innate_constraint_eval,
                                               terminal_innate_constraint_grad,
                                               DefaultRightMinus,
                                               zero_hessprod_xx,
                                               zero_hessprod_xu,
                                               zero_hessprod_uu);
    }

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        _constraints[k] =
          _constraints[k] == nullptr
            ? innate_constraints
            : stagewise::factory::merge_constraints(_constraints[k],
                                                    innate_constraints);
    }

    _constraints[_dims.horizon] =
      _constraints[_dims.horizon] == nullptr
        ? terminal_innate_constraints
        : stagewise::factory::merge_constraints(_constraints[_dims.horizon],
                                                terminal_innate_constraints);
}

void Model::_buildURDFDynamics(const Config &config)
{
    assert((config.init_states.empty() || config.init_states.size() == 1 ||
            config.init_states.size() == _dims.horizon + 1) &&
           "Invalid init_states size");

    assert((config.init_controls.empty() || config.init_controls.size() == 1 ||
            config.init_controls.size() == _dims.horizon) &&
           "Invalid init_controls size");

    // total joints (including universe, root)
    const uint total_joints = pinocchio_wrapper::modelJoints(_impl);

    MatX S;
    if (_free_flyer)
    {
        S = MatX::Zero(_dims.nv[0], _dims.nv[0]);
        S.bottomRightCorner(_dims.nv[0] - 6, _dims.nv[0] - 6) =
          MatX::Identity(_dims.nv[0] - 6, _dims.nv[0] - 6);
    }
    else
    {
        S = MatX::Identity(_dims.nv[0], _dims.nv[0]);
    }

    auto right_plus = [this](const VecX &x, const VecX &dx)
    {
        VecX out(_dims.nq[0] + _dims.nv[0]);

        out.head(_dims.nq[0]) =
          pinocchio_wrapper::integrate(_impl,
                                       x.head(_dims.nq[0]),
                                       dx.head(_dims.nv[0]));
        out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + dx.tail(_dims.nv[0]);

        return out;
    };

    auto right_minus = [this](const VecX &x, const VecX &y)
    {
        VecX out(2 * _dims.nv[0]);

        out.head(_dims.nv[0]) =
          pinocchio_wrapper::difference(_impl,
                                        y.head(_dims.nq[0]),
                                        x.head(_dims.nq[0]));
        out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) - y.tail(_dims.nv[0]);

        return out;
    };

    ManifoldOperators<decltype(right_plus), decltype(right_minus)> ops = {
        right_plus,
        right_minus
    };

    auto zero_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]); };

    auto zero_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]); };

    auto zero_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(_dims.nv[0], _dims.nv[0]); };

    for (uint k = 0; k < _dynamics.size(); ++k)
    {
        uint nf = _forces[k].size();

        if (nf == 0)
        {
            // TODO: find a better way than rewriting everything
            if (_integration == ExplicitEuler)
            {
                auto dyn_eval = [this, S](const VecX &x, const VecX &u)
                {
                    VecX ddq = pinocchio_wrapper::aba(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      S * u);

                    VecX out(_dims.nq[0] + _dims.nv[0]);

                    out.head(_dims.nq[0]) =
                      pinocchio_wrapper::integrate(_impl,
                                                   x.head(_dims.nq[0]),
                                                   x.tail(_dims.nv[0]) * _dt);
                    out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;

                    return out;
                };

                auto dyn_grad =
                  [this, S](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
                {
                    VecX q   = x.head(_dims.nq[0]);
                    VecX v   = x.tail(_dims.nv[0]);
                    VecX tau = S * u;

                    MatX ddq_dq, ddq_dv, Minv;

                    pinocchio_wrapper::computeABADerivatives(_impl,
                                                             q,
                                                             v,
                                                             tau,
                                                             ddq_dq,
                                                             ddq_dv,
                                                             Minv);

                    MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
                    MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

                    pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_q, 0);
                    pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_v, 1);

                    dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
                    dX.topLeftCorner(_dims.nv[0], _dims.nv[0])  = Jint_q;
                    dX.topRightCorner(_dims.nv[0], _dims.nv[0]) = Jint_v * _dt;
                    dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      ddq_dq * _dt;
                    dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                      MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                    dU.bottomRows(_dims.nv[0]) = Minv * S * _dt;
                };

                _dynamics[k] = stagewise::factory::make_dynamics(
                  _dims.nq[0] + _dims.nv[0],
                  _dims.nv[0],
                  _dims.nq[0] + _dims.nv[0],
                  2 * _dims.nv[0],
                  config.init_states.size() == 0
                    ? VecX::Zero(_dims.nq[0] + _dims.nv[0])
                    : (config.init_states.size() == 1 ? config.init_states[0]
                                                      : config.init_states[k]),
                  (config.init_controls.size() == 0 || k == _dims.horizon)
                    ? VecX::Zero(_dims.nv[0])
                    : (config.init_controls.size() == 1
                         ? config.init_controls[0]
                         : config.init_controls[k]),
                  ops,
                  dyn_eval,
                  dyn_grad,
                  zero_hessprod_xx,
                  zero_hessprod_xu,
                  zero_hessprod_uu);
            }
            else // Semi-Implicit Euler by default
            {
                auto dyn_eval = [this, S](const VecX &x, const VecX &u)
                {
                    VecX ddq = pinocchio_wrapper::aba(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      S * u);

                    VecX out(_dims.nq[0] + _dims.nv[0]);

                    out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;
                    out.head(_dims.nq[0]) =
                      pinocchio_wrapper::integrate(_impl,
                                                   x.head(_dims.nq[0]),
                                                   out.tail(_dims.nv[0]) * _dt);

                    return out;
                };

                auto dyn_grad =
                  [this, S](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
                {
                    VecX q   = x.head(_dims.nq[0]);
                    VecX v   = x.tail(_dims.nv[0]);
                    VecX tau = S * u;

                    VecX ddq = pinocchio_wrapper::aba(_impl, q, v, tau);
                    VecX vn  = v + ddq * _dt;

                    MatX ddq_dq, ddq_dv, Minv;
                    pinocchio_wrapper::computeABADerivatives(_impl,
                                                             q,
                                                             v,
                                                             tau,
                                                             ddq_dq,
                                                             ddq_dv,
                                                             Minv);

                    MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
                    MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

                    pinocchio_wrapper::dIntegrate(_impl,
                                                  q,
                                                  vn * _dt,
                                                  Jint_q,
                                                  0);
                    pinocchio_wrapper::dIntegrate(_impl,
                                                  q,
                                                  vn * _dt,
                                                  Jint_v,
                                                  1);

                    dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
                    dX.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_q + Jint_v * _dt * _dt * ddq_dq;
                    dX.topRightCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_v * _dt *
                      (MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt);
                    dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      ddq_dq * _dt;
                    dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                      MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0]);
                    dU.topRows(_dims.nv[0])    = Jint_v * _dt * _dt * Minv * S;
                    dU.bottomRows(_dims.nv[0]) = Minv * S * _dt;
                };

                _dynamics[k] = stagewise::factory::make_dynamics(
                  _dims.nq[0] + _dims.nv[0],
                  _dims.nv[0],
                  _dims.nq[0] + _dims.nv[0],
                  2 * _dims.nv[0],
                  config.init_states.size() == 0
                    ? VecX::Zero(_dims.nq[0] + _dims.nv[0])
                    : (config.init_states.size() == 1 ? config.init_states[0]
                                                      : config.init_states[k]),
                  (config.init_controls.size() == 0 || k == _dims.horizon)
                    ? VecX::Zero(_dims.nv[0])
                    : (config.init_controls.size() == 1
                         ? config.init_controls[0]
                         : config.init_controls[k]),
                  ops,
                  dyn_eval,
                  dyn_grad,
                  zero_hessprod_xx,
                  zero_hessprod_xu,
                  zero_hessprod_uu);
            }
        }
        else
        {
            std::vector<Force> &forces = _forces[k];

            if (_integration == ExplicitEuler)
            {
                auto dyn_eval_fext =
                  [this, S, total_joints, nf, forces](const VecX &x,
                                                      const VecX &u)
                {
                    VecX tau = S * u.head(_dims.nv[0]);

                    pinocchio_wrapper::computeJointJacobians(
                      _impl,
                      x.head(_dims.nq[0]));
                    std::vector<Vec6> joint_forces(total_joints, Vec6::Zero());

                    for (uint i = 0; i < nf; ++i)
                    {
                        Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                        uint joint_id;
                        Vec6 f_res =
                          pinocchio_wrapper::framejMfAct(_impl,
                                                         forces[i].frame,
                                                         f,
                                                         joint_id);

                        joint_forces[joint_id] += f_res;
                    }

                    VecX ddq = pinocchio_wrapper::aba(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      tau,
                                                      joint_forces);

                    VecX out(_dims.nq[0] + _dims.nv[0]);

                    out.head(_dims.nq[0]) =
                      pinocchio_wrapper::integrate(_impl,
                                                   x.head(_dims.nq[0]),
                                                   x.tail(_dims.nv[0]) * _dt);
                    out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;

                    return out;
                };

                auto dyn_grad_fext =
                  [this, S, total_joints, nf, forces](const VecX &x,
                                                      const VecX &u,
                                                      MatX &dX,
                                                      MatX &dU)
                {
                    VecX q   = x.head(_dims.nq[0]);
                    VecX v   = x.tail(_dims.nv[0]);
                    VecX tau = S * u.head(_dims.nv[0]);

                    pinocchio_wrapper::computeJointJacobians(_impl, q);

                    MatX              Jf_full = MatX::Zero(3 * nf, _dims.nv[0]);
                    std::vector<Vec6> joint_forces(total_joints, Vec6::Zero());

                    for (uint i = 0; i < nf; ++i)
                    {
                        MatX Jf =
                          frameJacobian(forces[i].frame, true).topRows<3>();
                        Jf_full.block(3 * i, 0, 3, _dims.nv[0]) = Jf;

                        Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                        uint joint_id;
                        Vec6 f_res =
                          pinocchio_wrapper::framejMfAct(_impl,
                                                         forces[i].frame,
                                                         f,
                                                         joint_id);

                        joint_forces[joint_id] += f_res;
                    }

                    MatX ddq_dq, ddq_dv, Minv;

                    pinocchio_wrapper::computeABADerivatives(_impl,
                                                             q,
                                                             v,
                                                             tau,
                                                             joint_forces,
                                                             ddq_dq,
                                                             ddq_dv,
                                                             Minv);

                    MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
                    MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

                    pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_q, 0);
                    pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_v, 1);

                    dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
                    dX.topLeftCorner(_dims.nv[0], _dims.nv[0])  = Jint_q;
                    dX.topRightCorner(_dims.nv[0], _dims.nv[0]) = Jint_v * _dt;
                    dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      ddq_dq * _dt;
                    dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                      MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * nf);
                    dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Minv * S * _dt;
                    dU.bottomRightCorner(_dims.nv[0], 3 * nf) =
                      Minv * Jf_full.transpose() * _dt;
                };

                auto zero_hessprod_xu_fext = [this, nf](const VecX &x,
                                                        const VecX &u,
                                                        const VecX &lambda,
                                                        MatX &Pxu)
                { Pxu = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * nf); };

                auto zero_hessprod_uu_fext = [this, nf](const VecX &x,
                                                        const VecX &u,
                                                        const VecX &lambda,
                                                        MatX &Puu)
                {
                    Puu =
                      MatX::Zero(_dims.nv[0] + 3 * nf, _dims.nv[0] + 3 * nf);
                };

                _dynamics[k] = stagewise::factory::make_dynamics(
                  _dims.nq[0] + _dims.nv[0],
                  _dims.nv[0] + 3 * nf,
                  _dims.nq[0] + _dims.nv[0],
                  2 * _dims.nv[0],
                  config.init_states.size() == 0
                    ? VecX::Zero(_dims.nq[0] + _dims.nv[0])
                    : (config.init_states.size() == 1 ? config.init_states[0]
                                                      : config.init_states[k]),
                  config.init_controls.size() == 0
                    ? VecX::Zero(_dims.nv[0] + 3 * nf)
                    : (config.init_controls.size() == 1
                         ? config.init_controls[0]
                         : config.init_controls[k]),
                  ops,
                  dyn_eval_fext,
                  dyn_grad_fext,
                  zero_hessprod_xx,
                  zero_hessprod_xu_fext,
                  zero_hessprod_uu_fext);
            }
            else
            {
                auto dyn_eval_fext =
                  [this, S, total_joints, nf, forces](const VecX &x,
                                                      const VecX &u)
                {
                    VecX tau = S * u.head(_dims.nv[0]);

                    pinocchio_wrapper::computeJointJacobians(
                      _impl,
                      x.head(_dims.nq[0]));
                    std::vector<Vec6> joint_forces(total_joints, Vec6::Zero());

                    for (uint i = 0; i < nf; ++i)
                    {
                        Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                        uint joint_id;
                        Vec6 f_res =
                          pinocchio_wrapper::framejMfAct(_impl,
                                                         forces[i].frame,
                                                         f,
                                                         joint_id);

                        joint_forces[joint_id] += f_res;
                    }

                    VecX ddq = pinocchio_wrapper::aba(_impl,
                                                      x.head(_dims.nq[0]),
                                                      x.tail(_dims.nv[0]),
                                                      tau,
                                                      joint_forces);

                    VecX out(_dims.nq[0] + _dims.nv[0]);

                    out.tail(_dims.nv[0]) = x.tail(_dims.nv[0]) + ddq * _dt;
                    out.head(_dims.nq[0]) =
                      pinocchio_wrapper::integrate(_impl,
                                                   x.head(_dims.nq[0]),
                                                   out.tail(_dims.nv[0]) * _dt);

                    return out;
                };

                // TODO: Something is wrong here
                auto dyn_grad_fext =
                  [this, S, total_joints, nf, forces](const VecX &x,
                                                      const VecX &u,
                                                      MatX &dX,
                                                      MatX &dU)
                {
                    VecX q   = x.head(_dims.nq[0]);
                    VecX v   = x.tail(_dims.nv[0]);
                    VecX tau = S * u.head(_dims.nv[0]);

                    pinocchio_wrapper::computeJointJacobians(_impl, q);

                    MatX              Jf_full = MatX::Zero(3 * nf, _dims.nv[0]);
                    std::vector<Vec6> joint_forces(total_joints, Vec6::Zero());
                    for (uint i = 0; i < nf; ++i)
                    {
                        MatX Jf =
                          frameJacobian(forces[i].frame, true).topRows<3>();
                        Jf_full.block(3 * i, 0, 3, _dims.nv[0]) = Jf;

                        Vec3 f = u.segment(_dims.nv[0] + 3 * i, 3);

                        uint joint_id;
                        Vec6 f_res =
                          pinocchio_wrapper::framejMfAct(_impl,
                                                         forces[i].frame,
                                                         f,
                                                         joint_id);

                        joint_forces[joint_id] += f_res;
                    }

                    VecX ddq =
                      pinocchio_wrapper::aba(_impl, q, v, tau, joint_forces);
                    VecX vn = v + ddq * _dt;

                    MatX ddq_dq, ddq_dv, Minv;
                    pinocchio_wrapper::computeABADerivatives(_impl,
                                                             q,
                                                             v,
                                                             tau,
                                                             joint_forces,
                                                             ddq_dq,
                                                             ddq_dv,
                                                             Minv);

                    MatX Jint_q = MatX::Zero(_dims.nv[0], _dims.nv[0]);
                    MatX Jint_v = MatX::Zero(_dims.nv[0], _dims.nv[0]);

                    pinocchio_wrapper::dIntegrate(_impl,
                                                  q,
                                                  vn * _dt,
                                                  Jint_q,
                                                  0);
                    pinocchio_wrapper::dIntegrate(_impl,
                                                  q,
                                                  vn * _dt,
                                                  Jint_v,
                                                  1);

                    dX = MatX::Zero(2 * _dims.nv[0], 2 * _dims.nv[0]);
                    dX.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_q + Jint_v * _dt * _dt * ddq_dq;
                    dX.topRightCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_v * _dt *
                      (MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt);
                    dX.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      ddq_dq * _dt;
                    dX.bottomRightCorner(_dims.nv[0], _dims.nv[0]) =
                      MatX::Identity(_dims.nv[0], _dims.nv[0]) + ddq_dv * _dt;

                    dU = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * nf);
                    dU.topLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Jint_v * _dt * _dt * Minv * S;
                    dU.topRightCorner(_dims.nv[0], 3 * nf) =
                      Jint_v * _dt * _dt * Minv * Jf_full.transpose();
                    dU.bottomLeftCorner(_dims.nv[0], _dims.nv[0]) =
                      Minv * S * _dt;
                    dU.bottomRightCorner(_dims.nv[0], 3 * nf) =
                      Minv * Jf_full.transpose() * _dt;
                };

                auto zero_hessprod_xu_fext = [this, nf](const VecX &x,
                                                        const VecX &u,
                                                        const VecX &lambda,
                                                        MatX &Pxu)
                { Pxu = MatX::Zero(2 * _dims.nv[0], _dims.nv[0] + 3 * nf); };

                auto zero_hessprod_uu_fext = [this, nf](const VecX &x,
                                                        const VecX &u,
                                                        const VecX &lambda,
                                                        MatX &Puu)
                {
                    Puu =
                      MatX::Zero(_dims.nv[0] + 3 * nf, _dims.nv[0] + 3 * nf);
                };

                _dynamics[k] = stagewise::factory::make_dynamics(
                  _dims.nq[0] + _dims.nv[0],
                  _dims.nv[0] + 3 * nf,
                  _dims.nq[0] + _dims.nv[0],
                  2 * _dims.nv[0],
                  config.init_states.size() == 0
                    ? VecX::Zero(_dims.nq[0] + _dims.nv[0])
                    : (config.init_states.size() == 1 ? config.init_states[0]
                                                      : config.init_states[k]),
                  config.init_controls.size() == 0
                    ? VecX::Zero(_dims.nv[0] + 3 * nf)
                    : (config.init_controls.size() == 1
                         ? config.init_controls[0]
                         : config.init_controls[k]),
                  ops,
                  dyn_eval_fext,
                  dyn_grad_fext,
                  zero_hessprod_xx,
                  zero_hessprod_xu_fext,
                  zero_hessprod_uu_fext);
            }
        }
    }
}

bool Model::finalized() const
{
    return _finalized;
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

const std::shared_ptr<terrain::Terrain> Model::terrain() const
{
    return _terrain;
}

void Model::addCost(const uint                                   k,
                    const std::shared_ptr<stagewise::StageCost> &cost)
{
    if (_costs[k] == nullptr)
    {
        _costs[k] = cost;
        return;
    }

    _costs[k] = stagewise::factory::merge_costs(_costs[k], cost);
}

void Model::addConstraints(
  const uint k, const std::shared_ptr<stagewise::StageConstraints> &constraints)
{
    if (_constraints[k] == nullptr)
    {
        _constraints[k] = constraints;
        return;
    }

    _constraints[k] =
      stagewise::factory::merge_constraints(_constraints[k], constraints);
}

Force Model::addForce(const uint k, const std::string &frame)
{
    _forces[k].push_back(Force(frame, _forces[k].size()));
    return _forces[k].back();
}

uint Model::nforces(const uint k) const
{
    return _forces[k].size();
}

void Model::printFramesAndJoints()
{
    pinocchio_wrapper::printFramesAndJoints(_impl);
}

std::vector<std::string> Model::orderedActuatedJoints() const
{
    if (_impl == nullptr)
    {
        return {};
    }

    return pinocchio_wrapper::getOrderedActuatedJoints(_impl);
}

bool Model::floating_base() const
{
    return _free_flyer;
}

uint Model::nq(const uint k) const
{
    return _dims.nq[k];
}

uint Model::nv(const uint k) const
{
    return _dims.nv[k];
}

uint Model::njoints(const uint k) const
{
    return _dims.njoints[k];
}

double Model::dt() const
{
    return _dt;
}

const std::pair<VecX, MatX> Model::forwardKinematics(
  const VecX &q, const std::string &frame) const
{
    return pinocchio_wrapper::forwardKinematics(_impl, q, frame);
}

const MatX Model::forwardKinematicsSE3(const VecX        &q,
                                       const std::string &frame) const
{
    return pinocchio_wrapper::forwardKinematicsSE3(_impl, q, frame);
}

const Vec3 Model::frameVelocity(const VecX &q, const VecX &v,
                                const std::string &frame, bool local) const
{
    pinocchio_wrapper::forwardKinematics(_impl, q, v, frame);
    pinocchio_wrapper::updateFramePlacements(_impl);
    return pinocchio_wrapper::frameVelocity(_impl, frame, local);
}

std::pair<MatX, MatX> Model::frameVelocityDerivatives(const VecX        &q,
                                                      const VecX        &v,
                                                      const VecX        &a,
                                                      const std::string &frame,
                                                      bool local) const
{
    MatX dVdq(6, v.size()), dVdv(6, v.size());

    pinocchio_wrapper::frameVelocityDerivatives(_impl,
                                                q,
                                                v,
                                                a,
                                                frame,
                                                dVdq,
                                                dVdv,
                                                local);

    return { dVdq, dVdv };
}

const VecX Model::inverseDynamics(
  const VecX &q, const VecX &v, const VecX &a,
  const std::unordered_map<std::string, std::vector<Vec3>> &forces) const
{
    const uint total_joints = pinocchio_wrapper::modelJoints(_impl);

    pinocchio_wrapper::computeJointJacobians(_impl, q);
    std::vector<Vec3> joint_forces(total_joints, Vec3::Zero());

    for (const std::pair<std::string, std::vector<Vec3>> &pair : forces)
    {
        for (uint i = 0; i < pair.second.size(); ++i)
        {
            Vec3 f = pair.second[i];
            uint joint_id;
            Vec3 translation;
            Mat3 rotation;
            pinocchio_wrapper::getFramejMfMatrix(_impl,
                                                 pair.first,
                                                 joint_id,
                                                 rotation,
                                                 translation);

            joint_forces[joint_id] += rotation * f;
        }
    }

    return pinocchio_wrapper::rnea(_impl, q, v, a, joint_forces);
}

const MatX Model::frameJacobian(const VecX &x, const std::string &frame,
                                bool local) const
{
    uint nv;
    std::tie(std::ignore, nv) = pinocchio_wrapper::modelDimensions(_impl);

    MatX J6 = MatX::Zero(6, nv);
    pinocchio_wrapper::frameJacobian(_impl, x, frame, J6, local);

    return J6;
}

const MatX Model::frameJacobian(const std::string &frame, bool local) const
{
    uint nv;
    std::tie(std::ignore, nv) = pinocchio_wrapper::modelDimensions(_impl);

    MatX J6 = MatX::Zero(6, nv);
    pinocchio_wrapper::frameJacobian(_impl, frame, J6, local);

    return J6;
}

VecX Model::randomState(const uint stage) const
{
    // TODO: Velocity limit
    VecX rx(_dynamics[stage]->state_size());

    rx.head(_dims.nq[stage]) = pinocchio_wrapper::randomConfiguration(_impl);
    rx.tail(_dims.nv[stage]) = _randomVector(_dims.nv[stage]);

    if (_free_flyer)
    {
        rx.head(3) = _randomVector(3);
    }

    return rx;
}

VecX Model::randomControls(const uint stage) const
{
    // TODO: Effort limit
    VecX ru = _randomVector(_dims.nv[stage] + 3 * _forces[stage].size());
    return ru;
}

VecX Model::_randomVector(const uint size, const double start,
                          const double end) const
{
    return start + (end - start) * 0.5 * (VecX::Random(size).array() + 1.0);
}

}
}