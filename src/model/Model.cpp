#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
Model::Model(uint horizon, double dt, const Model::Structure &structure)
  : _dt(dt)
  , _terrain(structure.terrain == nullptr
               ? terrain::make_terrain<terrain::predefined::FlatTerrain>()
               : structure.terrain)
  , _free_flyer(structure.free_flyer)
  , _costs(horizon + 1)
  , _constraints(horizon + 1)
  , _dynamics(horizon + 1) //H+1 to keep init_states for final stage
  , _forces(horizon + 1)
  , _dynamics_mode(structure.dynamics_mode)
  , _integration(structure.integration)
  , _ignore_innate(structure.ignore_innate_constraints)
  , _actuated_joints(structure.actuated_joints)
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
        _total_joints    = pinocchio_wrapper::modelJoints(_impl);
        njoints          = pinocchio_wrapper::getTotalActuatedJoints(_impl);

        std::fill(_dims.nq.begin(), _dims.nq.end(), nq);
        std::fill(_dims.nv.begin(), _dims.nv.end(), nv);
        std::fill(_dims.njoints.begin(), _dims.njoints.end(), njoints);
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
    if (_impl != nullptr && !_ignore_innate)
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
    _StageInnateConstraintsContext ctx;

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        std::shared_ptr<stagewise::StageConstraints> innate_constraints;
        if (_dynamics_mode == DynamicsMode::Forward)
        {
            if (_free_flyer)
            {
                innate_constraints =
                  _makeStageInnateConstraints<true, false>(ctx);
            }
            else
            {
                innate_constraints =
                  _makeStageInnateConstraints<false, false>(ctx);
            }
        }
        else
        {
            uint nf        = _forces[k].size();
            ctx.force_data = _forces[k];

            if (nf == 0)
            {
                if (_free_flyer)
                {
                    innate_constraints =
                      _makeStageInnateConstraints<true,
                                                  false,
                                                  DynamicsMode::Inverse,
                                                  false>(ctx);
                }
                else
                {
                    innate_constraints =
                      _makeStageInnateConstraints<false,
                                                  false,
                                                  DynamicsMode::Inverse,
                                                  false>(ctx);
                }
            }
            else
            {
                if (_free_flyer)
                {
                    innate_constraints =
                      _makeStageInnateConstraints<true,
                                                  false,
                                                  DynamicsMode::Inverse,
                                                  true>(ctx);
                }
                else
                {
                    innate_constraints =
                      _makeStageInnateConstraints<false,
                                                  false,
                                                  DynamicsMode::Inverse,
                                                  true>(ctx);
                }
            }
        }

        _constraints[k] =
          _constraints[k] == nullptr
            ? innate_constraints
            : stagewise::factory::merge_constraints(_constraints[k],
                                                    innate_constraints);
    }

    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints;

    if (_free_flyer)
    {
        terminal_innate_constraints =
          _makeStageInnateConstraints<true, true>(ctx);
    }
    else
    {
        terminal_innate_constraints =
          _makeStageInnateConstraints<false, true>(ctx);
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

    uint nv_act = _dims.nv[0];
    MatX S      = MatX::Zero(nv_act, nv_act);

    if (_free_flyer)
    {
        nv_act -= 6;
    }

    if (_actuated_joints.empty())
    {
        S.bottomRightCorner(nv_act, nv_act) = MatX::Identity(nv_act, nv_act);
    }
    else
    {
        assert(_actuated_joints.size() <= nv_act &&
               "Invalid actuated joints size");

        for (uint j = 0; j < _actuated_joints.size(); ++j)
        {
            assert(_actuated_joints[j] < nv_act &&
                   "Invalid actuated joint index");
            S.bottomRightCorner(nv_act, nv_act)(_actuated_joints[j],
                                                _actuated_joints[j]) = 1.0;
        }
    }

    _StageDynamicsContext ctx { S };

    for (uint k = 0; k < _dynamics.size(); ++k)
    {
        uint nf        = _forces[k].size();
        ctx.force_data = _forces[k];

        ctx.init_state =
          config.init_states.size() == 0
            ? VecX::Zero(_dims.nq[0] + _dims.nv[0])
            : (config.init_states.size() == 1 ? config.init_states[0]
                                              : config.init_states[k]);
        ctx.init_control =
          (config.init_controls.size() == 0 || k == _dims.horizon)
            ? VecX::Zero(_dims.nv[0] + 3 * nf)
            : (config.init_controls.size() == 1 ? config.init_controls[0]
                                                : config.init_controls[k]);

        // TODO: Replace ifs
        if (nf == 0)
        {
            if (_dynamics_mode == DynamicsMode::Forward)
            {
                if (_integration == ExplicitEuler)
                {
                    _dynamics[k] =
                      _makeStageDynamics<false,
                                         DynamicsMode::Forward,
                                         IntegrationScheme::ExplicitEuler>(ctx);
                }
                else
                {
                    _dynamics[k] =
                      _makeStageDynamics<false,
                                         DynamicsMode::Forward,
                                         IntegrationScheme::SemiImplicitEuler>(
                        ctx);
                }
            }
            else
            {
                if (_integration == ExplicitEuler)
                {
                    _dynamics[k] =
                      _makeStageDynamics<false,
                                         DynamicsMode::Inverse,
                                         IntegrationScheme::ExplicitEuler>(ctx);
                }
                else
                {
                    _dynamics[k] =
                      _makeStageDynamics<false,
                                         DynamicsMode::Inverse,
                                         IntegrationScheme::SemiImplicitEuler>(
                        ctx);
                }
            }
        }
        else
        {
            if (_dynamics_mode == DynamicsMode::Forward)
            {
                if (_integration == ExplicitEuler)
                {
                    _dynamics[k] =
                      _makeStageDynamics<true,
                                         DynamicsMode::Forward,
                                         IntegrationScheme::ExplicitEuler>(ctx);
                }
                else
                {
                    _dynamics[k] =
                      _makeStageDynamics<true,
                                         DynamicsMode::Forward,
                                         IntegrationScheme::SemiImplicitEuler>(
                        ctx);
                }
            }
            else
            {
                if (_integration == ExplicitEuler)
                {
                    _dynamics[k] =
                      _makeStageDynamics<true,
                                         DynamicsMode::Inverse,
                                         IntegrationScheme::ExplicitEuler>(ctx);
                }
                else
                {
                    _dynamics[k] =
                      _makeStageDynamics<true,
                                         DynamicsMode::Inverse,
                                         IntegrationScheme::SemiImplicitEuler>(
                        ctx);
                }
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

bool Model::floatingBase() const
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

const VecX Model::inverseDynamics(const VecX &q, const VecX &v,
                                  const VecX &a) const
{
    return pinocchio_wrapper::rnea(_impl, q, v, a);
}

const VecX Model::inverseDynamics(
  const VecX &q, const VecX &v, const VecX &a,
  const std::unordered_map<std::string, std::vector<Vec3>> &forces) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);
    std::vector<Vec6> joint_forces(_total_joints, Vec6::Zero());

    for (const std::pair<std::string, std::vector<Vec3>> &pair : forces)
    {
        for (uint i = 0; i < pair.second.size(); ++i)
        {
            Vec3 f = pair.second[i];
            uint joint_id;

            Vec6 F = framejMfAct(_impl, pair.first, f, joint_id);
            joint_forces[joint_id] += F;
        }
    }

    return pinocchio_wrapper::rnea(_impl, q, v, a, joint_forces);
}

const VecX Model::inverseDynamics(const VecX &q, const VecX &v, const VecX &a,
                                  const std::vector<Vec6> &joint_forces) const
{
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

void Model::dDifference(const VecX &q0, const VecX &q1, MatX &J,
                        const uint arg_pos) const
{
    pinocchio_wrapper::dDifference(_impl, q0, q1, J, arg_pos);
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

std::vector<Vec6> Model::_calculateJointForces(
  const VecX &q, const VecX &f, const std::vector<Force> &force_data) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);
    std::vector<Vec6> joint_forces(_total_joints, Vec6::Zero());

    for (uint i = 0; i < force_data.size(); ++i)
    {
        Vec3 fi = f.segment(3 * i, 3);

        uint joint_id;
        Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                    force_data[i].frame,
                                                    fi,
                                                    joint_id);

        joint_forces[joint_id] += f_res;
    }

    return joint_forces;
}

std::pair<std::vector<Vec6>, MatX> Model::_calculateJointForcesAndJacobian(
  const VecX &q, const VecX &f, const std::vector<Force> &force_data) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);

    MatX              Jf_full = MatX::Zero(3 * force_data.size(), _dims.nv[0]);
    std::vector<Vec6> joint_forces(_total_joints, Vec6::Zero());

    for (uint i = 0; i < force_data.size(); ++i)
    {
        MatX Jf = frameJacobian(force_data[i].frame, true).topRows<3>();
        Jf_full.block(3 * i, 0, 3, _dims.nv[0]) = Jf;

        Vec3 fi = f.segment(3 * i, 3);

        uint joint_id;
        Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                    force_data[i].frame,
                                                    fi,
                                                    joint_id);

        joint_forces[joint_id] += f_res;
    }

    return { joint_forces, Jf_full };
}

}
}