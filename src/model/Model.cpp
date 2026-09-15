#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
auto to_compile_time(bool b)
{
    using T = std::integral_constant<bool, true>;
    using F = std::integral_constant<bool, false>;
    using V = std::variant<T, F>;
    return b ? V { T {} } : V { F {} };
}

auto to_compile_time(SolverMode sm)
{
    using DMK = std::integral_constant<SolverMode, SolverMode::Kinematics>;
    using DMF = std::integral_constant<SolverMode, SolverMode::ForwardDynamics>;
    using DMI = std::integral_constant<SolverMode, SolverMode::InverseDynamics>;
    using DMRI =
      std::integral_constant<SolverMode, SolverMode::ReducedInverseDynamics>;
    using V = std::variant<DMK, DMF, DMI, DMRI>;

    switch (sm)
    {
    case SolverMode::Kinematics:
        return V { DMK {} };
    case SolverMode::ForwardDynamics:
        return V { DMF {} };
    case SolverMode::InverseDynamics:
        return V { DMI {} };
    case SolverMode::ReducedInverseDynamics:
        return V { DMRI {} };
    default:
        throw std::invalid_argument("Unknown dynamics mode");
    }
}

auto to_compile_time(IntegrationScheme is)
{
    using ISEE  = std::integral_constant<IntegrationScheme,
                                         IntegrationScheme::ExplicitEuler>;
    using ISSIE = std::integral_constant<IntegrationScheme,
                                         IntegrationScheme::SemiImplicitEuler>;
    using V     = std::variant<ISEE, ISSIE>;

    switch (is)
    {
    case IntegrationScheme::ExplicitEuler:
        return V { ISEE {} };
    case IntegrationScheme::SemiImplicitEuler:
        return V { ISSIE {} };
    default:
        throw std::invalid_argument("Unknown integration scheme");
    }
}

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
  , _solver_mode(structure.solver_mode)
  , _integration(structure.integration)
  , _ignore_innate(structure.ignore_innate_constraints)
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

    _lin_dims.stage_state.resize(_lin_dims.horizon + 1);
    _lin_dims.stage_control.resize(_lin_dims.horizon);
    _lin_dims.stage_constraint.resize(_lin_dims.horizon + 1);

    _xml = !structure.urdf_path.empty() || !structure.mjcf_path.empty();

    if (_xml)
    {
        _impl = pinocchio_wrapper::makeImpl();

        if (!structure.urdf_path.empty())
        {
            pinocchio_wrapper::buildModelFromUrdf(structure.urdf_path,
                                                  _impl,
                                                  _free_flyer);
        }
        else
        {
            pinocchio_wrapper::buildModelFromMjcf(structure.mjcf_path,
                                                  _impl,
                                                  _free_flyer);
        }

        std::tie(_dims.nq, _dims.nv) =
          pinocchio_wrapper::modelDimensions(_impl);
        _dims.nu = _dims.nv;
        if (_solver_mode == SolverMode::ReducedInverseDynamics && _free_flyer)
        {
            _dims.nu -= 6;
        }

        _dims.njoints      = pinocchio_wrapper::getTotalActuatedJoints(_impl);
        _dims.total_joints = pinocchio_wrapper::modelJoints(_impl);

        _bounded_joint_indices = pinocchio_wrapper::modelBoundedJointIndices(
          _impl,
          _bounded_joint_indices_to_config,
          _bounded_joint_indices_to_tangent,
          _free_flyer);

        setActuatedJoints(structure.actuated_joints);
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

    // Add innate constraints if model comes from XML
    if (_xml)
    {
        if (config.custom_lower_position_bounds.size() &&
              config.custom_lower_position_bounds.size() !=
                _bounded_joint_indices.size() ||
            config.custom_upper_position_bounds.size() &&
              config.custom_upper_position_bounds.size() !=
                _bounded_joint_indices.size())
        {
            throw std::runtime_error(
              "Invalid custom position "
              "bound sizes (expected " +
              std::to_string(_bounded_joint_indices.size()) + ")");
        }

        if (config.custom_velocity_bounds.size() &&
            config.custom_velocity_bounds.size() != _dims.njoints)
        {
            throw std::runtime_error("Invalid custom velocity "
                                     "bound size (expected " +
                                     std::to_string(_dims.njoints) + ")");
        }

        if (config.custom_torque_bounds.size() &&
            config.custom_torque_bounds.size() != _actuated_joints.size())
        {
            throw std::runtime_error("Invalid torque bound size "
                                     "(expected " +
                                     std::to_string(_actuated_joints.size()) +
                                     ")");
        }

        if (!_ignore_innate)
        {
            _buildXMLInnateConstraints(config);
        }

        if (_solver_mode == SolverMode::InverseDynamics ||
            _solver_mode == SolverMode::ReducedInverseDynamics)
        {
            _buildXMLIDTorqueConstraints(config);
        }
    }

    // Setup dynamics
    if (!(!_xml ^ config.dynamics.empty()))
    {
        throw std::runtime_error("Expected either an XML description or "
                                 "custom dynamics");
    }

    if (!(config.init_states.empty() || config.init_states.size() == 1 ||
          config.init_states.size() == _dims.horizon + 1))
    {
        throw std::runtime_error("Invalid init_states size");
    }

    if (!(config.init_controls.empty() || config.init_controls.size() == 1 ||
          config.init_controls.size() == _dims.horizon))
    {
        throw std::runtime_error("Invalid init_controls size");
    }

    // Build dynamics from XML
    if (config.dynamics.empty())
    {
        _buildXMLDynamics(config);
    }
    // Use custom dynamics
    else
    {
        if (!(config.dynamics.size() == _dims.horizon + 1))
        {
            throw std::runtime_error("Invalid stage dynamics "
                                     "vector size for custom "
                                     "dynamics");
        }

        std::unordered_set<stagewise::StageDynamics *> dyn_set;
        for (uint k = 0; k <= _dims.horizon; ++k)
        {
            if (dyn_set.find(config.dynamics[k].get()) != dyn_set.end())
            {
                logln_at(LogLevel::Warning,
                         "Warning: "
                         "multiple "
                         "uses of the "
                         "same "
                         "StageDynamic"
                         "s "
                         "object is "
                         "discouraged "
                         "and can "
                         "result in "
                         "undefined "
                         "behavior if "
                         "different "
                         "initial "
                         "states are "
                         "expected "
                         "(e.g. in "
                         "MPC)");
                break;
            }
            dyn_set.emplace(config.dynamics[k].get());
        }

        if (config.init_states.size() == 1)
        {
            for (uint k = 0; k <= _dims.horizon; ++k)
            {
                config.dynamics[k]->set_init_state(config.init_states[0]);
            }
        }
        else if (config.init_states.size() == _dims.horizon + 1)
        {
            for (uint k = 0; k <= _dims.horizon; ++k)
            {
                config.dynamics[k]->set_init_state(config.init_states[k]);
            }
        }

        if (config.init_controls.size() == 1)
        {
            for (uint k = 0; k < _dims.horizon; ++k)
            {
                config.dynamics[k]->set_init_control(config.init_controls[0]);
            }
        }
        else if (config.init_controls.size() == _dims.horizon)
        {
            for (uint k = 0; k < _dims.horizon; ++k)
            {
                config.dynamics[k]->set_init_control(config.init_controls[k]);
            }
        }

        _dynamics = config.dynamics;
    }

    // Setup costs
    if (config.stage_cost == nullptr && !config.costs.empty() &&
        config.costs.size() != _dims.horizon + 1)
    {
        throw std::runtime_error("Costs size mismatch");
    }

    if (!config.constraints.empty() &&
        config.constraints.size() != _dims.horizon + 1)
    {
        throw std::runtime_error("Constraints size mismatch");
    }

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
        if (_dynamics[k] == nullptr)
        {
            throw std::runtime_error("Stage " + std::to_string(k) +
                                     " dynamics are undefined");
        }

        if (_costs[k] == nullptr)
        {
            throw std::runtime_error("Stage " + std::to_string(k) +
                                     " costs are undefined");
        }

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
            if (_dynamics[k - 1]->output_size() != _dims.stage_state[k])
            {
                throw std::runtime_error("Dynamics "
                                         "function "
                                         "stage " +
                                         std::to_string(k - 1) +
                                         " output "
                                         "and "
                                         "state " +
                                         std::to_string(k) +
                                         " dimension"
                                         " mismatc"
                                         "h");
            }
        }

        _dims.state      += _dims.stage_state[k];
        _dims.control    += _dims.stage_control[k];
        _dims.constraint += _dims.stage_constraint[k];

        _lin_dims.state      += _lin_dims.stage_state[k];
        _lin_dims.control    += _lin_dims.stage_control[k];
        _lin_dims.constraint += _lin_dims.stage_constraint[k];
    }

    if (_dynamics[_dims.horizon] == nullptr)
    {
        throw std::runtime_error("No dynamics for terminal stage passed "
                                 "(required for initial state)");
    }

    if (_costs[_dims.horizon] == nullptr)
    {
        throw std::runtime_error("Terminal stage costs are undefined");
    }

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

void Model::_buildXMLInnateConstraints(const Config &config)
{
    _StageInnateConstraintsContext ctx;
    ctx.nq              = _dims.nq;
    ctx.nv              = _dims.nv;
    ctx.nu              = _dims.nu;
    ctx.njoints         = _dims.njoints;
    ctx.lower_pos_bound = config.custom_lower_position_bounds;
    ctx.upper_pos_bound = config.custom_upper_position_bounds;
    ctx.vel_bound       = config.custom_velocity_bounds;

    if (config.custom_torque_bounds.size() &&
        _actuated_joints.size() != _dims.nu)
    {
        ctx.tau_bound =
          VecX::Constant(_dims.nu, std::numeric_limits<double>::infinity());

        for (uint j = 0; j < _actuated_joints.size(); ++j)
        {
            ctx.tau_bound[_actuated_joints[j]] = config.custom_torque_bounds[j];
        }
    }
    else
    {
        ctx.tau_bound = config.custom_torque_bounds;
    }

    auto ct_floating_base = to_compile_time(_free_flyer);
    auto ct_true          = to_compile_time(true);
    auto ct_false         = to_compile_time(false);
    auto ct_sm            = to_compile_time(_solver_mode);

    for (uint k = 0; k <= _dims.horizon; ++k)
    {
        ctx.stage_constraints_name =
          "StageInnateConstraints" + std::to_string(k);
        ctx.nf = _forces[k].size();

        auto ctx_ptr =
          std::make_shared<const _StageInnateConstraintsContext>(ctx);

        std::shared_ptr<stagewise::StageConstraints> innate_constraints =
          std::visit(
            [this, ctx_ptr](auto fb, auto it, auto dm)
            {
                return _makeStageInnateConstraints<fb.value,
                                                   it.value,
                                                   dm.value>(ctx_ptr);
            },
            ct_floating_base,
            k == _dims.horizon ? ct_true : ct_false,
            ct_sm);

        _constraints[k] =
          _constraints[k] == nullptr
            ? innate_constraints
            : stagewise::factory::merge_constraints(_constraints[k],
                                                    innate_constraints);
    }
}

void Model::_buildXMLIDTorqueConstraints(const Config &config)
{
    _StageInnateConstraintsContext ctx;
    ctx.nq                = _dims.nq;
    ctx.nv                = _dims.nv;
    ctx.nu                = _dims.nu;
    ctx.njoints           = _dims.njoints;
    ctx.actuated_joints   = _actuated_joints;
    ctx.unactuated_joints = _unactuated_joints;

    if (config.custom_torque_bounds.size() &&
        _actuated_joints.size() != _dims.nu)
    {
        ctx.tau_bound =
          VecX::Constant(_dims.nu, std::numeric_limits<double>::infinity());

        for (uint j = 0; j < _actuated_joints.size(); ++j)
        {
            ctx.tau_bound[_actuated_joints[j]] = config.custom_torque_bounds[j];
        }
    }
    else
    {
        ctx.tau_bound = config.custom_torque_bounds;
    }

    auto ct_floating_base = to_compile_time(_free_flyer);
    auto ct_true          = to_compile_time(true);
    auto ct_false         = to_compile_time(false);

    ctx.joint_forces_scratch.resize(_dims.total_joints);

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        ctx.stage_constraints_name =
          "StageIDTorqueConstraints" + std::to_string(k);
        ctx.nf         = _forces[k].size();
        ctx.force_data = _forces[k];
        ctx.jf_full_scratch.resize(3 * ctx.nf, _dims.nv);

        auto ctx_ptr =
          std::make_shared<const _StageInnateConstraintsContext>(ctx);

        std::shared_ptr<stagewise::StageConstraints> innate_constraints =
          std::visit(
            [this, ctx_ptr](auto fb, auto ri, auto wf)
            {
                return _makeStageIDTorqueConstraints<fb.value,
                                                     ri.value,
                                                     wf.value>(ctx_ptr);
            },
            ct_floating_base,
            _solver_mode == SolverMode::ReducedInverseDynamics ? ct_true
                                                               : ct_false,
            ctx.nf == 0 ? ct_false : ct_true);

        _constraints[k] =
          _constraints[k] == nullptr
            ? innate_constraints
            : stagewise::factory::merge_constraints(_constraints[k],
                                                    innate_constraints);
    }
}

void Model::_buildXMLDynamics(const Config &config)
{
    uint nv_act = _dims.nv;

    if (_free_flyer)
    {
        nv_act -= 6;
    }

    // Dynamics generation below masks unactuated joints directly from
    // _actuated_joints/_unactuated_joints (via the stage context)
    if (_actuated_joints.size() != _dims.njoints)
    {
        if (_actuated_joints.size() > nv_act)
        {
            throw std::runtime_error("Invalid actuated_joints "
                                     "passed (size " +
                                     std::to_string(_actuated_joints.size()) +
                                     " > " + std::to_string(nv_act) + ")");
        }

        for (uint i = 0; i < _actuated_joints.size(); ++i)
        {
            if (_actuated_joints[i] >= nv_act)
            {
                throw std::runtime_error("Invalid "
                                         "actuated_"
                                         "joints "
                                         "value (" +
                                         std::to_string(_actuated_joints[i]) +
                                         " >= " + std::to_string(nv_act) + ")");
            }
        }
    }

    VecX init_state;

    if (config.init_states.empty())
    {
        const std::vector<pinocchio_wrapper::JointType> joint_types =
          pinocchio_wrapper::getOrderedJointTypes(_impl);

        uint nx;

        if (_solver_mode == SolverMode::Kinematics)
        {
            nx = _dims.nq;
        }
        else
        {
            nx = _dims.nq + _dims.nv;
        }

        init_state.setZero(nx);

        uint offset = 0;
        if (_free_flyer)
        {
            init_state[6] = 1.0;
            offset        = 7;
        }

        for (uint i = 0; i < joint_types.size(); ++i)
        {
            switch (joint_types[i])
            {
            case pinocchio_wrapper::JointType::Revolute:
            case pinocchio_wrapper::JointType::Prismatic:
            default:
                ++offset;
                break;
            case pinocchio_wrapper::JointType::Continuous:
                init_state[offset]  = 1.0;
                offset             += 2;
                break;
            case pinocchio_wrapper::JointType::Spherical:
                init_state[offset + 3]  = 1.0;
                offset                 += 4;
                break;
            case pinocchio_wrapper::JointType::SphericalEuler:
                offset += 3;
                break;
            }
        }
    }
    else if (config.init_states.size() == 1)
    {
        init_state = config.init_states[0];
    }

    _StageDynamicsContext ctx {};
    ctx.nq                = _dims.nq;
    ctx.nv                = _dims.nv;
    ctx.nu                = _dims.nu;
    ctx.actuated_joints   = _actuated_joints;
    ctx.unactuated_joints = _unactuated_joints;
    ctx.joint_forces_scratch.resize(_dims.total_joints);

    auto ct_floating_base = to_compile_time(_free_flyer);
    auto ct_true          = to_compile_time(true);
    auto ct_false         = to_compile_time(false);
    auto ct_sm            = to_compile_time(_solver_mode);
    auto ct_is            = to_compile_time(_integration);

    for (uint k = 0; k < _dynamics.size(); ++k)
    {
        ctx.stage_dynamics_name = "StageDynamics" + std::to_string(k);
        ctx.nf                  = _forces[k].size();
        ctx.force_data          = _forces[k];
        ctx.jf_full_scratch.resize(3 * ctx.nf, _dims.nv);

        ctx.init_state =
          init_state.size() != 0 ? init_state : config.init_states[k];

        ctx.init_control =
          (config.init_controls.size() == 0 || k == _dims.horizon)
            ? VecX::Zero(ctx.nu + 3 * ctx.nf)
            : (config.init_controls.size() == 1 ? config.init_controls[0]
                                                : config.init_controls[k]);

        auto ctx_ptr = std::make_shared<const _StageDynamicsContext>(ctx);

        _dynamics[k] = std::visit(
          [this, ctx_ptr](auto fb, auto wf, auto dm, auto is)
          {
              return _makeStageDynamics<fb.value, wf.value, dm.value, is.value>(
                ctx_ptr);
          },
          ct_floating_base,
          ctx.nf == 0 ? ct_false : ct_true,
          ct_sm,
          ct_is);
    }
}

bool Model::finalized() const
{
    return _finalized;
}

bool Model::xml() const
{
    return _xml;
}

const uint Model::horizon() const
{
    return _dims.horizon;
}

const Dimensions &Model::dims() const noexcept
{
    return _dims;
}

const Dimensions &Model::linear_dims() const noexcept
{
    return _lin_dims;
}

void Model::updateDimensions()
{
    _dims.constraint     = 0;
    _lin_dims.constraint = 0;

    for (uint k = 0; k < _dims.horizon; ++k)
    {
        _dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->output_size() : 0;
        _lin_dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->constraint_diff_size()
                                       : 0;

        _dims.constraint     += _dims.stage_constraint[k];
        _lin_dims.constraint += _lin_dims.stage_constraint[k];
    }

    _dims.stage_constraint[_dims.horizon] =
      (_constraints[_dims.horizon] != nullptr)
        ? _constraints[_dims.horizon]->output_size()
        : 0;
    _lin_dims.stage_constraint[_lin_dims.horizon] =
      (_constraints[_lin_dims.horizon] != nullptr)
        ? _constraints[_lin_dims.horizon]->constraint_diff_size()
        : 0;

    _dims.constraint     += _dims.stage_constraint[_dims.horizon];
    _lin_dims.constraint += _lin_dims.stage_constraint[_lin_dims.horizon];
}

const CostsContainer &Model::costs() const
{
    return _costs;
}

const DynamicsContainer &Model::dynamics() const
{
    return _dynamics;
}

const ConstraintsContainer &Model::constraints() const
{
    return _constraints;
}

const std::shared_ptr<terrain::Terrain> Model::terrain() const
{
    return _terrain;
}

void Model::setInitState(const uint k, const VecX &init_state)
{
    if (!_finalized)
    {
        throw std::runtime_error("Cannot set initial state for an unfinalized "
                                 "model");
    }

    if (init_state.size() != _dynamics[k]->state_size())
    {
        throw std::runtime_error("Invalid initial state size for stage " +
                                 std::to_string(k));
    }

    _dynamics[k]->set_init_state(init_state);
}

void Model::setInitControl(const uint k, const VecX &init_control)
{
    if (!_finalized)
    {
        throw std::runtime_error("Cannot set initial controls for unfinalized "
                                 "model");
    }

    if (init_control.size() != _dynamics[k]->control_size())
    {
        throw std::runtime_error("Invalid initial controls size for stage " +
                                 std::to_string(k));
    }

    _dynamics[k]->set_init_control(init_control);
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

void Model::addRunningCost(const std::shared_ptr<stagewise::StageCost> &cost)
{
    for (uint k = 0; k < _dims.horizon; ++k)
    {
        addCost(k, cost);
    }
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

std::shared_ptr<Force> Model::addForce(const uint k, const std::string &frame)
{
    if (_finalized)
    {
        throw std::runtime_error("Cannot add a force to a model that is "
                                 "already finalized");
    }

    std::shared_ptr<Force> force =
      std::make_shared<Force>(frame, _forces[k].size());
    force->frame_id = pinocchio_wrapper::getFrameId(_impl, frame);

    _forces[k].push_back(force);
    return _forces[k].back();
}

uint Model::nforces(const uint k) const
{
    return _forces[k].size();
}

std::vector<std::string> Model::orderedForceFrames(const uint k) const
{
    std::vector<std::string> frames;

    for (uint i = 0; i < _forces[k].size(); ++i)
    {
        frames.push_back(_forces[k][i]->frame);
    }

    return frames;
}

bool Model::hasFrame(const std::string &frame) const
{
    return pinocchio_wrapper::hasFrame(_impl, frame);
}

uint Model::frameId(const std::string &frame) const
{
    return pinocchio_wrapper::getFrameId(_impl, frame);
}

void Model::printFramesAndJoints() const
{
    pinocchio_wrapper::printFramesAndJoints(_impl);
}

std::vector<std::string> Model::orderedBodyFrames() const
{
    if (_impl == nullptr)
    {
        return {};
    }

    return pinocchio_wrapper::getOrderedBodyFrames(_impl);
}

std::vector<std::string> Model::jointTypes() const
{
    if (_impl == nullptr)
    {
        if (_has_custom_properties)
        {
            return _custom_properties.joint_types;
        }
        return {};
    }

    std::vector<pinocchio_wrapper::JointType> types =
      pinocchio_wrapper::getOrderedJointTypes(_impl);
    std::vector<std::string> types_str(types.size());

    for (uint i = 0; i < types.size(); ++i)
    {
        switch (types[i])
        {
        case pinocchio_wrapper::JointType::Revolute:
            types_str[i] = "revolute";
            break;
        case pinocchio_wrapper::JointType::Continuous:
            types_str[i] = "continuous";
            break;
        case pinocchio_wrapper::JointType::Prismatic:
            types_str[i] = "prismatic";
            break;
        case pinocchio_wrapper::JointType::Spherical:
            types_str[i] = "spherical";
            break;
        case pinocchio_wrapper::JointType::SphericalEuler:
            types_str[i] = "spherical-euler";
            break;
        default:
        case pinocchio_wrapper::JointType::Unknown:
            types_str[i] = "unknown";
            break;
        }
    }

    return types_str;
}

std::vector<std::string> Model::orderedJoints() const
{
    if (_impl == nullptr)
    {
        if (_has_custom_properties)
        {
            return _custom_properties.ordered_joints;
        }

        return {};
    }

    return pinocchio_wrapper::getOrderedActuatedJoints(_impl);
}

std::vector<std::string> Model::orderedActuatedJoints() const
{
    if (_impl == nullptr)
    {
        return {};
    }

    return pinocchio_wrapper::getOrderedActuatedJoints(_impl, _actuated_joints);
}

void Model::setActuatedJoints(const std::vector<uint> &actuated_joints)
{
    _actuated_joints.clear();
    _unactuated_joints.clear();

    if (actuated_joints.empty())
    {
        for (uint j = 0; j < (_solver_mode == SolverMode::ReducedInverseDynamics
                                ? _dims.nu
                                : _dims.njoints);
             ++j)
        {
            _actuated_joints.push_back(j);
        }
    }
    else
    {
        for (const uint j : actuated_joints)
        {
            if (j >= _dims.njoints)
            {
                throw std::runtime_error(
                  "Model::setActuatedJoints: Invalid actuated joint index");
            }
        }

        _actuated_joints = actuated_joints;
        std::sort(_actuated_joints.begin(), _actuated_joints.end());
        _actuated_joints.erase(
          std::unique(_actuated_joints.begin(), _actuated_joints.end()),
          _actuated_joints.end());
    }

    if (_solver_mode == ReducedInverseDynamics && !_actuated_joints.empty())
    {
        _dims.nu = _actuated_joints.size();
    }

    std::vector<bool> is_actuated(_dims.njoints, false);

    for (uint i = 0; i < _actuated_joints.size(); ++i)
    {
        is_actuated[_actuated_joints[i]] = true;
    }

    for (uint j = 0; j < _dims.njoints; ++j)
    {
        if (!is_actuated[j])
        {
            _unactuated_joints.push_back(j);
        }
    }
}

bool Model::floatingBase() const
{
    return _free_flyer;
}

uint Model::nq() const
{
    return _dims.nq;
}

uint Model::nv() const
{
    return _dims.nv;
}

uint Model::nu() const
{
    return _dims.nu;
}

uint Model::njoints() const
{
    return _dims.njoints;
}

uint Model::total_joints() const
{
    return _dims.total_joints;
}

double Model::dt() const
{
    return _dt;
}

SolverMode Model::solverMode() const
{
    return _solver_mode;
}

const VecX Model::neutralPose() const
{
    return pinocchio_wrapper::neutralPose(_impl);
}

const std::pair<Vec3, Mat3> Model::forwardKinematics(
  const VecX &q, const std::string &frame) const
{
    return forwardKinematics(q, pinocchio_wrapper::getFrameId(_impl, frame));
}

const std::pair<Vec3, Mat3> Model::forwardKinematics(
  const VecX &q, const uint frame_id) const
{
    return pinocchio_wrapper::forwardKinematics(_impl, q, frame_id);
}

const Mat4 Model::forwardKinematicsSE3(const VecX        &q,
                                       const std::string &frame) const
{
    return forwardKinematicsSE3(q, pinocchio_wrapper::getFrameId(_impl, frame));
}

const Mat4 Model::forwardKinematicsSE3(const VecX &q,
                                       const uint  frame_id) const
{
    return pinocchio_wrapper::forwardKinematicsSE3(_impl, q, frame_id);
}

const Vec6 Model::frameVelocity(const VecX &q, const VecX &v,
                                const std::string &frame, bool local) const
{
    return frameVelocity(q, v, pinocchio_wrapper::getFrameId(_impl, frame),
                         local);
}

const Vec6 Model::frameVelocity(const VecX &q, const VecX &v,
                                const uint frame_id, bool local) const
{
    pinocchio_wrapper::forwardKinematics(_impl, q, v, frame_id);
    pinocchio_wrapper::updateFramePlacements(_impl);
    return pinocchio_wrapper::frameVelocity(_impl, frame_id, local);
}

std::pair<MatX, MatX> Model::frameVelocityDerivatives(const VecX        &q,
                                                      const VecX        &v,
                                                      const std::string &frame,
                                                      bool local) const
{
    return frameVelocityDerivatives(
      q, v, pinocchio_wrapper::getFrameId(_impl, frame), local);
}

std::pair<MatX, MatX> Model::frameVelocityDerivatives(const VecX &q,
                                                      const VecX &v,
                                                      const uint  frame_id,
                                                      bool local) const
{
    MatX dVdq(6, v.size()), dVdv(6, v.size());

    pinocchio_wrapper::frameVelocityDerivatives(_impl,
                                                q,
                                                v,
                                                frame_id,
                                                dVdq,
                                                dVdv,
                                                local);

    return { dVdq, dVdv };
}

void Model::frameVelocityDerivatives(const VecX &q, const VecX &v,
                                     const std::string &frame, bool local,
                                     Eigen::Ref<MatX> dVdq,
                                     Eigen::Ref<MatX> dVdv) const
{
    frameVelocityDerivatives(q, v, pinocchio_wrapper::getFrameId(_impl, frame),
                             local, dVdq, dVdv);
}

void Model::frameVelocityDerivatives(const VecX &q, const VecX &v,
                                     const uint frame_id, bool local,
                                     Eigen::Ref<MatX> dVdq,
                                     Eigen::Ref<MatX> dVdv) const
{
    pinocchio_wrapper::frameVelocityDerivatives(_impl, q, v, frame_id, dVdq, dVdv,
                                                local);
}

const VecX Model::forwardDynamics(const VecX &q, const VecX &v,
                                  const VecX &tau) const
{
    return pinocchio_wrapper::aba(_impl, q, v, tau);
}

const VecX Model::forwardDynamics(
  const VecX &q, const VecX &v, const VecX &tau,
  const std::unordered_map<std::string, std::vector<Vec3>> &forces) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);
    std::vector<Vec6> joint_forces(_dims.total_joints, Vec6::Zero());

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

    return pinocchio_wrapper::aba(_impl, q, v, tau, joint_forces);
}

const VecX Model::forwardDynamics(const VecX &q, const VecX &v, const VecX &tau,
                                  const std::vector<Vec6> &joint_forces) const
{
    return pinocchio_wrapper::aba(_impl, q, v, tau, joint_forces);
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
    std::vector<Vec6> joint_forces(_dims.total_joints, Vec6::Zero());

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

const void Model::inverseDynamicsDerivatives(
  const Eigen::VectorXd &q, const Eigen::VectorXd &v, const Eigen::VectorXd &a,
  Eigen::MatrixXd &rnea_partial_dq, Eigen::MatrixXd &rnea_partial_dv,
  Eigen::MatrixXd &rnea_partial_da) const
{
    pinocchio_wrapper::computeRNEADerivatives(_impl,
                                              q,
                                              v,
                                              a,
                                              rnea_partial_dq,
                                              rnea_partial_dv,
                                              rnea_partial_da);
}

const void Model::inverseDynamicsDerivatives(
  const Eigen::VectorXd &q, const Eigen::VectorXd &v, const Eigen::VectorXd &a,
  const std::vector<Eigen::Vector<double, 6>> &fext,
  Eigen::MatrixXd &rnea_partial_dq, Eigen::MatrixXd &rnea_partial_dv,
  Eigen::MatrixXd &rnea_partial_da) const
{
    pinocchio_wrapper::computeRNEADerivatives(_impl,
                                              q,
                                              v,
                                              a,
                                              fext,
                                              rnea_partial_dq,
                                              rnea_partial_dv,
                                              rnea_partial_da);
}

const MatX Model::frameJacobian(const VecX &q, const std::string &frame,
                                bool local) const
{
    return frameJacobian(q, pinocchio_wrapper::getFrameId(_impl, frame),
                         local);
}

const MatX Model::frameJacobian(const VecX &q, const uint frame_id,
                                bool local) const
{
    MatX J6 = MatX::Zero(6, _dims.nv);
    frameJacobian(q, frame_id, local, J6);

    return J6;
}

void Model::frameJacobian(const VecX &q, const std::string &frame, bool local,
                          Eigen::Ref<MatX> J) const
{
    frameJacobian(q, pinocchio_wrapper::getFrameId(_impl, frame), local, J);
}

void Model::frameJacobian(const VecX &q, const uint frame_id, bool local,
                          Eigen::Ref<MatX> J) const
{
    // getFrameJacobian relies on J starting fully zeroed
    J.setZero();
    pinocchio_wrapper::frameJacobian(_impl, q, frame_id, J, local);
}

const MatX Model::frameJacobian(const std::string &frame, bool local) const
{
    return frameJacobian(pinocchio_wrapper::getFrameId(_impl, frame), local);
}

const MatX Model::frameJacobian(const uint frame_id, bool local) const
{
    MatX J6 = MatX::Zero(6, _dims.nv);
    frameJacobian(frame_id, local, J6);

    return J6;
}

void Model::frameJacobian(const std::string &frame, bool local,
                          Eigen::Ref<MatX> J) const
{
    frameJacobian(pinocchio_wrapper::getFrameId(_impl, frame), local, J);
}

void Model::frameJacobian(const uint frame_id, bool local,
                          Eigen::Ref<MatX> J) const
{
    // getFrameJacobian relies on J starting fully zeroed
    J.setZero();
    pinocchio_wrapper::frameJacobian(_impl, frame_id, J, local);
}

const Vec3 Model::centerOfMass(const VecX &q) const
{
    return pinocchio_wrapper::centerOfMass(_impl, q);
}

const MatX Model::jacobianCenterOfMass(const VecX &q) const
{
    return pinocchio_wrapper::jacobianCenterOfMass(_impl, q);
}

const VecX Model::getAllDynamicParameters() const
{
    return pinocchio_wrapper::getAllDynamicParameters(_impl);
}

void Model::setAllDynamicParameters(const VecX &parameters)
{
    pinocchio_wrapper::setAllDynamicParameters(_impl, parameters);
}

const MatX Model::getJointTorqueRegressor(const VecX &q, const VecX &v,
                                          const VecX &a) const
{
    return pinocchio_wrapper::computeJointTorqueRegressor(_impl, q, v, a);
}

VecX Model::difference(const VecX &q0, const VecX &q1) const
{
    if (_xml)
    {
        return pinocchio_wrapper::difference(_impl, q1, q0);
    }

    return q0 - q1;
}

void Model::dDifference(const VecX &q0, const VecX &q1, MatX &J,
                        const uint arg_pos) const
{
    if (_xml)
    {
        pinocchio_wrapper::dDifference(_impl, q1, q0, J, !arg_pos);
        return;
    }

    J.diagonal().setConstant(arg_pos == 0 ? 1.0 : -1.0);
}

MatX Model::getMatrixActuatedPart(const MatX &M) const
{
    MatX M_act(_dims.nu, _dims.nu);
    getMatrixActuatedPart(M, M_act);

    return M_act;
}

void Model::getMatrixActuatedPart(const MatX &M, Eigen::Ref<MatX> out) const
{
    if (M.rows() != _dims.nv || M.cols() != _dims.nv)
    {
        throw std::runtime_error("Model::getMatrixActuatedPart: Incorrect "
                                 "matrix size passed");
    }

    if (_dims.nu == _dims.nv)
    {
        out = M;
        return;
    }

    uint offset = _free_flyer ? 6 : 0;

    for (uint i = 0; i < _dims.nu; ++i)
    {
        for (uint j = 0; j < _dims.nu; ++j)
        {
            out(i, j) =
              M(offset + _actuated_joints[i], offset + _actuated_joints[j]);
        }
    }
}

VecX Model::getFullAccelerationVector(const VecX &x, const VecX &u,
                                      const uint k) const
{
    switch (_solver_mode)
    {
    default:
    case SolverMode::ForwardDynamics:
        return forwardDynamics(x.head(_dims.nq),
                               x.tail(_dims.nv),
                               u.head(_dims.nv));
    case SolverMode::InverseDynamics:
        return u.head(_dims.nv);
    case SolverMode::ReducedInverseDynamics:
        auto ct_floating_base = to_compile_time(_free_flyer);
        auto ct_with_forces   = to_compile_time(!_forces[k].empty());

        return std::visit(
          [this, x, u, k](auto fb, auto wf)
          {
              _ReducedInverseDynamicsScratch scratch;
              std::vector<Vec6> joint_forces(_dims.total_joints);
              MatX               Jf_full(3 * _forces[k].size(), _dims.nv);

              return _calculateReducedInverseFullAcceleration<fb.value,
                                                              wf.value>(
                x.head(_dims.nq),
                x.tail(_dims.nv),
                u.head(_dims.nu),
                u.tail(3 * _forces[k].size()),
                _forces[k],
                _actuated_joints,
                _unactuated_joints,
                scratch,
                joint_forces,
                Jf_full);
          },
          ct_floating_base,
          ct_with_forces);
    }
}

VecX Model::getActuatedTorqueVector(const VecX &x, const VecX &u,
                                    const uint k) const
{
    VecX q = x.head(_dims.nq);
    VecX v = x.tail(_dims.nv);

    if (_solver_mode == SolverMode::ReducedInverseDynamics)
    {
        VecX full_accel = getFullAccelerationVector(x, u, k);

        VecX tau;

        if (_forces[k].empty())
        {
            tau = inverseDynamics(q, v, full_accel);
        }
        else
        {
            std::vector<Vec6> joint_forces(_dims.total_joints);
            _calculateJointForces(q,
                                  u.tail(3 * _forces[k].size()),
                                  _forces[k],
                                  joint_forces);
            tau = inverseDynamics(q, v, full_accel, joint_forces);
        }

        uint offset = _free_flyer ? 6 : 0;
        VecX tau_act(_dims.nu);

        for (uint i = 0; i < _dims.nu; ++i)
        {
            tau_act[i] = tau[offset + _actuated_joints[i]];
        }

        return tau_act;
    }

    // InverseDynamics (non-reduced): the acceleration control is the full
    // acceleration vector, and every joint is actuated (nu == nv)
    VecX a = u.head(_dims.nv);

    if (_forces[k].empty())
    {
        return inverseDynamics(q, v, a);
    }

    std::vector<Vec6> joint_forces(_dims.total_joints);
    _calculateJointForces(q, u.tail(3 * _forces[k].size()), _forces[k],
                          joint_forces);

    return inverseDynamics(q, v, a, joint_forces);
}

void Model::getActuatedTorqueVectorDerivatives(const VecX &x, const VecX &u,
                                               MatX &dtau_dx, MatX &dtau_du,
                                               const uint k) const
{
    if (_solver_mode == SolverMode::ReducedInverseDynamics)
    {
        auto ct_floating_base = to_compile_time(_free_flyer);
        auto ct_with_forces   = to_compile_time(!_forces[k].empty());

        std::visit(
          [this, &x, &u, k, &dtau_dx, &dtau_du](auto fb, auto wf)
          {
              _getActuatedTorqueVectorReducedDerivatives<fb.value, wf.value>(
                x,
                u,
                k,
                dtau_dx,
                dtau_du);
          },
          ct_floating_base,
          ct_with_forces);

        return;
    }

    VecX q = x.head(_dims.nq);
    VecX v = x.tail(_dims.nv);
    VecX a = u.head(_dims.nv);

    MatX dtau_dq, dtau_dv, M;

    dtau_dx.resize(_dims.nv, 2 * _dims.nv);
    dtau_du.setZero(_dims.nv, u.size());

    if (_forces[k].empty())
    {
        inverseDynamicsDerivatives(q, v, a, dtau_dq, dtau_dv, M);

        dtau_du.leftCols(_dims.nv) = M;
    }
    else
    {
        std::vector<Vec6> joint_forces(_dims.total_joints);
        MatX               Jf_full(3 * _forces[k].size(), _dims.nv);

        _calculateJointForcesAndJacobian(q,
                                         u.tail(3 * _forces[k].size()),
                                         _forces[k],
                                         joint_forces,
                                         Jf_full);

        inverseDynamicsDerivatives(q, v, a, joint_forces, dtau_dq, dtau_dv, M);

        dtau_du.leftCols(_dims.nv)                = M;
        dtau_du.rightCols(3 * _forces[k].size())  = -Jf_full.transpose();
    }

    dtau_dx.leftCols(_dims.nv)  = dtau_dq;
    dtau_dx.rightCols(_dims.nv) = dtau_dv;
}

VecX Model::randomState(const uint stage) const
{
    // TODO: Velocity limit
    VecX rx(_dynamics[stage]->state_size());

    if (_impl != nullptr)
    {
        rx.head(_dims.nq) = pinocchio_wrapper::randomConfiguration(_impl);
    }
    else
    {
        rx.head(_dims.nq) = _randomVector(_dims.nq);
    }

    if (_solver_mode == SolverMode::ForwardDynamics ||
        _solver_mode == SolverMode::InverseDynamics ||
        _solver_mode == SolverMode::ReducedInverseDynamics)
    {
        rx.tail(_dims.nv) = _randomVector(_dims.nv);
    }

    if (_free_flyer)
    {
        rx.head(3) = _randomVector(3);
    }

    return rx;
}

VecX Model::randomControls(const uint stage) const
{
    // TODO: Effort limit
    VecX ru = _randomVector(_dims.nu + 3 * _forces[stage].size());
    return ru;
}

void Model::setCustomProperties(const CustomProperties &custom_properties)
{
    _has_custom_properties = true;
    _custom_properties     = custom_properties;

    _dims.nq      = custom_properties.nq;
    _dims.nv      = custom_properties.nv;
    _dims.nu      = custom_properties.nu;
    _dims.njoints = custom_properties.njoints;
    _free_flyer   = custom_properties.floating_base;
}

bool Model::hasCustomProperties() const
{
    return _has_custom_properties;
}

VecX Model::_randomVector(const uint size, const double start,
                          const double end) const
{
    return start + (end - start) * 0.5 * (VecX::Random(size).array() + 1.0);
}

void Model::_calculateJointForces(
  const VecX &q, const VecX &f,
  const std::vector<std::shared_ptr<Force>> &force_data,
  std::vector<Vec6>                         &joint_forces) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);
    std::fill(joint_forces.begin(), joint_forces.end(), Vec6::Zero());

    for (uint i = 0; i < force_data.size(); ++i)
    {
        if (force_data[i]->ignore)
        {
            continue;
        }

        Vec3 fi = f.segment(3 * i, 3);

        uint joint_id;
        Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                    force_data[i]->frame_id,
                                                    fi,
                                                    joint_id);

        joint_forces[joint_id] += f_res;
    }
}

void Model::_calculateJointForcesAndJacobian(
  const VecX &q, const VecX &f,
  const std::vector<std::shared_ptr<Force>> &force_data,
  std::vector<Vec6> &joint_forces, MatX &Jf_full) const
{
    pinocchio_wrapper::computeJointJacobians(_impl, q);

    // Jf_full is reused across calls (unlike a fresh MatX::Zero() every
    // call), so it must be explicitly re-zeroed: a force marked `ignore`
    // skips writing its block below, and must still read as zero rather than
    // keep stale data from a previous call where it was active
    Jf_full.setZero();
    std::fill(joint_forces.begin(), joint_forces.end(), Vec6::Zero());

    MatX J6(6, _dims.nv);

    for (uint i = 0; i < force_data.size(); ++i)
    {
        if (force_data[i]->ignore)
        {
            continue;
        }

        frameJacobian(force_data[i]->frame_id, true, J6);
        Jf_full.block(3 * i, 0, 3, _dims.nv) = J6.topRows<3>();

        Vec3 fi = f.segment(3 * i, 3);

        uint joint_id;
        Vec6 f_res = pinocchio_wrapper::framejMfAct(_impl,
                                                    force_data[i]->frame_id,
                                                    fi,
                                                    joint_id);

        joint_forces[joint_id] += f_res;
    }
}
}
}