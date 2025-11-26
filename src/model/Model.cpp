#include "model/Model.hpp"

namespace ssqp
{
namespace model
{
Force::Force(const std::string &frame, const uint force_index)
  : frame(frame)
  , force_index(force_index)
{
}

// TODO: Add optional parameters for dt, free_flyer
// TODO: nq, nv, njoints per stage
Model::Model(
  const std::vector<std::shared_ptr<stagewise::StageCost>>        &costs,
  const std::vector<std::shared_ptr<stagewise::StageDynamics>>    &dynamics,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::shared_ptr<terrain::Terrain>                         &terrain)
  : _impl(pinocchio_wrapper::makeImpl())
  , _nq(0)
  , _nv(0)
  , _njoints(0)
  , _free_flyer(false)
  , _dt(-1.0)
  , _costs(costs)
  , _dynamics(dynamics)
  , _constraints(constraints.empty()
                   ? std::vector<std::shared_ptr<stagewise::StageConstraints>>(
                       costs.size(), nullptr)
                   : constraints)
  , _terrain(terrain)
{
    _setup();
}

void Model::_constructDynamics(const std::vector<VecX> &init_states)
{
    assert((init_states.empty() || init_states.size() == 1 ||
            init_states.size() == _dynamics.size()) &&
           "Invalid init_states size");

    // total joints (including universe, root)
    const uint total_joints = pinocchio_wrapper::modelJoints(_impl);

    if (_free_flyer)
    {
        _S = MatX::Zero(_nv, _nv);
        _S.bottomRightCorner(_nv - 6, _nv - 6) =
          MatX::Identity(_nv - 6, _nv - 6);
    }
    else
    {
        _S = MatX::Identity(_nv, _nv);
    }

    VecX init_state = VecX::Zero(0);
    if (init_states.empty())
    {
        init_state = VecX::Zero(_nq + _nv);
    }
    else if (init_states.size() == 1)
    {
        init_state = init_states[0];
    }

    auto right_plus = [this](const VecX &x, const VecX &dx)
    {
        VecX out(_nq + _nv);

        out.head(_nq) =
          pinocchio_wrapper::integrate(_impl, x.head(_nq), dx.head(_nv));
        out.tail(_nv) = x.tail(_nv) + dx.tail(_nv);

        return out;
    };

    auto right_minus = [this](const VecX &x, const VecX &y)
    {
        VecX out(2 * _nv);

        out.head(_nv) =
          pinocchio_wrapper::difference(_impl, y.head(_nq), x.head(_nq));
        out.tail(_nv) = x.tail(_nv) - y.tail(_nv);

        return out;
    };

    ManifoldOperators<decltype(right_plus), decltype(right_minus)> ops = {
        right_plus,
        right_minus
    };

    auto dyn_eval = [this](const VecX &x, const VecX &u)
    {
        VecX ddq =
          pinocchio_wrapper::aba(_impl, x.head(_nq), x.tail(_nv), _S * u);

        VecX out(_nq + _nv);
        out.head(_nq) =
          pinocchio_wrapper::integrate(_impl, x.head(_nq), x.tail(_nv) * _dt);
        out.tail(_nv) = x.tail(_nv) + ddq * _dt;

        return out;
    };

    auto dyn_grad = [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
    {
        VecX q   = x.head(_nq);
        VecX v   = x.tail(_nv);
        VecX tau = _S * u;

        MatX ddq_dq, ddq_dv, Minv;

        pinocchio_wrapper::computeABADerivatives(_impl,
                                                 q,
                                                 v,
                                                 tau,
                                                 ddq_dq,
                                                 ddq_dv,
                                                 Minv);

        MatX Jint_q = MatX::Zero(_nv, _nv);
        MatX Jint_v = MatX::Zero(_nv, _nv);

        pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_q, 0);
        pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_v, 1);

        dX                            = MatX::Zero(2 * _nv, 2 * _nv);
        dX.topLeftCorner(_nv, _nv)    = Jint_q;
        dX.topRightCorner(_nv, _nv)   = Jint_v * _dt;
        dX.bottomLeftCorner(_nv, _nv) = ddq_dq * _dt;
        dX.bottomRightCorner(_nv, _nv) =
          MatX::Identity(_nv, _nv) + ddq_dv * _dt;

        dU                 = MatX::Zero(2 * _nv, _nv);
        dU.bottomRows(_nv) = Minv * _S * _dt;
    };

    auto zero_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _nv, 2 * _nv); };

    auto zero_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _nv, _nv); };

    auto zero_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(_nv, _nv); };

    for (uint k = 0; k < _dynamics.size(); ++k)
    {
        uint nf = _forces[k].size();

        // TODO: Other integration schemes?
        if (nf == 0)
        {
            _dynamics[k] = stagewise::factory::make_dynamics(
              _nq + _nv,
              _nv,
              _nq + _nv,
              2 * _nv,
              init_state.size() != 0 ? init_state : init_states[k],
              ops,
              dyn_eval,
              dyn_grad,
              zero_hessprod_xx,
              zero_hessprod_xu,
              zero_hessprod_uu);
        }
        else
        {
            std::vector<Force> &forces = _forces[k];

            auto dyn_eval_fext =
              [this, total_joints, nf, forces](const VecX &x, const VecX &u)
            {
                VecX tau = _S * u.head(_nv);

                pinocchio_wrapper::computeJointJacobians(_impl, x.head(_nq));
                std::vector<Vec3> joint_forces(total_joints, Vec3::Zero());

                for (uint i = 0; i < nf; ++i)
                {
                    Vec3 f = u.segment(_nv + 3 * i, 3);
                    uint joint_id;
                    Vec3 translation;
                    Mat3 rotation;
                    pinocchio_wrapper::getFramejMfMatrix(_impl,
                                                         forces[i].frame,
                                                         joint_id,
                                                         rotation,
                                                         translation);

                    joint_forces[joint_id] += rotation * f;
                }

                VecX ddq = pinocchio_wrapper::aba(_impl,
                                                  x.head(_nq),
                                                  x.tail(_nv),
                                                  tau,
                                                  joint_forces);

                VecX out(_nq + _nv);
                out.head(_nq) = pinocchio_wrapper::integrate(_impl,
                                                             x.head(_nq),
                                                             x.tail(_nv) * _dt);
                out.tail(_nv) = x.tail(_nv) + ddq * _dt;

                return out;
            };

            auto dyn_grad_fext = [this, total_joints, nf, forces](const VecX &x,
                                                                  const VecX &u,
                                                                  MatX &dX,
                                                                  MatX &dU)
            {
                VecX q   = x.head(_nq);
                VecX v   = x.tail(_nv);
                VecX tau = _S * u.head(_nv);

                pinocchio_wrapper::computeJointJacobians(_impl, q);

                MatX              Jf_full = MatX::Zero(3 * nf, _nv);
                std::vector<Vec3> joint_forces(total_joints, Vec3::Zero());

                for (uint i = 0; i < nf; ++i)
                {
                    MatX Jf = frameJacobian(forces[i].frame, true).topRows<3>();
                    Jf_full.block(3 * i, 0, 3, _nv) = Jf;

                    Vec3 f = u.segment(_nv + 3 * i, 3);

                    uint joint_id;
                    Vec3 f_res = pinocchio_wrapper::framejMfAct(_impl,
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

                MatX Jint_q = MatX::Zero(_nv, _nv);
                MatX Jint_v = MatX::Zero(_nv, _nv);

                pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_q, 0);
                pinocchio_wrapper::dIntegrate(_impl, q, v * _dt, Jint_v, 1);

                dX                            = MatX::Zero(2 * _nv, 2 * _nv);
                dX.topLeftCorner(_nv, _nv)    = Jint_q;
                dX.topRightCorner(_nv, _nv)   = Jint_v * _dt;
                dX.bottomLeftCorner(_nv, _nv) = ddq_dq * _dt;
                dX.bottomRightCorner(_nv, _nv) =
                  MatX::Identity(_nv, _nv) + ddq_dv * _dt;

                dU = MatX::Zero(2 * _nv, _nv + 3 * nf);
                dU.bottomLeftCorner(_nv, _nv) = Minv * _S * _dt;
                dU.bottomRightCorner(_nv, 3 * nf) =
                  Minv * Jf_full.transpose() * _dt;
            };

            auto zero_hessprod_xu_fext =
              [this,
               nf](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
            { Pxu = MatX::Zero(2 * _nv, _nv + 3 * nf); };

            auto zero_hessprod_uu_fext =
              [this,
               nf](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
            { Puu = MatX::Zero(_nv + 3 * nf, _nv + 3 * nf); };

            _dynamics[k] = stagewise::factory::make_dynamics(
              _nq + _nv,
              _nv + 3 * nf,
              _nq + _nv,
              2 * _nv,
              init_state.size() != 0 ? init_state : init_states[k],
              ops,
              dyn_eval_fext,
              dyn_grad_fext,
              zero_hessprod_xx,
              zero_hessprod_xu_fext,
              zero_hessprod_uu_fext);
        }
    }
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

        _lin_dims.stage_state[k]   = _dynamics[k]->state_diff_size();
        _lin_dims.stage_control[k] = _dims.stage_control[k];
        _lin_dims.stage_constraint[k] =
          (_constraints[k] != nullptr) ? _constraints[k]->constraint_diff_size()
                                       : 0;
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
        ? _constraints[_lin_dims.horizon]->constraint_diff_size()
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

const std::shared_ptr<terrain::Terrain> Model::terrain() const
{
    return _terrain;
}

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX> &init_states, const bool free_flyer,
  const std::shared_ptr<terrain::Terrain> &terrain)
  : Model(urdf_path, horizon, dt, free_flyer, terrain)
{
    finalize(cost, terminal_cost, constraints, init_states);
}

Model::Model(
  const std::string &urdf_path, const uint horizon, const double dt,
  const std::vector<std::shared_ptr<stagewise::StageCost>> &costs,
  const std::shared_ptr<stagewise::StageCost>              &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX> &init_states, const bool free_flyer,
  const std::shared_ptr<terrain::Terrain> &terrain)
  : Model(urdf_path, horizon, dt, free_flyer, terrain)
{
    finalize(costs, terminal_cost, constraints, init_states);
}

Model::Model(const std::string &urdf_path, const uint horizon, const double dt,
             const bool                               free_flyer,
             const std::shared_ptr<terrain::Terrain> &terrain)
  : _impl(pinocchio_wrapper::makeImpl())
  , _free_flyer(free_flyer)
  , _dt(dt)
  , _costs(horizon + 1)
  , _dynamics(horizon + 1)
  , _constraints(horizon + 1)
  , _terrain(terrain)
  , _forces(std::vector<std::vector<Force>>(
      horizon +
      1)) // TODO: forces at terminal stage (where we have no controls)?
{
    pinocchio_wrapper::buildModelFromUrdf(urdf_path, _impl, free_flyer);

    std::tie(_nq, _nv) = pinocchio_wrapper::modelDimensions(_impl);

    std::shared_ptr<stagewise::StageConstraints> innate_constraints;
    std::shared_ptr<stagewise::StageConstraints> terminal_innate_constraints;

    VecX lb, ub;
    std::tie(lb, ub) = pinocchio_wrapper::modelBounds(_impl, free_flyer);
    _njoints         = lb.size() / 3;

    auto zero_hessprod_xx =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxx)
    { Pxx = MatX::Zero(2 * _nv, 2 * _nv); };

    auto zero_hessprod_xu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Pxu)
    { Pxu = MatX::Zero(2 * _nv, u.size()); };

    auto zero_hessprod_uu =
      [this](const VecX &x, const VecX &u, const VecX &lambda, MatX &Puu)
    { Puu = MatX::Zero(u.size(), u.size()); };

    if (free_flyer)
    {
        auto innate_constraint_eval = [this](const VecX &x, const VecX &u)
        {
            VecX c(3 * _njoints);
            c.head(_njoints)              = x.segment(7, _njoints);
            c.segment(_njoints, _njoints) = x.tail(_njoints);
            c.tail(_njoints)              = u.segment(6, _njoints);

            return c;
        };

        auto innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(3 * _njoints, 2 * _nv);
            dX.block(0, 6, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);
            dX.block(_njoints, _nv + 6, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);

            dU = MatX::Zero(3 * _njoints, u.size());
            dU.block(2 * _njoints, 6, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);
        };

        innate_constraints =
          stagewise::factory::make_constraints(3 * _njoints,
                                               3 * _njoints,
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
            VecX c(2 * _njoints);
            c.head(_njoints)              = x.segment(7, _njoints);
            c.segment(_njoints, _njoints) = x.tail(_njoints);

            return c;
        };

        auto terminal_innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(2 * _njoints, 2 * _nv);
            dX.block(0, 6, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);
            dX.block(_njoints, _nv + 6, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);

            dU = MatX::Zero(2 * _njoints, 1);
        };

        terminal_innate_constraints =
          stagewise::factory::make_constraints(2 * _njoints,
                                               2 * _njoints,
                                               lb.head(2 * _njoints),
                                               ub.head(2 * _njoints),
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
            VecX c(3 * _njoints);
            c.head(2 * _njoints) = x;
            c.tail(_njoints)     = u.head(_njoints);

            return c;
        };

        auto innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(3 * _njoints, 2 * _nv);
            dX.block(0, 0, 2 * _njoints, 2 * _njoints) =
              MatX::Identity(2 * _njoints, 2 * _njoints);

            dU = MatX::Zero(3 * _njoints, u.size());
            dU.block(2 * _njoints, 0, _njoints, _njoints) =
              MatX::Identity(_njoints, _njoints);
        };

        innate_constraints =
          stagewise::factory::make_constraints(3 * _njoints,
                                               3 * _njoints,
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
            VecX c(2 * _njoints);
            c.head(2 * _njoints) = x;

            return c;
        };

        auto terminal_innate_constraint_grad =
          [this](const VecX &x, const VecX &u, MatX &dX, MatX &dU)
        {
            dX = MatX::Zero(2 * _njoints, 2 * _nv);
            dX.block(0, 0, 2 * _njoints, 2 * _njoints) =
              MatX::Identity(2 * _njoints, 2 * _njoints);

            dU = MatX::Zero(2 * _njoints, 1);
        };

        terminal_innate_constraints =
          stagewise::factory::make_constraints(2 * _njoints,
                                               2 * _njoints,
                                               lb.head(2 * _njoints),
                                               ub.head(2 * _njoints),
                                               terminal_innate_constraint_eval,
                                               terminal_innate_constraint_grad,
                                               DefaultRightMinus,
                                               zero_hessprod_xx,
                                               zero_hessprod_xu,
                                               zero_hessprod_uu);
    }

    for (uint k = 0; k <= horizon; ++k)
    {
        _constraints[k] = innate_constraints;
    }

    _constraints[horizon] = terminal_innate_constraints;
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
    return pinocchio_wrapper::getOrderedActuatedJoints(_impl);
}

bool Model::floating_base() const
{
    return _free_flyer;
}

uint Model::nq() const
{
    return _nq;
}

uint Model::nv() const
{
    return _nv;
}

uint Model::njoints() const
{
    return _njoints;
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

std::pair<MatX, MatX> Model::frameVelocityDerivatives(const VecX        &x,
                                                      const VecX        &u,
                                                      const std::string &frame,
                                                      bool local) const
{
    VecX q = x.head(_nq);
    VecX v = x.tail(_nv);
    VecX a = u.head(_nv);

    MatX dVdq(6, _nv), dVdv(6, _nv);

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

void Model::finalize(
  const std::shared_ptr<stagewise::StageCost> &cost,
  const std::shared_ptr<stagewise::StageCost> &terminal_cost,
  const std::vector<std::shared_ptr<stagewise::StageConstraints>> &constraints,
  const std::vector<VecX>                                         &init_states)
{
    finalize(
      std::vector<std::shared_ptr<stagewise::StageCost>>(_dynamics.size(),
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
    _constructDynamics(init_states);

    uint horizon = _dynamics.size() - 1;

    assert(costs.size() == horizon + 1 && "Costs size mismatch");

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
    _constructDynamics(init_states);

    uint horizon = _dynamics.size() - 1;

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