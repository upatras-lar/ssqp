#include "model/stagewise/predefined/Costs.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace predefined
{
namespace costs
{
// SquaredJointAngleCost
SquaredJointAngleCost::SquaredJointAngleCost(const Model       &model,
                                             const double       weight,
                                             const std::string &name)
  : StageCost(name)
  , _free_flyer(model.floatingBase())
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _weight(_free_flyer ? _nq - 7 : _nq)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointAngleCost with a custom Model with no custom properties set!");
    // clang-format on
    _weight.setConstant(weight);
}

SquaredJointAngleCost::SquaredJointAngleCost(const Model       &model,
                                             const VecX        &weight,
                                             const std::string &name)
  : StageCost(name)
  , _free_flyer(model.floatingBase())
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _weight(weight)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointAngleCost with a custom Model with no custom properties set!");
    // clang-format on
    assert(_weight.size() == (_free_flyer ? _nq - 7 : _nq) &&
           "Invalid weight vector size");
}

void SquaredJointAngleCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredJointAngleCost::set_weight(const VecX &weight)
{
    assert(weight.size() == _weight.size() && "Invalid weight vector size");
    _weight = weight;
}

VecX SquaredJointAngleCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    VecX q = _free_flyer ? x.segment(7, _nq - 7) : x.head(_nq);
    return (Scalar() << 0.5 *
                          (_weight.array() * q.array()).matrix().squaredNorm())
      .finished();
}

void SquaredJointAngleCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                     VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    dx = VecX::Zero(_ndx);
    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) =
          _weight.array().square() * x.segment(7, _nq - 7).array();
    }
    else
    {
        dx.head(_nv) = _weight.array().square() * x.head(_nv).array();
    }

    du = VecX::Zero(u.size());
}
void SquaredJointAngleCost::hessian_xx(const VecX &x, const VecX &u,
                                       MatX &Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Hxx = MatX::Zero(_ndx, _ndx);

    if (_free_flyer)
    {
        Hxx.block(7, 7, _nq - 7, _nq - 7).diagonal() = _weight.array().square();
    }
    else
    {
        Hxx.topLeftCorner(_nv, _nv).diagonal() = _weight.array().square();
    }
}

void SquaredJointAngleCost::hessian_xu(const VecX &x, const VecX &u,
                                       MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void SquaredJointAngleCost::hessian_uu(const VecX &x, const VecX &u,
                                       MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// SquaredJointVelocityCost
SquaredJointVelocityCost::SquaredJointVelocityCost(const Model       &model,
                                                   const double       weight,
                                                   const std::string &name)
  : StageCost(name)
  , _nv(model.nv())
  , _njoints(model.floatingBase() ? _nv - 6 : _nv)
  , _weight(_njoints)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointVelocityCost with a custom Model with no custom properties set!");
    // clang-format on
    _weight.setConstant(weight);
}

SquaredJointVelocityCost::SquaredJointVelocityCost(const Model       &model,
                                                   const VecX        &weight,
                                                   const std::string &name)
  : StageCost(name)
  , _nv(model.nv())
  , _njoints(model.floatingBase() ? _nv - 6 : _nv)
  , _weight(weight)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointVelocityCost with a custom Model with no custom properties set!");
    // clang-format on
    assert(weight.size() == _njoints && "Invalid weight vector size");
}

void SquaredJointVelocityCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredJointVelocityCost::set_weight(const VecX &weight)
{
    assert(weight.size() == _njoints && "Invalid weight vector size");
    _weight = weight;
}

VecX SquaredJointVelocityCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    return (Scalar() << 0.5 * (_weight.array() * x.tail(_njoints).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void SquaredJointVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    dx                = VecX::Zero(2 * _nv);
    dx.tail(_njoints) = _weight.array().square() * x.tail(_njoints).array();

    du = VecX::Zero(u.size());
}

void SquaredJointVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_njoints, _njoints).diagonal() =
      _weight.array().square();
}

void SquaredJointVelocityCost::hessian_xu(const VecX &x, const VecX &u,
                                          MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void SquaredJointVelocityCost::hessian_uu(const VecX &x, const VecX &u,
                                          MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// SquaredControlCost
SquaredControlCost::SquaredControlCost(const Model &model, const double weight,
                                       const std::string &name)
  : StageCost(name)
  , _nu(model.nu())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? model.nv()
                                                      : 2 * model.nv())
  , _weight(_nu)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointControlCost with a custom Model with no custom properties set!");
    // clang-format on
    _weight.setConstant(weight);
}

SquaredControlCost::SquaredControlCost(const Model &model, const VecX &weight,
                                       const std::string &name)
  : StageCost(name)
  , _nu(model.nu())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? model.nv()
                                                      : 2 * model.nv())
  , _weight(weight)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredJointControlCost with a custom Model with no custom properties set!");
    // clang-format on
    assert(weight.size() == _nu && "Invalid weight vector size");
}

void SquaredControlCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredControlCost::set_weight(const VecX &weight)
{
    assert(weight.size() == _nu && "Invalid weight vector size");
    _weight = weight;
}

VecX SquaredControlCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    return (Scalar()
            << 0.5 *
                 (_weight.array() * u.head(_nu).array()).matrix().squaredNorm())
      .finished();
}

void SquaredControlCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                  VecX &du) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    dx           = VecX::Zero(_ndx);
    du           = VecX::Zero(u.size());
    du.head(_nu) = _weight.array().square() * u.head(_nu).array();
}

void SquaredControlCost::hessian_xx(const VecX &x, const VecX &u,
                                    MatX &Hxx) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    Hxx = MatX::Zero(_ndx, _ndx);
}

void SquaredControlCost::hessian_xu(const VecX &x, const VecX &u,
                                    MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void SquaredControlCost::hessian_uu(const VecX &x, const VecX &u,
                                    MatX &Huu) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    Huu                                    = MatX::Zero(u.size(), u.size());
    Huu.topLeftCorner(_nu, _nu).diagonal() = _weight.array().square();
}

// SquaredIDTorqueCost
SquaredIDTorqueCost::SquaredIDTorqueCost(const Model       &model,
                                         const double       weight,
                                         const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _weight(_nu)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredIDTorqueCost with a custom Model with no custom properties set!");
    // clang-format on
    assert((model.solverMode() == SolverMode::InverseDynamics ||
            model.solverMode() == SolverMode::ReducedInverseDynamics) &&
           "Cannot use SquaredIDTorqueCost with a non-ID solver mode!");
    _weight.setConstant(weight);
}
SquaredIDTorqueCost::SquaredIDTorqueCost(const Model &model, const VecX &weight,
                                         const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _weight(weight)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredIDTorqueCost with a custom Model with no custom properties set!");
    // clang-format on
    assert((model.solverMode() == SolverMode::InverseDynamics ||
            model.solverMode() == SolverMode::ReducedInverseDynamics) &&
           "Cannot use SquaredIDTorqueCost with a non-ID solver mode!");
    assert(weight.size() == _nu && "Invalid weight vector size");
}

void SquaredIDTorqueCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredIDTorqueCost::set_weight(const VecX &weight)
{
    assert(weight.size() == _nu && "Invalid weight vector size");
    _weight = weight;
}

VecX SquaredIDTorqueCost::evaluate(const VecX &x, const VecX &u) const
{
    // TODO: joint forces?
    return (Scalar()
            << 0.5 *
                 (_weight.array() *
                  _model->inverseDynamics(x.head(_nq), x.tail(_nv), u.head(_nu))
                    .array())
                   .matrix()
                   .squaredNorm())
      .finished();
}

void SquaredIDTorqueCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                   VecX &du) const
{
    VecX w_tau =
      _weight.array().square() *
      _model->inverseDynamics(x.head(_nq), x.tail(_nv), u.head(_nu)).array();

    MatX ddq_dq, ddq_dv, M;
    _model->inverseDynamicsDerivatives(x.head(_nq),
                                       x.tail(_nv),
                                       u.head(_nu),
                                       ddq_dq,
                                       ddq_dv,
                                       M);

    dx           = VecX::Zero(2 * _nv);
    dx.head(_nv) = ddq_dq.transpose() * w_tau;
    dx.tail(_nv) = ddq_dv.transpose() * w_tau;

    du           = VecX::Zero(u.size());
    du.head(_nu) = M.transpose() * w_tau;
}

void SquaredIDTorqueCost::hessian_xx(const VecX &x, const VecX &u,
                                     MatX &Hxx) const
{
    MatX ddq_dq, ddq_dv, M;
    _model->inverseDynamicsDerivatives(x.head(_nq),
                                       x.tail(_nv),
                                       u.head(_nu),
                                       ddq_dq,
                                       ddq_dv,
                                       M);

    Hxx.resize(2 * _nv, 2 * _nv);

    MatX W2 = _weight.array().square().matrix().asDiagonal();

    Hxx.topLeftCorner(_nv, _nv)    = ddq_dq.transpose() * W2 * ddq_dq;
    Hxx.topRightCorner(_nv, _nv)   = ddq_dq.transpose() * W2 * ddq_dv;
    Hxx.bottomLeftCorner(_nv, _nv) = Hxx.topRightCorner(_nv, _nv).transpose();
    Hxx.bottomLeftCorner(_nv, _nv) = ddq_dv.transpose() * W2 * ddq_dv;
}

void SquaredIDTorqueCost::hessian_xu(const VecX &x, const VecX &u,
                                     MatX &Hxu) const
{
    MatX ddq_dq, ddq_dv, M;
    _model->inverseDynamicsDerivatives(x.head(_nq),
                                       x.tail(_nv),
                                       u.head(_nu),
                                       ddq_dq,
                                       ddq_dv,
                                       M);

    Hxu.resize(2 * _nv, u.size());

    MatX W2 = _weight.array().square().matrix().asDiagonal();

    Hxu.topRows(_nv)         = ddq_dq.transpose() * W2 * M;
    Hxu.middleRows(_nv, _nu) = ddq_dq.transpose() * W2 * M;
}

void SquaredIDTorqueCost::hessian_uu(const VecX &x, const VecX &u,
                                     MatX &Huu) const
{
    MatX ddq_dq, ddq_dv, M;
    _model->inverseDynamicsDerivatives(x.head(_nq),
                                       x.tail(_nv),
                                       u.head(_nu),
                                       ddq_dq,
                                       ddq_dv,
                                       M);

    Huu.resize(u.size(), u.size());

    Huu = M.transpose() * _weight.array().square().matrix().asDiagonal() * M;
}

// SquaredForceCost
SquaredForceCost::SquaredForceCost(const Model &model, const double weight,
                                   const std::string &name)
  : StageCost(name)
  , _nv(model.nv())
  , _nu(model.nu())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _weight(weight)
  , _single_weight(true)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredForceCost with a custom Model with no custom properties set!");
    // clang-format on
}

SquaredForceCost::SquaredForceCost(const Model &model, const VecX &weight,
                                   const std::string &name)
  : StageCost(name)
  , _nv(model.nv())
  , _nu(model.nu())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _weight_vec(weight)
  , _single_weight(false)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use SquaredForceCost with a custom Model with no custom properties set!");
    // clang-format on
}

void SquaredForceCost::set_weight(const double weight)
{
    _weight        = weight;
    _single_weight = true;
}

void SquaredForceCost::set_weight(const VecX &weight)
{
    _weight_vec    = weight;
    _single_weight = false;
}

VecX SquaredForceCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    Scalar res;

    if (_single_weight)
    {
        res[0] = 0.5 * (_weight * u.tail(u.size() - _nu)).squaredNorm();
    }
    else
    {
        res[0] = 0.5 * (_weight_vec.array() * u.tail(u.size() - _nu).array())
                         .matrix()
                         .squaredNorm();
    }

    return res;
}

void SquaredForceCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                VecX &du) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    dx = VecX::Zero(_ndx);
    du = VecX::Zero(u.size());

    if (_single_weight)
    {
        du.tail(u.size() - _nu) = _weight * _weight * u.tail(u.size() - _nu);
    }
    else
    {
        du.tail(u.size() - _nu) =
          _weight_vec.array().square() * u.tail(u.size() - _nu).array();
    }
}

void SquaredForceCost::hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const
{
    Hxx = MatX::Zero(_ndx, _ndx);
}

void SquaredForceCost::hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void SquaredForceCost::hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    Huu = MatX::Zero(u.size(), u.size());
    if (_single_weight)
    {
        Huu.bottomRightCorner(u.size() - _nu, u.size() - _nu)
          .diagonal()
          .setConstant(_weight * _weight);
    }
    else
    {
        Huu.bottomRightCorner(u.size() - _nu, u.size() - _nu).diagonal() =
          _weight_vec.array().square();
    }
}

// SquaredConfigurationErrorCost
SquaredConfigurationErrorCost::SquaredConfigurationErrorCost(
  const Model &model, const VecX &desired_configuration, const double weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_configuration(desired_configuration)
  , _full_desired_configuration(_nq)
  , _free_flyer(model.floatingBase())
  , _weight(_free_flyer ? _nv - 6 : _nv)
{
    // clang-format off
    assert(
      (model.xml() || model.hasCustomProperties()) &&
      "Cannot use SquaredConfigurationErrorCost with a custom Model with no custom properties set!");
    // clang-format on
    assert(desired_configuration.size() == (_free_flyer ? _nq - 7 : _nq) &&
           "Invalid desired configuration");
    _weight.setConstant(weight);

    if (_free_flyer)
    {
        _full_desired_configuration.head<7>().setZero();
        _full_desired_configuration[3]            = 1.0;
        _full_desired_configuration.tail(_nq - 7) = _desired_configuration;
    }
    else
    {
        _full_desired_configuration = _desired_configuration;
    }
}

SquaredConfigurationErrorCost::SquaredConfigurationErrorCost(
  const Model &model, const VecX &desired_configuration, const VecX &weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_configuration(desired_configuration)
  , _full_desired_configuration(_nq)
  , _free_flyer(model.floatingBase())
  , _weight(weight)
{
    // clang-format off
    assert(
      (model.xml() || model.hasCustomProperties()) &&
      "Cannot use SquaredConfigurationErrorCost with a custom Model with no custom properties set!");
    // clang-format on
    assert(desired_configuration.size() == (_free_flyer ? _nq - 7 : _nq) &&
           "Invalid desired configuration");
    assert(weight.size() == (_free_flyer ? _nv - 6 : _nv) &&
           "Invalid weight vector size");

    if (_free_flyer)
    {
        _full_desired_configuration.head<7>().setZero();
        _full_desired_configuration.tail(_nq - 7) = _desired_configuration;
        _full_desired_configuration[3]            = 1.0;
    }
    else
    {
        _full_desired_configuration = _desired_configuration;
    }
}

void SquaredConfigurationErrorCost::set_desired_configuration(
  const VecX &desired_configuration)
{
    _desired_configuration = desired_configuration;

    if (_free_flyer)
    {
        _full_desired_configuration.head<7>().setZero();
        _full_desired_configuration[3]            = 1.0;
        _full_desired_configuration.tail(_nq - 7) = _desired_configuration;
    }
    else
    {
        _full_desired_configuration = _desired_configuration;
    }
}

void SquaredConfigurationErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredConfigurationErrorCost::set_weight(const VecX &weight)
{
    assert(weight.size() == (_free_flyer ? _nv - 7 : _nv) &&
           "Invalid weight vector size");

    _weight = weight;
}

VecX SquaredConfigurationErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    VecX diff = _model->difference(x.head(_nq), _full_desired_configuration);

    return (Scalar()
            << 0.5 * (_weight.array() *
                      (_free_flyer ? diff.segment(6, _nv - 6) : diff).array())
                       .matrix()
                       .squaredNorm())
      .finished();
}

void SquaredConfigurationErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    dx = VecX::Zero(_ndx);

    VecX diff = _model->difference(x.head(_nq), _full_desired_configuration);
    VecX weighted_diff =
      _weight.array().square() *
      (_free_flyer ? diff.segment(6, _nv - 6) : diff).array();

    MatX J = MatX::Zero(_nv, _nv);
    _model->dDifference(x.head(_nq), _full_desired_configuration, J, 0);

    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) =
          J.bottomRightCorner(_nv - 6, _nv - 6).transpose() * weighted_diff;
    }
    else
    {
        dx.head(_nv) = J.transpose() * weighted_diff;
    }

    du = VecX::Zero(u.size());
}

void SquaredConfigurationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Hxx = MatX::Zero(_ndx, _ndx);

    MatX J = MatX::Zero(_nv, _nv);
    _model->dDifference(x.head(_nq), _full_desired_configuration, J, 0);

    if (_free_flyer)
    {
        Hxx.block(6, 6, _nv - 6, _nv - 6) =
          J.bottomRightCorner(_nv - 6, _nv - 6).transpose() *
          _weight.array().square().matrix().asDiagonal() *
          J.bottomRightCorner(_nv - 6, _nv - 6);
    }
    else
    {
        Hxx.topLeftCorner(_nv, _nv) =
          J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
    }
}

void SquaredConfigurationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void SquaredConfigurationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FloatingBaseSquaredTranslationErrorCost
FloatingBaseSquaredTranslationErrorCost::
  FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                          const Vec3  &desired_translation,
                                          const double weight,
                                          const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_translation(desired_translation)
{
    assert(
      model.xml() &&
      "Cannot use FloatingBaseSquaredTranslationErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
}

FloatingBaseSquaredTranslationErrorCost::
  FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                          const Vec3  &desired_translation,
                                          const Vec3  &weight,
                                          const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_translation(desired_translation)
  , _weight(weight)
{
    assert(
      model.xml() &&
      "Cannot use FloatingBaseSquaredTranslationErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

void FloatingBaseSquaredTranslationErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_translation = desired_translation;
}

void FloatingBaseSquaredTranslationErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FloatingBaseSquaredTranslationErrorCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

VecX FloatingBaseSquaredTranslationErrorCost::evaluate(const VecX &x,
                                                       const VecX &u) const
{
    return (Scalar() << 0.5 * (_weight.array() *
                               (x.head<3>() - _desired_translation).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

// TODO: Figure out this gradient!
void FloatingBaseSquaredTranslationErrorCost::gradient(const VecX &x,
                                                       const VecX &u, VecX &dx,
                                                       VecX &du) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    dx = VecX::Zero(_ndx);
    dx.head<3>() =
      R.transpose() *
      (_weight.array().square() * (x.head<3>() - _desired_translation).array())
        .matrix();

    du = VecX::Zero(u.size());
}

// TODO: This hessian also!
void FloatingBaseSquaredTranslationErrorCost::hessian_xx(const VecX &x,
                                                         const VecX &u,
                                                         MatX       &Hxx) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.diagonal().head<3>() =
      R.transpose() * _weight.array().square().matrix();
}

void FloatingBaseSquaredTranslationErrorCost::hessian_xu(const VecX &x,
                                                         const VecX &u,
                                                         MatX       &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FloatingBaseSquaredTranslationErrorCost::hessian_uu(const VecX &x,
                                                         const VecX &u,
                                                         MatX       &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FloatingBaseSquaredRotationErrorCost
FloatingBaseSquaredRotationErrorCost::FloatingBaseSquaredRotationErrorCost(
  const Model &model, const Mat3 &desired_rotation, const double weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_rotation(desired_rotation)
{
    assert(
      model.xml() &&
      "Cannot use FloatingBaseSquaredRotationErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
}

FloatingBaseSquaredRotationErrorCost::FloatingBaseSquaredRotationErrorCost(
  const Model &model, const Mat3 &desired_rotation, const Vec3 &weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_rotation(desired_rotation)
  , _weight(weight)
{
    assert(
      model.xml() &&
      "Cannot use FloatingBaseSquaredRotationErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

void FloatingBaseSquaredRotationErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_rotation = desired_rotation;
}

void FloatingBaseSquaredRotationErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FloatingBaseSquaredRotationErrorCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

VecX FloatingBaseSquaredRotationErrorCost::evaluate(const VecX &x,
                                                    const VecX &u) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    return (Scalar() << 0.5 * (_weight.array() *
                               pinocchio_wrapper::logSO3(R.transpose() *
                                                         _desired_rotation)
                                 .array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void FloatingBaseSquaredRotationErrorCost::gradient(const VecX &x,
                                                    const VecX &u, VecX &dx,
                                                    VecX &du) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    dx = VecX::Zero(_ndx);

    Mat3 d           = R.transpose() * _desired_rotation;
    Vec3 phi         = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv     = pinocchio_wrapper::JlogSO3(d);
    dx.segment<3>(3) = _weight.array().square() *
                       ((-j_r_inv * d.transpose()).transpose() * phi).array();

    du = VecX::Zero(u.size());
}

// Hessians use GN approximation
void FloatingBaseSquaredRotationErrorCost::hessian_xx(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Hxx) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    Mat3 d       = R.transpose() * _desired_rotation;
    Vec3 phi     = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv = pinocchio_wrapper::JlogSO3(d);

    Mat3 J = -j_r_inv * d.transpose();

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.block<3, 3>(3, 3) =
      J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
}

void FloatingBaseSquaredRotationErrorCost::hessian_xu(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FloatingBaseSquaredRotationErrorCost::hessian_uu(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FloatingBaseSquaredPoseErrorCost
FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Mat4 &desired_pose, const double weight,
  const std::string &name)
  : FloatingBaseSquaredPoseErrorCost(model, desired_pose, weight, weight, name)
{
}
FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Mat4 &desired_pose, const double translation_weight,
  const double rotation_weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_pose(desired_pose)
  , _desired_pose_inv(desired_pose.inverse())
  , _weight(Mat6::Zero())
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredPoseErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    set_translation_weight(translation_weight);
    set_rotation_weight(rotation_weight);
}

FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Mat4 &desired_pose, const Vec6 &weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_pose(desired_pose)
  , _desired_pose_inv(desired_pose.inverse())
  , _weight(Mat6::Zero())
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredPoseErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    set_weight(weight);
}

FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const double weight, const std::string &name)
  : FloatingBaseSquaredPoseErrorCost(model,
                                     Mat4 {
                                       { desired_rotation(0, 0),
                                        desired_rotation(0, 1),
                                        desired_rotation(0, 2),
                                        desired_translation(0) },
                                       { desired_rotation(1, 0),
                                        desired_rotation(1, 1),
                                        desired_rotation(1, 2),
                                        desired_translation(1) },
                                       { desired_rotation(2, 0),
                                        desired_rotation(2, 1),
                                        desired_rotation(2, 2),
                                        desired_translation(2) },
                                       { 0.0, 0.0, 0.0, 1.0 }
},
                                     weight, weight, name)
{
}

FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const double translation_weight,
  const double rotation_weight, const std::string &name)
  : FloatingBaseSquaredPoseErrorCost(model,
                                     Mat4 {
                                       { desired_rotation(0, 0),
                                        desired_rotation(0, 1),
                                        desired_rotation(0, 2),
                                        desired_translation(0) },
                                       { desired_rotation(1, 0),
                                        desired_rotation(1, 1),
                                        desired_rotation(1, 2),
                                        desired_translation(1) },
                                       { desired_rotation(2, 0),
                                        desired_rotation(2, 1),
                                        desired_rotation(2, 2),
                                        desired_translation(2) },
                                       { 0.0, 0.0, 0.0, 1.0 }
},
                                     translation_weight, rotation_weight, name)
{
}

FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const Vec6 &weight, const std::string &name)
  : FloatingBaseSquaredPoseErrorCost(model,
                                     Mat4 {
                                       { desired_rotation(0, 0),
                                        desired_rotation(0, 1),
                                        desired_rotation(0, 2),
                                        desired_translation(0) },
                                       { desired_rotation(1, 0),
                                        desired_rotation(1, 1),
                                        desired_rotation(1, 2),
                                        desired_translation(1) },
                                       { desired_rotation(2, 0),
                                        desired_rotation(2, 1),
                                        desired_rotation(2, 2),
                                        desired_translation(2) },
                                       { 0.0, 0.0, 0.0, 1.0 }
},
                                     weight, name)
{
}

void FloatingBaseSquaredPoseErrorCost::set_desired_pose(
  const Mat4 &desired_pose)
{
    _desired_pose     = desired_pose;
    _desired_pose_inv = desired_pose.inverse();
}

void FloatingBaseSquaredPoseErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_pose.topRightCorner<3, 1>() = desired_translation;
    _desired_pose_inv                    = _desired_pose.inverse();
}

void FloatingBaseSquaredPoseErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_pose.topLeftCorner<3, 3>() = desired_rotation;
    _desired_pose_inv                   = _desired_pose.inverse();
}

void FloatingBaseSquaredPoseErrorCost::set_translation_weight(
  const double translation_weight)
{
    _weight.topLeftCorner<3, 3>().diagonal().setConstant(translation_weight);
}

void FloatingBaseSquaredPoseErrorCost::set_rotation_weight(
  const double rotation_weight)
{
    _weight.bottomRightCorner<3, 3>().diagonal().setConstant(rotation_weight);
}

void FloatingBaseSquaredPoseErrorCost::set_weight(const double weight)
{
    _weight.diagonal().setConstant(weight);
}

void FloatingBaseSquaredPoseErrorCost::set_weight(const Vec6 &weight)
{
    _weight.diagonal() = weight;
}

VecX FloatingBaseSquaredPoseErrorCost::evaluate(const VecX &x,
                                                const VecX &u) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Vec6 diff = pinocchio_wrapper::logSE3(_desired_pose_inv * pose);
    return (Scalar() << 0.5 * diff.transpose() * _weight * diff).finished();
}

void FloatingBaseSquaredPoseErrorCost::gradient(const VecX &x, const VecX &u,
                                                VecX &dx, VecX &du) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    dx           = VecX::Zero(_ndx);
    dx.head<6>() = Jlog6.transpose() * _weight * log6;

    du = VecX::Zero(u.size());
}

// Hessians use GN approximation
void FloatingBaseSquaredPoseErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                                  MatX &Hxx) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    Hxx                       = MatX::Zero(_ndx, _ndx);
    Hxx.topLeftCorner<6, 6>() = Jlog6.transpose() * _weight * Jlog6;
}

void FloatingBaseSquaredPoseErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                                  MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FloatingBaseSquaredPoseErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                                  MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FloatingBaseSquaredVelocityErrorCost
FloatingBaseSquaredVelocityErrorCost::FloatingBaseSquaredVelocityErrorCost(
  const Model &model, const Vec6 &desired_velocity, const double weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_velocity(desired_velocity)
{
    assert(
      model.xml() &&
      "Cannot use FloatingBaseSquaredVelocityErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
}

FloatingBaseSquaredVelocityErrorCost::FloatingBaseSquaredVelocityErrorCost(
  const Model &model, const Vec6 &desired_velocity, const Vec6 &weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _desired_velocity(desired_velocity)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredPoseErrorCost with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

void FloatingBaseSquaredVelocityErrorCost::set_desired_velocity(
  const Vec6 &desired_velocity)
{
    _desired_velocity = desired_velocity;
}

void FloatingBaseSquaredVelocityErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FloatingBaseSquaredVelocityErrorCost::set_weight(const Vec6 &weight)
{
    _weight = weight;
}

VecX FloatingBaseSquaredVelocityErrorCost::evaluate(const VecX &x,
                                                    const VecX &u) const
{
    assert(x.size() >= _nq + 6 && "Invalid state passed (size < nq + 6)");

    return (Scalar() << 0.5 * (_weight.array() *
                               (x.segment<6>(_nq) - _desired_velocity).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void FloatingBaseSquaredVelocityErrorCost::gradient(const VecX &x,
                                                    const VecX &u, VecX &dx,
                                                    VecX &du) const
{
    assert(x.size() >= _nq + 6 && "Invalid state passed (size < nq + 6)");

    dx = VecX::Zero(_ndx);

    dx.segment<6>(_nv) = _weight.array().square() *
                         (x.segment<6>(_nq) - _desired_velocity).array();

    du = VecX::Zero(u.size());
}

void FloatingBaseSquaredVelocityErrorCost::hessian_xx(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Hxx) const
{
    Hxx                                  = MatX::Zero(_ndx, _ndx);
    Hxx.block<6, 6>(_nv, _nv).diagonal() = _weight.array().square();
}

void FloatingBaseSquaredVelocityErrorCost::hessian_xu(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FloatingBaseSquaredVelocityErrorCost::hessian_uu(const VecX &x,
                                                      const VecX &u,
                                                      MatX       &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredTranslationErrorCost
FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const double weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_translation(desired_translation)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredTranslationErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
}

FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const Vec3 &weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_translation(desired_translation)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredTranslationErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameSquaredTranslationErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_translation = desired_translation;
}

void FrameSquaredTranslationErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FrameSquaredTranslationErrorCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

VecX FrameSquaredTranslationErrorCost::evaluate(const VecX &x,
                                                const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);

    return (Scalar() << 0.5 * (_weight.array() *
                               (pos - _desired_translation).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void FrameSquaredTranslationErrorCost::gradient(const VecX &x, const VecX &u,
                                                VecX &dx, VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk                   = _model->frameJacobian(x.head(_nq), _frame);

    dx = VecX::Zero(_ndx);

    dx.head(_nv) =
      Jfk.topRows(3).transpose() *
      (_weight.array().square() * (pos - _desired_translation).array())
        .matrix();

    du = VecX::Zero(u.size());
}

void FrameSquaredTranslationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                                  MatX &Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.topLeftCorner(_nv, _nv) =
      Jfk.topRows(3).transpose() *
      _weight.array().square().matrix().asDiagonal() * Jfk.topRows(3);
}

void FrameSquaredTranslationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                                  MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FrameSquaredTranslationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                                  MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredRotationErrorCost
FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const std::string &frame, const Mat3 &desired_rotation,
  const double weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_rotation(desired_rotation)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredRotationErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
}

FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const std::string &frame, const Mat3 &desired_rotation,
  const Vec3 &weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_rotation(desired_rotation)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredRotationErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameSquaredRotationErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_rotation = desired_rotation;
}

void FrameSquaredRotationErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FrameSquaredRotationErrorCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

VecX FrameSquaredRotationErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 R;
    std::tie(std::ignore, R) = _model->forwardKinematics(x.head(_nq), _frame);

    return (Scalar() << 0.5 * (_weight.array() *
                               pinocchio_wrapper::logSO3(R.transpose() *
                                                         _desired_rotation)
                                 .array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void FrameSquaredRotationErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 R;
    std::tie(std::ignore, R) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk                 = _model->frameJacobian(x.head(_nq), _frame, true);

    dx = VecX::Zero(_ndx);

    Mat3 d       = R.transpose() * _desired_rotation;
    Vec3 phi     = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv = pinocchio_wrapper::JlogSO3(d);
    MatX jac     = Jfk.bottomRows(3);
    dx.head(_nv) = (-j_r_inv * _weight.array().square().matrix().asDiagonal() *
                    d.transpose() * jac)
                     .transpose() *
                   phi;

    du = VecX::Zero(u.size());
}

void FrameSquaredRotationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk     = _model->frameJacobian(x.head(_nq), _frame, true);
    Mat3 d       = rot.transpose() * _desired_rotation;
    Vec3 phi     = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv = pinocchio_wrapper::JlogSO3(d);
    MatX jac     = Jfk.bottomRows(3);

    MatX J = -j_r_inv * d.transpose() * jac;

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.topLeftCorner(_nv, _nv) =
      J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
}

void FrameSquaredRotationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FrameSquaredRotationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredPoseErrorCost
FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(const Model       &model,
                                                     const std::string &frame,
                                                     const Mat4  &desired_pose,
                                                     const double weight,
                                                     const std::string &name)
  : FrameSquaredPoseErrorCost(model, frame, desired_pose, weight, weight, name)
{
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const std::string &frame, const Mat4 &desired_pose,
  const double translation_weight, const double rotation_weight,
  const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_pose(desired_pose)
  , _desired_pose_inv(desired_pose.inverse())
  , _weight(Mat6::Zero())
{
    assert(model.xml() &&
           "Cannot use FrameSquaredPoseErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    set_translation_weight(translation_weight);
    set_rotation_weight(rotation_weight);
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(const Model       &model,
                                                     const std::string &frame,
                                                     const Mat4 &desired_pose,
                                                     const Vec6 &weight,
                                                     const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_pose(desired_pose)
  , _desired_pose_inv(desired_pose.inverse())
  , _weight(Mat6::Zero())
{
    assert(model.xml() &&
           "Cannot use FrameSquaredPoseErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    set_weight(weight);
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const double weight, const std::string &name)
  : FrameSquaredPoseErrorCost(model, frame,
                              Mat4 {
                                { desired_rotation(0, 0),
                                 desired_rotation(0, 1),
                                 desired_rotation(0, 2),
                                 desired_translation(0) },
                                { desired_rotation(1, 0),
                                 desired_rotation(1, 1),
                                 desired_rotation(1, 2),
                                 desired_translation(1) },
                                { desired_rotation(2, 0),
                                 desired_rotation(2, 1),
                                 desired_rotation(2, 2),
                                 desired_translation(2) },
                                { 0.0, 0.0, 0.0, 1.0 }
},
                              weight, weight, name)
{
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const double translation_weight,
  const double rotation_weight, const std::string &name)
  : FrameSquaredPoseErrorCost(model, frame,
                              Mat4 {
                                { desired_rotation(0, 0),
                                 desired_rotation(0, 1),
                                 desired_rotation(0, 2),
                                 desired_translation(0) },
                                { desired_rotation(1, 0),
                                 desired_rotation(1, 1),
                                 desired_rotation(1, 2),
                                 desired_translation(1) },
                                { desired_rotation(2, 0),
                                 desired_rotation(2, 1),
                                 desired_rotation(2, 2),
                                 desired_translation(2) },
                                { 0.0, 0.0, 0.0, 1.0 }
},
                              translation_weight, rotation_weight, name)
{
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const Vec6 &weight, const std::string &name)
  : FrameSquaredPoseErrorCost(model, frame,
                              Mat4 {
                                { desired_rotation(0, 0),
                                 desired_rotation(0, 1),
                                 desired_rotation(0, 2),
                                 desired_translation(0) },
                                { desired_rotation(1, 0),
                                 desired_rotation(1, 1),
                                 desired_rotation(1, 2),
                                 desired_translation(1) },
                                { desired_rotation(2, 0),
                                 desired_rotation(2, 1),
                                 desired_rotation(2, 2),
                                 desired_translation(2) },
                                { 0.0, 0.0, 0.0, 1.0 }
},
                              weight, name)
{
}

void FrameSquaredPoseErrorCost::set_desired_pose(const Mat4 &desired_pose)
{
    _desired_pose     = desired_pose;
    _desired_pose_inv = desired_pose.inverse();
}

void FrameSquaredPoseErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_pose.block<3, 1>(0, 3) = desired_translation;
    _desired_pose_inv               = _desired_pose.inverse();
}

void FrameSquaredPoseErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_pose.block<3, 3>(0, 0) = desired_rotation;
    _desired_pose_inv               = _desired_pose.inverse();
}

void FrameSquaredPoseErrorCost::set_translation_weight(
  const double translation_weight)
{
    _weight.topLeftCorner<3, 3>().diagonal().setConstant(translation_weight);
}

void FrameSquaredPoseErrorCost::set_rotation_weight(
  const double rotation_weight)
{
    _weight.bottomRightCorner<3, 3>().diagonal().setConstant(rotation_weight);
}

void FrameSquaredPoseErrorCost::set_weight(const double weight)
{
    _weight.diagonal().setConstant(weight);
}

void FrameSquaredPoseErrorCost::set_weight(const Vec6 &weight)
{
    _weight.diagonal() = weight;
}

VecX FrameSquaredPoseErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);

    Vec6 diff = pinocchio_wrapper::logSE3(_desired_pose_inv * pose);
    return (Scalar() << 0.5 * diff.transpose() * _weight * diff).finished();
}

void FrameSquaredPoseErrorCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                         VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);
    MatX Jfk  = _model->frameJacobian(x.head(_nq), _frame, true);

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    dx           = VecX::Zero(_ndx);
    dx.head(_nv) = (Jlog6 * Jfk).transpose() * _weight * log6;

    du = VecX::Zero(u.size());
}

void FrameSquaredPoseErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                           MatX &Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);
    MatX Jfk  = _model->frameJacobian(x.head(_nq), _frame, true);

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    MatX J = Jlog6 * Jfk;

    Hxx                         = MatX::Zero(_ndx, _ndx);
    Hxx.topLeftCorner(_nv, _nv) = J.transpose() * _weight * J;
}

void FrameSquaredPoseErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                           MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FrameSquaredPoseErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                           MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredVelocityErrorCost
FrameSquaredVelocityErrorCost::FrameSquaredVelocityErrorCost(
  const Model &model, const std::string &frame, const Vec6 &desired_velocity,
  const double weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_velocity(desired_velocity)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredVelocityErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
}

FrameSquaredVelocityErrorCost::FrameSquaredVelocityErrorCost(
  const Model &model, const std::string &frame, const Vec6 &desired_velocity,
  const Vec6 &weight, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _desired_velocity(desired_velocity)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FrameSquaredPoseErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameSquaredVelocityErrorCost::set_desired_velocity(
  const Vec6 &desired_velocity)
{
    _desired_velocity = desired_velocity;
}

void FrameSquaredVelocityErrorCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FrameSquaredVelocityErrorCost::set_weight(const Vec6 &weight)
{
    _weight = weight;
}

VecX FrameSquaredVelocityErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    assert(x.size() >= _nq + _nv && "Invalid state passed (size < nq + nv)");

    return (Scalar() << 0.5 *
                          (_weight.array() * (_model->frameVelocity(x.head(_nq),
                                                                    x.tail(_nv),
                                                                    _frame,
                                                                    true) -
                                              _desired_velocity)
                                               .array())
                            .matrix()
                            .squaredNorm())
      .finished();
}

void FrameSquaredVelocityErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    assert(x.size() >= _nq + _nv && "Invalid state passed (size < nq + nv)");

    Vec6 diff = _weight.array().square() *
                (_model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true) -
                 _desired_velocity)
                  .array();

    MatX dVdq, dVdv;
    std::tie(dVdq, dVdv) =
      _model->frameVelocityDerivatives(x.head(_nq), x.tail(_nv), _frame, true);

    dx = VecX::Zero(_ndx);

    dx.head(_nv) = dVdq.transpose() * diff;
    dx.tail(_nv) = dVdv.transpose() * diff;

    du = VecX::Zero(u.size());
}

void FrameSquaredVelocityErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.bottomRightCorner(_nv, _nv) =
      Jfk.transpose() * _weight.array().square().matrix().asDiagonal() * Jfk;
}
void FrameSquaredVelocityErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}
void FrameSquaredVelocityErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// CenterOfMassTrackingCost
CenterOfMassTrackingCost::CenterOfMassTrackingCost(const Model       &model,
                                                   const Vec3        &cref,
                                                   const double       weight,
                                                   const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _cref(cref)
{
    assert(model.xml() &&
           "Cannot use CenterOfMassTrackingCost with a custom Model");
    _weight.setConstant(weight);
}

CenterOfMassTrackingCost::CenterOfMassTrackingCost(const Model       &model,
                                                   const Vec3        &cref,
                                                   const Vec3        &weight,
                                                   const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _cref(cref)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use CenterOfMassTrackingCost with a custom Model");
}

void CenterOfMassTrackingCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void CenterOfMassTrackingCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

void CenterOfMassTrackingCost::set_cref(const Vec3 &cref)
{
    _cref = cref;
}

VecX CenterOfMassTrackingCost::evaluate(const VecX &x, const VecX &u) const
{
    Vec3 CoM = _model->centerOfMass(x.head(_nq));

    return (Scalar() << 0.5 * (_weight.array() * (CoM - _cref).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void CenterOfMassTrackingCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    const VecX q = x.head(_nq);

    Vec3 CoM  = _model->centerOfMass(q);
    MatX JCoM = _model->jacobianCenterOfMass(q);

    dx = VecX::Zero(_ndx);
    dx.head(_nv).noalias() =
      JCoM.transpose() *
      (_weight.array().square() * (CoM - _cref).array()).matrix();

    du = VecX::Zero(u.size());
}

void CenterOfMassTrackingCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    MatX JCoM = _model->jacobianCenterOfMass(x.head(_nq));

    Hxx = MatX::Zero(_ndx, _ndx);
    Hxx.topLeftCorner(_nv, _nv) =
      JCoM.transpose() * _weight.array().square().matrix().asDiagonal() * JCoM;
}

void CenterOfMassTrackingCost::hessian_xu(const VecX &x, const VecX &u,
                                          MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void CenterOfMassTrackingCost::hessian_uu(const VecX &x, const VecX &u,
                                          MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FramesInverseSquaredDistanceCost
FramesInverseSquaredDistanceCost::FramesInverseSquaredDistanceCost(
  const Model &model, const std::vector<std::string> &frames,
  const double weight, const double distance_scale, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frames(frames)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FramesInverseSquaredDistanceCost with a custom Model");
    for (uint i = 0; i < frames.size(); ++i)
    {
        assert(model.hasFrame(frames[i]) && "Invalid frame for given model");
    }
    set_distance_scale(distance_scale);
}

FramesInverseSquaredDistanceCost::FramesInverseSquaredDistanceCost(
  const Model &model, const std::vector<std::string> &frames,
  const double weight, const Vec3 &distance_scale, const std::string &name)
  : StageCost(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frames(frames)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FramesInverseSquaredDistanceCost with a custom Model");
    for (uint i = 0; i < frames.size(); ++i)
    {
        assert(model.hasFrame(frames[i]) && "Invalid frame for given model");
    }
    set_distance_scale(distance_scale);
}

void FramesInverseSquaredDistanceCost::set_weight(const double weight)
{
    _weight = weight;
}

void FramesInverseSquaredDistanceCost::set_distance_scale(
  const double distance_scale)
{
    assert(distance_scale > 0.0 && "Distance scale must be positive");
    _distance_scale.setConstant(distance_scale);
}

void FramesInverseSquaredDistanceCost::set_distance_scale(
  const Vec3 &distance_scale)
{
    assert((distance_scale.array() > 0.0).all() &&
           "Distance scale must be positive");
    _distance_scale = distance_scale;
}

VecX FramesInverseSquaredDistanceCost::evaluate(const VecX &x,
                                                const VecX &u) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    std::vector<Vec3> translations(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frames[i]);
    }

    Scalar res = Scalar::Zero();

    for (uint i = 0; i < _frames.size(); ++i)
    {
        for (uint j = i + 1; j < _frames.size(); ++j)
        {
            res[0] +=
              _weight / (2 * (_distance_scale.array() *
                              (translations[i] - translations[j]).array())
                               .matrix()
                               .squaredNorm());
        }
    }

    return res;
}

void FramesInverseSquaredDistanceCost::gradient(const VecX &x, const VecX &u,
                                                VecX &dx, VecX &du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    std::vector<Vec3> translations(_frames.size());
    std::vector<MatX> jacobians(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frames[i]);
        jacobians[i] =
          _model->frameJacobian(x.head(_nq), _frames[i]).topRows<3>();
    }

    dx = VecX::Zero(_ndx);

    Vec3 d2 = _distance_scale.array().square();

    for (uint i = 0; i < _frames.size(); ++i)
    {
        for (uint j = i + 1; j < _frames.size(); ++j)
        {
            Vec3 d = translations[i] - translations[j];

            double s =
              (_distance_scale.array() * d.array()).matrix().squaredNorm();

            Vec3 g_d = -(d2.array() * d.array()).matrix() / (s * s);

            dx.head(_nv) +=
              _weight * (jacobians[i] - jacobians[j]).transpose() * g_d;
        }
    }

    du = VecX::Zero(u.size());
}

void FramesInverseSquaredDistanceCost::hessian_xx(const VecX &x, const VecX &u,
                                                  MatX &Hxx) const
{
    std::vector<Vec3> translations(_frames.size());
    std::vector<MatX> jacobians(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frames[i]);
        jacobians[i] =
          _model->frameJacobian(x.head(_nq), _frames[i]).topRows<3>();
    }

    Hxx = MatX::Zero(_ndx, _ndx);

    Vec3 d2 = _distance_scale.array().square();

    for (uint i = 0; i < _frames.size(); ++i)
    {
        for (uint j = i + 1; j < _frames.size(); ++j)
        {
            Vec3 d  = translations[i] - translations[j];
            MatX Jd = jacobians[i] - jacobians[j];

            double s =
              (_distance_scale.array() * d.array()).matrix().squaredNorm();

            Vec3 Ad = d2.array() * d.array();

            Mat3 Hd  = (-d2 / (s * s)).asDiagonal();
            Hd      += 4.0 * (Ad * Ad.transpose()) / (s * s * s);

            Hxx.topLeftCorner(_nv, _nv) += _weight * Jd.transpose() * Hd * Jd;
        }
    }
}

void FramesInverseSquaredDistanceCost::hessian_xu(const VecX &x, const VecX &u,
                                                  MatX &Hxu) const
{
    Hxu = MatX::Zero(_ndx, u.size());
}

void FramesInverseSquaredDistanceCost::hessian_uu(const VecX &x, const VecX &u,
                                                  MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

}
}
}
}
}