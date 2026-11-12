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
// PredefinedStageCost
PredefinedStageCost::PredefinedStageCost(const Model       &model,
                                         const std::string &name)
  : StageCost(name)
  , _ndx(model.solverMode() == SolverMode::Kinematics ? model.nv()
                                                      : 2 * model.nv())
{
}

void PredefinedStageCost::evaluate(ConstVecXRef x, ConstVecXRef u,
                                   VecX &out) const
{
    out.resize(1);
    evaluate_into(x, u, out);
}

void PredefinedStageCost::gradient(ConstVecXRef x, ConstVecXRef u, VecX &dx,
                                   VecX &du) const
{
    dx.setZero(_ndx);
    du.setZero(u.size());
    add_gradient(x, u, dx, du);
}

void PredefinedStageCost::hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                     MatX &Hxx) const
{
    Hxx.setZero(_ndx, _ndx);
    add_hessian_xx(x, u, Hxx);
}

void PredefinedStageCost::hessian_xu(ConstVecXRef x, ConstVecXRef u,
                                     MatX &Hxu) const
{
    Hxu.setZero(_ndx, u.size());
    add_hessian_xu(x, u, Hxu);
}

void PredefinedStageCost::hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                     MatX &Huu) const
{
    Huu.setZero(u.size(), u.size());
    add_hessian_uu(x, u, Huu);
}

void PredefinedStageCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Hxx) const
{
}

void PredefinedStageCost::add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Hxu) const
{
}

void PredefinedStageCost::add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Huu) const
{
}

// SquaredJointAngleCost
SquaredJointAngleCost::SquaredJointAngleCost(const Model       &model,
                                             const double       weight,
                                             const std::string &name)
  : PredefinedStageCost(model, name)
  , _free_flyer(model.floatingBase())
  , _nq(model.nq())
  , _nv(model.nv())
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
  : PredefinedStageCost(model, name)
  , _free_flyer(model.floatingBase())
  , _nq(model.nq())
  , _nv(model.nv())
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

void SquaredJointAngleCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                          VecXRef out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    ConstVecXRef q = _free_flyer ? x.segment(7, _nq - 7) : x.head(_nq);
    out[0]         = 0.5 * (_weight.array() * q.array()).matrix().squaredNorm();
}

void SquaredJointAngleCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                         VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) +=
          (_weight.array().square() * x.segment(7, _nq - 7).array()).matrix();
    }
    else
    {
        dx.head(_nv) +=
          (_weight.array().square() * x.head(_nv).array()).matrix();
    }
}
void SquaredJointAngleCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                           MatXRef Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    if (_free_flyer)
    {
        Hxx.block(6, 6, _nv - 6, _nv - 6).diagonal() +=
          _weight.array().square().matrix();
    }
    else
    {
        Hxx.topLeftCorner(_nv, _nv).diagonal() +=
          _weight.array().square().matrix();
    }
}

// SquaredJointVelocityCost
SquaredJointVelocityCost::SquaredJointVelocityCost(const Model       &model,
                                                   const double       weight,
                                                   const std::string &name)
  : PredefinedStageCost(model, name)
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
  : PredefinedStageCost(model, name)
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

void SquaredJointVelocityCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                             VecXRef out) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    out[0] =
      0.5 * (_weight.array() * x.tail(_njoints).array()).matrix().squaredNorm();
}

void SquaredJointVelocityCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                            VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    dx.tail(_njoints) +=
      (_weight.array().square() * x.tail(_njoints).array()).matrix();
}

void SquaredJointVelocityCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                              MatXRef Hxx) const
{
    assert(x.size() >= _njoints && "Invalid state passed (size < njoints)");

    Hxx.bottomRightCorner(_njoints, _njoints).diagonal() +=
      _weight.array().square().matrix();
}

// SquaredControlCost
SquaredControlCost::SquaredControlCost(const Model &model, const double weight,
                                       const std::string &name)
  : PredefinedStageCost(model, name)
  , _nu(model.nu())
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
  : PredefinedStageCost(model, name)
  , _nu(model.nu())
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

void SquaredControlCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                       VecXRef out) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    out[0] =
      0.5 * (_weight.array() * u.head(_nu).array()).matrix().squaredNorm();
}

void SquaredControlCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                      VecXRef dx, VecXRef du) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    du.head(_nu) += (_weight.array().square() * u.head(_nu).array()).matrix();
}

void SquaredControlCost::add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                        MatXRef Huu) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    Huu.topLeftCorner(_nu, _nu).diagonal() += _weight.array().square().matrix();
}

// SquaredIDTorqueCost
SquaredIDTorqueCost::SquaredIDTorqueCost(const Model       &model,
                                         const double       weight,
                                         const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _weight(_nu)
  , _tau_scratch(_nu)
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
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _weight(weight)
  , _tau_scratch(_nu)
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

void SquaredIDTorqueCost::set_stage_forces(
  const std::vector<std::shared_ptr<Force>> &forces,
  const std::vector<std::shared_ptr<Force>> &wrenches)
{
    if (_has_stage_forces && (forces != _forces || wrenches != _wrenches))
    {
        throw std::runtime_error(
          "SquaredIDTorqueCost was added to stages with different forces; "
          "construct one per stage instead");
    }

    _forces           = forces;
    _wrenches         = wrenches;
    _has_stage_forces = true;
}

void SquaredIDTorqueCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                        VecXRef out) const
{
    _model->getActuatedTorqueVector(x, u, _tau_scratch, _forces, _wrenches);

    out[0] =
      0.5 * (_weight.array() * _tau_scratch.array()).matrix().squaredNorm();
}

void SquaredIDTorqueCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                       VecXRef dx, VecXRef du) const
{
    // The torque weighted by the squared weights
    VecX &w_tau = _tau_scratch;
    _model->getActuatedTorqueVector(x, u, w_tau, _forces, _wrenches);
    w_tau.array() *= _weight.array().square();

    MatX dtau_dx, dtau_du;
    _model->getActuatedTorqueVectorDerivatives(x,
                                               u,
                                               dtau_dx,
                                               dtau_du,
                                               _forces,
                                               _wrenches);

    dx += dtau_dx.transpose() * w_tau;
    du += dtau_du.transpose() * w_tau;
}

void SquaredIDTorqueCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Hxx) const
{
    MatX dtau_dx, dtau_du;
    _model->getActuatedTorqueVectorDerivatives(x,
                                               u,
                                               dtau_dx,
                                               dtau_du,
                                               _forces,
                                               _wrenches);

    MatX W2 = _weight.array().square().matrix().asDiagonal();

    Hxx += dtau_dx.transpose() * W2 * dtau_dx;
}

void SquaredIDTorqueCost::add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Hxu) const
{
    MatX dtau_dx, dtau_du;
    _model->getActuatedTorqueVectorDerivatives(x,
                                               u,
                                               dtau_dx,
                                               dtau_du,
                                               _forces,
                                               _wrenches);

    MatX W2 = _weight.array().square().matrix().asDiagonal();

    Hxu += dtau_dx.transpose() * W2 * dtau_du;
}

void SquaredIDTorqueCost::add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                         MatXRef Huu) const
{
    MatX dtau_dx, dtau_du;
    _model->getActuatedTorqueVectorDerivatives(x,
                                               u,
                                               dtau_dx,
                                               dtau_du,
                                               _forces,
                                               _wrenches);

    Huu += dtau_du.transpose() *
           _weight.array().square().matrix().asDiagonal() * dtau_du;
}

// SquaredForceCost
SquaredForceCost::SquaredForceCost(const Model &model, const double weight,
                                   const std::string &name)
  : PredefinedStageCost(model, name)
  , _nv(model.nv())
  , _nu(model.nu())
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
  : PredefinedStageCost(model, name)
  , _nv(model.nv())
  , _nu(model.nu())
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

void SquaredForceCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                     VecXRef out) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    if (_single_weight)
    {
        out[0] = 0.5 * (_weight * u.tail(u.size() - _nu)).squaredNorm();
    }
    else
    {
        out[0] = 0.5 * (_weight_vec.array() * u.tail(u.size() - _nu).array())
                         .matrix()
                         .squaredNorm();
    }
}

void SquaredForceCost::add_gradient(ConstVecXRef x, ConstVecXRef u, VecXRef dx,
                                    VecXRef du) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    if (_single_weight)
    {
        du.tail(u.size() - _nu) += _weight * _weight * u.tail(u.size() - _nu);
    }
    else
    {
        du.tail(u.size() - _nu) +=
          (_weight_vec.array().square() * u.tail(u.size() - _nu).array())
            .matrix();
    }
}

void SquaredForceCost::add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                      MatXRef Huu) const
{
    assert(u.size() >= _nu && "Invalid control passed (size < nu)");

    if (_single_weight)
    {
        Huu.bottomRightCorner(u.size() - _nu, u.size() - _nu)
          .diagonal()
          .array() += _weight * _weight;
    }
    else
    {
        Huu.bottomRightCorner(u.size() - _nu, u.size() - _nu).diagonal() +=
          _weight_vec.array().square().matrix();
    }
}

// SquaredConfigurationErrorCost
SquaredConfigurationErrorCost::SquaredConfigurationErrorCost(
  const Model &model, const VecX &desired_configuration, const double weight,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_configuration(desired_configuration)
  , _full_desired_configuration(_nq)
  , _free_flyer(model.floatingBase())
  , _weight(_free_flyer ? _nv - 6 : _nv)
  , _diff_scratch(_nv)
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
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_configuration(desired_configuration)
  , _full_desired_configuration(_nq)
  , _free_flyer(model.floatingBase())
  , _weight(weight)
  , _diff_scratch(_nv)
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
    assert(weight.size() == (_free_flyer ? _nv - 6 : _nv) &&
           "Invalid weight vector size");

    _weight = weight;
}

void SquaredConfigurationErrorCost::evaluate_into(ConstVecXRef x,
                                                  ConstVecXRef u,
                                                  VecXRef      out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    VecX &diff = _diff_scratch;
    _model->difference(x.head(_nq), _full_desired_configuration, diff);

    out[0] =
      0.5 * (_weight.array() * diff.tail(_free_flyer ? _nv - 6 : _nv).array())
              .matrix()
              .squaredNorm();
}

void SquaredConfigurationErrorCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                                 VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    VecX &diff = _diff_scratch;
    _model->difference(x.head(_nq), _full_desired_configuration, diff);
    auto weighted_diff     = diff.tail(_free_flyer ? _nv - 6 : _nv);
    weighted_diff.array() *= _weight.array().square();

    MatX &J = _jac_scratch;
    J.setZero(_nv, _nv);
    _model->dDifference(x.head(_nq), _full_desired_configuration, J, 0);

    if (_free_flyer)
    {
        dx.segment(6, _nv - 6).noalias() +=
          J.bottomRightCorner(_nv - 6, _nv - 6).transpose() * weighted_diff;
    }
    else
    {
        dx.head(_nv).noalias() += J.transpose() * weighted_diff;
    }
}

void SquaredConfigurationErrorCost::add_hessian_xx(ConstVecXRef x,
                                                   ConstVecXRef u,
                                                   MatXRef      Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    MatX &J = _jac_scratch;
    J.setZero(_nv, _nv);
    _model->dDifference(x.head(_nq), _full_desired_configuration, J, 0);

    // Holds the transposed Jacobian times the squared weights
    MatX &JtW = _weighted_jac_scratch;
    if (_free_flyer)
    {
        JtW.noalias() = J.bottomRightCorner(_nv - 6, _nv - 6).transpose() *
                        _weight.array().square().matrix().asDiagonal();
        Hxx.block(6, 6, _nv - 6, _nv - 6).noalias() +=
          JtW * J.bottomRightCorner(_nv - 6, _nv - 6);
    }
    else
    {
        JtW.noalias() =
          J.transpose() * _weight.array().square().matrix().asDiagonal();
        Hxx.topLeftCorner(_nv, _nv).noalias() += JtW * J;
    }
}

// FloatingBaseSquaredTranslationErrorCost
FloatingBaseSquaredTranslationErrorCost::
  FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                          const Vec3  &desired_translation,
                                          const double weight,
                                          const ReferenceFrame &reference_frame,
                                          const std::string    &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_translation(desired_translation)
{
    assert(model.xml() && "Cannot use FloatingBaseSquaredTranslationErrorCost "
                          "with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
    set_reference_frame(reference_frame);
}

FloatingBaseSquaredTranslationErrorCost::
  FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                          const Vec3  &desired_translation,
                                          const Vec3  &weight,
                                          const ReferenceFrame &reference_frame,
                                          const std::string    &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_translation(desired_translation)
  , _weight(weight)
{
    assert(model.xml() && "Cannot use FloatingBaseSquaredTranslationErrorCost "
                          "with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    set_reference_frame(reference_frame);
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

void FloatingBaseSquaredTranslationErrorCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FloatingBaseSquaredTranslationErrorCost");
    // The base's own axes are the base axes
    _reference_frame = reference_frame.type() == ReferenceFrame::Local
                         ? ReferenceFrame(ReferenceFrame::Base)
                         : reference_frame;
}

void FloatingBaseSquaredTranslationErrorCost::evaluate_into(ConstVecXRef x,
                                                            ConstVecXRef u,
                                                            VecXRef out) const
{
    const Vec3 error = _reference_frame.axes(x).transpose() *
                       (x.head<3>() - _desired_translation);

    out[0] = 0.5 * (_weight.array() * error.array()).matrix().squaredNorm();
}

void FloatingBaseSquaredTranslationErrorCost::add_gradient(ConstVecXRef x,
                                                           ConstVecXRef u,
                                                           VecXRef      dx,
                                                           VecXRef du) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    const Mat3 A     = _reference_frame.axes(x);
    const Vec3 error = A.transpose() * (x.head<3>() - _desired_translation);

    // The error's Jacobian over the base's tangent. Base axes turn with the
    // base, which adds the error crossed with its angular tangent
    Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
    J.leftCols<3>()               = A.transpose() * R;
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        J.rightCols<3>() = skew(error);
    }

    dx.head<6>() +=
      J.transpose() * (_weight.array().square() * error.array()).matrix();
}

void FloatingBaseSquaredTranslationErrorCost::add_hessian_xx(ConstVecXRef x,
                                                             ConstVecXRef u,
                                                             MatXRef Hxx) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    const Mat3 A = _reference_frame.axes(x);

    // Gauss-Newton Hessian of the error through the base's tangent step
    Eigen::Matrix<double, 3, 6> J = Eigen::Matrix<double, 3, 6>::Zero();
    J.leftCols<3>()               = A.transpose() * R;
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        J.rightCols<3>() =
          skew(A.transpose() * (x.head<3>() - _desired_translation));
    }

    Hxx.topLeftCorner<6, 6>() +=
      J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
}

// FloatingBaseSquaredRotationErrorCost
FloatingBaseSquaredRotationErrorCost::FloatingBaseSquaredRotationErrorCost(
  const Model &model, const Mat3 &desired_rotation, const double weight,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_rotation(desired_rotation)
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredRotationErrorCost with a "
           "custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
}

FloatingBaseSquaredRotationErrorCost::FloatingBaseSquaredRotationErrorCost(
  const Model &model, const Mat3 &desired_rotation, const Vec3 &weight,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_rotation(desired_rotation)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredRotationErrorCost with a "
           "custom Model");
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

void FloatingBaseSquaredRotationErrorCost::evaluate_into(ConstVecXRef x,
                                                         ConstVecXRef u,
                                                         VecXRef      out) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    out[0] =
      0.5 *
      (_weight.array() *
       pinocchio_wrapper::logSO3(R.transpose() * _desired_rotation).array())
        .matrix()
        .squaredNorm();
}

void FloatingBaseSquaredRotationErrorCost::add_gradient(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        VecXRef      dx,
                                                        VecXRef      du) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    Mat3 d            = R.transpose() * _desired_rotation;
    Vec3 phi          = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv      = pinocchio_wrapper::JlogSO3(d);
    dx.segment<3>(3) += (_weight.array().square() *
                         ((-j_r_inv * d.transpose()).transpose() * phi).array())
                          .matrix();
}

// Hessians use GN approximation
void FloatingBaseSquaredRotationErrorCost::add_hessian_xx(ConstVecXRef x,
                                                          ConstVecXRef u,
                                                          MatXRef Hxx) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    Mat3 d       = R.transpose() * _desired_rotation;
    Vec3 phi     = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv = pinocchio_wrapper::JlogSO3(d);

    Mat3 J = -j_r_inv * d.transpose();

    Hxx.block<3, 3>(3, 3) +=
      J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
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
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_pose(desired_pose)
  , _desired_pose_inv(inverseSE3(desired_pose))
  , _weight(Mat6::Zero())
{
    assert(model.xml() && "Cannot use FloatingBaseSquaredPoseErrorCost with a "
                          "custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    set_translation_weight(translation_weight);
    set_rotation_weight(rotation_weight);
}

FloatingBaseSquaredPoseErrorCost::FloatingBaseSquaredPoseErrorCost(
  const Model &model, const Mat4 &desired_pose, const Vec6 &weight,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_pose(desired_pose)
  , _desired_pose_inv(inverseSE3(desired_pose))
  , _weight(Mat6::Zero())
{
    assert(model.xml() && "Cannot use FloatingBaseSquaredPoseErrorCost with a "
                          "custom Model");
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
    _desired_pose_inv = inverseSE3(desired_pose);
}

void FloatingBaseSquaredPoseErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_pose.topRightCorner<3, 1>() = desired_translation;
    _desired_pose_inv                    = inverseSE3(_desired_pose);
}

void FloatingBaseSquaredPoseErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_pose.topLeftCorner<3, 3>() = desired_rotation;
    _desired_pose_inv                   = inverseSE3(_desired_pose);
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

void FloatingBaseSquaredPoseErrorCost::evaluate_into(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     VecXRef      out) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Vec6 diff = pinocchio_wrapper::logSE3(_desired_pose_inv * pose);
    out       = 0.5 * diff.transpose() * _weight * diff;
}

void FloatingBaseSquaredPoseErrorCost::add_gradient(ConstVecXRef x,
                                                    ConstVecXRef u, VecXRef dx,
                                                    VecXRef du) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    dx.head<6>() += Jlog6.transpose() * _weight * log6;
}

// Hessians use GN approximation
void FloatingBaseSquaredPoseErrorCost::add_hessian_xx(ConstVecXRef x,
                                                      ConstVecXRef u,
                                                      MatXRef      Hxx) const
{
    Mat4 pose = Mat4::Zero();

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    pose.topLeftCorner<3, 3>()  = q.toRotationMatrix();
    pose.topRightCorner<3, 1>() = x.head<3>();
    pose(3, 3)                  = 1.0;

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    Hxx.topLeftCorner<6, 6>() += Jlog6.transpose() * _weight * Jlog6;
}

// FloatingBaseSquaredVelocityErrorCost
FloatingBaseSquaredVelocityErrorCost::FloatingBaseSquaredVelocityErrorCost(
  const Model &model, const Vec6 &desired_velocity, const double weight,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_velocity(desired_velocity)
  , _reference_frame(reference_frame)
{
    assert(model.xml() &&
           "Cannot use FloatingBaseSquaredVelocityErrorCost with a "
           "custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    _weight.setConstant(weight);
}

FloatingBaseSquaredVelocityErrorCost::FloatingBaseSquaredVelocityErrorCost(
  const Model &model, const Vec6 &desired_velocity, const Vec6 &weight,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _desired_velocity(desired_velocity)
  , _weight(weight)
  , _reference_frame(reference_frame)
{
    assert(model.xml() && "Cannot use FloatingBaseSquaredPoseErrorCost with a "
                          "custom Model");
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

void FloatingBaseSquaredVelocityErrorCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    _reference_frame = reference_frame;
}

void FloatingBaseSquaredVelocityErrorCost::evaluate_into(ConstVecXRef x,
                                                         ConstVecXRef u,
                                                         VecXRef      out) const
{
    assert(x.size() >= _nq + 6 && "Invalid state passed (size < nq + 6)");

    Vec6 velocity = x.segment<6>(_nq);

    const ReferenceFrame::Type type = _reference_frame.type();
    if (type != ReferenceFrame::Local && type != ReferenceFrame::Base)
    {
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        const Mat3         A =
          _reference_frame.rotation().transpose() * q.toRotationMatrix();

        velocity.head<3>() = A * velocity.head<3>();
        velocity.tail<3>() = A * velocity.tail<3>();
    }

    out[0] = 0.5 * (_weight.array() * (velocity - _desired_velocity).array())
                     .matrix()
                     .squaredNorm();
}

void FloatingBaseSquaredVelocityErrorCost::add_gradient(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        VecXRef      dx,
                                                        VecXRef      du) const
{
    assert(x.size() >= _nq + 6 && "Invalid state passed (size < nq + 6)");

    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        dx.segment<6>(_nv) += (_weight.array().square() *
                               (x.segment<6>(_nq) - _desired_velocity).array())
                                .matrix();
    }
    else
    {
        // The base velocity rotated by the base's rotation, which turns with
        // the base's angular tangent
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        const Mat3         A =
          _reference_frame.rotation().transpose() * q.toRotationMatrix();
        const Vec3 linear  = x.segment<3>(_nq);
        const Vec3 angular = x.segment<3>(_nq + 3);

        Vec6 velocity;
        velocity.head<3>() = A * linear;
        velocity.tail<3>() = A * angular;

        MatX J                    = MatX::Zero(6, _ndx);
        J.block<3, 3>(0, _nv)     = A;
        J.block<3, 3>(3, _nv + 3) = A;
        J.block<3, 3>(0, 3)       = -A * skew(linear);
        J.block<3, 3>(3, 3)       = -A * skew(angular);

        dx += J.transpose() * (_weight.array().square() *
                               (velocity - _desired_velocity).array())
                                .matrix();
    }
}

void FloatingBaseSquaredVelocityErrorCost::add_hessian_xx(ConstVecXRef x,
                                                          ConstVecXRef u,
                                                          MatXRef Hxx) const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        Hxx.block<6, 6>(_nv, _nv).diagonal() +=
          _weight.array().square().matrix();
    }
    else
    {
        // Gauss-Newton Hessian through the base's rotation
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        const Mat3         A =
          _reference_frame.rotation().transpose() * q.toRotationMatrix();

        MatX J                    = MatX::Zero(6, _ndx);
        J.block<3, 3>(0, _nv)     = A;
        J.block<3, 3>(3, _nv + 3) = A;
        J.block<3, 3>(0, 3)       = -A * skew(x.segment<3>(_nq));
        J.block<3, 3>(3, 3)       = -A * skew(x.segment<3>(_nq + 3));

        Hxx +=
          J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
    }
}

// FrameSquaredTranslationErrorCost
FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const double weight, const ReferenceFrame &reference_frame,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_translation(desired_translation)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use FrameSquaredTranslationErrorCost with a "
                          "custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
    set_reference_frame(reference_frame);
}

FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const std::string &frame, const Vec3 &desired_translation,
  const Vec3 &weight, const ReferenceFrame &reference_frame,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_translation(desired_translation)
  , _weight(weight)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use FrameSquaredTranslationErrorCost with a "
                          "custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    set_reference_frame(reference_frame);
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

void FrameSquaredTranslationErrorCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FrameSquaredTranslationErrorCost",
                            _model->floatingBase());
    _reference_frame = reference_frame;
}

void FrameSquaredTranslationErrorCost::evaluate_into(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     VecXRef      out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    const Mat3 A     = _reference_frame.type() == ReferenceFrame::Local
                         ? R
                         : _reference_frame.axes(x);
    const Vec3 error = A.transpose() * (pos - _desired_translation);

    out[0] = 0.5 * (_weight.array() * error.array()).matrix().squaredNorm();
}

void FrameSquaredTranslationErrorCost::add_gradient(ConstVecXRef x,
                                                    ConstVecXRef u, VecXRef dx,
                                                    VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    const ReferenceFrame::Type type = _reference_frame.type();
    const Mat3 A = type == ReferenceFrame::Local ? R : _reference_frame.axes(x);
    const Vec3 error = A.transpose() * (pos - _desired_translation);

    // Local or Base axes turn with the frame or the base, which adds the error
    // crossed with their angular tangent
    MatX &J     = _error_jac_scratch;
    J.noalias() = A.transpose() * _jac_scratch.topRows(3);
    if (type == ReferenceFrame::Local)
    {
        J.noalias() += skew(error) * R.transpose() * _jac_scratch.bottomRows(3);
    }
    else if (type == ReferenceFrame::Base)
    {
        J.middleCols<3>(3) += skew(error);
    }

    dx.head(_nv).noalias() +=
      J.transpose() * (_weight.array().square() * error.array()).matrix();
}

void FrameSquaredTranslationErrorCost::add_hessian_xx(ConstVecXRef x,
                                                      ConstVecXRef u,
                                                      MatXRef      Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    const ReferenceFrame::Type type = _reference_frame.type();

    MatX &J = _error_jac_scratch;
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        // Gauss-Newton Hessian through the turning axes
        Vec3 pos;
        Mat3 R;
        std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

        const Mat3 A =
          type == ReferenceFrame::Local ? R : _reference_frame.axes(x);
        const Vec3 error = A.transpose() * (pos - _desired_translation);

        J.noalias() = A.transpose() * _jac_scratch.topRows(3);
        if (type == ReferenceFrame::Local)
        {
            J.noalias() +=
              skew(error) * R.transpose() * _jac_scratch.bottomRows(3);
        }
        else
        {
            J.middleCols<3>(3) += skew(error);
        }
    }
    else
    {
        J.noalias() =
          _reference_frame.rotation().transpose() * _jac_scratch.topRows(3);
    }

    MatX &JtW = _weighted_jac_scratch;
    JtW.noalias() =
      J.transpose() * _weight.array().square().matrix().asDiagonal();
    Hxx.topLeftCorner(_nv, _nv).noalias() += JtW * J;
}

// FrameSquaredTerrainHeightErrorCost
FrameSquaredTerrainHeightErrorCost::FrameSquaredTerrainHeightErrorCost(
  const Model &model, const std::string &frame, const double desired_height,
  const double weight, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_height(desired_height)
  , _weight(weight)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use FrameSquaredTerrainHeightErrorCost with "
                          "a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameSquaredTerrainHeightErrorCost::set_desired_height(
  const double desired_height)
{
    _desired_height = desired_height;
}

void FrameSquaredTerrainHeightErrorCost::set_weight(const double weight)
{
    _weight = weight;
}

void FrameSquaredTerrainHeightErrorCost::set_terrain(
  const std::shared_ptr<terrain::Terrain> &terrain)
{
    _terrain = terrain;
}

std::shared_ptr<terrain::Terrain>
FrameSquaredTerrainHeightErrorCost::cost_terrain() const
{
    return _terrain ? _terrain : _model->terrain();
}

void FrameSquaredTerrainHeightErrorCost::evaluate_into(ConstVecXRef x,
                                                       ConstVecXRef u,
                                                       VecXRef      out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), _frame_id);

    const double height = pos[2] - cost_terrain()->at(pos[0], pos[1]).height;

    out[0] = 0.5 * _weight * _weight * (height - _desired_height) *
             (height - _desired_height);
}

void FrameSquaredTerrainHeightErrorCost::add_gradient(ConstVecXRef x,
                                                      ConstVecXRef u,
                                                      VecXRef      dx,
                                                      VecXRef      du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    terrain::Terrain::TerrainPoint tp = cost_terrain()->at(pos[0], pos[1]);

    dx.head(_nv).noalias() +=
      _weight * _weight * (pos[2] - tp.height - _desired_height) *
      _jac_scratch.topRows(3).transpose() * Vec3(-tp.dhdx, -tp.dhdy, 1);
}

void FrameSquaredTerrainHeightErrorCost::add_hessian_xx(ConstVecXRef x,
                                                        ConstVecXRef u,
                                                        MatXRef      Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Vec3 pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    terrain::Terrain::TerrainPoint tp = cost_terrain()->at(pos[0], pos[1]);

    VecX &dh = _height_grad_scratch;
    dh.noalias() =
      _jac_scratch.topRows(3).transpose() * Vec3(-tp.dhdx, -tp.dhdy, 1);

    Hxx.topLeftCorner(_nv, _nv).noalias() +=
      _weight * _weight * dh * dh.transpose();
}

// FrameSquaredRotationErrorCost
FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const std::string &frame, const Mat3 &desired_rotation,
  const double weight, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_rotation(desired_rotation)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameSquaredRotationErrorCost with a custom "
           "Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
}

FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const std::string &frame, const Mat3 &desired_rotation,
  const Vec3 &weight, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_rotation(desired_rotation)
  , _weight(weight)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameSquaredRotationErrorCost with a custom "
           "Model");
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

void FrameSquaredRotationErrorCost::evaluate_into(ConstVecXRef x,
                                                  ConstVecXRef u,
                                                  VecXRef      out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 R;
    std::tie(std::ignore, R) =
      _model->forwardKinematics(x.head(_nq), _frame_id);

    out[0] =
      0.5 *
      (_weight.array() *
       pinocchio_wrapper::logSO3(R.transpose() * _desired_rotation).array())
        .matrix()
        .squaredNorm();
}

void FrameSquaredRotationErrorCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                                 VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 R;
    std::tie(std::ignore, R) =
      _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);

    Mat3 d        = R.transpose() * _desired_rotation;
    Vec3 phi      = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv  = pinocchio_wrapper::JlogSO3(d);
    MatX jac      = _jac_scratch.bottomRows(3);
    dx.head(_nv) += (-j_r_inv * _weight.array().square().matrix().asDiagonal() *
                     d.transpose() * jac)
                      .transpose() *
                    phi;
}

void FrameSquaredRotationErrorCost::add_hessian_xx(ConstVecXRef x,
                                                   ConstVecXRef u,
                                                   MatXRef      Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat3 rot;
    std::tie(std::ignore, rot) =
      _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);
    Mat3 d       = rot.transpose() * _desired_rotation;
    Vec3 phi     = pinocchio_wrapper::logSO3(d);
    Mat3 j_r_inv = pinocchio_wrapper::JlogSO3(d);
    MatX jac     = _jac_scratch.bottomRows(3);

    MatX J = -j_r_inv * d.transpose() * jac;

    Hxx.topLeftCorner(_nv, _nv) +=
      J.transpose() * _weight.array().square().matrix().asDiagonal() * J;
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
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_pose(desired_pose)
  , _desired_pose_inv(inverseSE3(desired_pose))
  , _weight(Mat6::Zero())
  , _jac_scratch(MatX::Zero(6, _nv))
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
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_pose(desired_pose)
  , _desired_pose_inv(inverseSE3(desired_pose))
  , _weight(Mat6::Zero())
  , _jac_scratch(MatX::Zero(6, _nv))
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
    _desired_pose_inv = inverseSE3(desired_pose);
}

void FrameSquaredPoseErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_pose.block<3, 1>(0, 3) = desired_translation;
    _desired_pose_inv               = inverseSE3(_desired_pose);
}

void FrameSquaredPoseErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_pose.block<3, 3>(0, 0) = desired_rotation;
    _desired_pose_inv               = inverseSE3(_desired_pose);
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

void FrameSquaredPoseErrorCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                              VecXRef out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);

    Vec6 diff = pinocchio_wrapper::logSE3(_desired_pose_inv * pose);
    out       = 0.5 * diff.transpose() * _weight * diff;
}

void FrameSquaredPoseErrorCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                             VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    dx.head(_nv).noalias() +=
      _jac_scratch.transpose() * (Jlog6.transpose() * (_weight * log6));
}

void FrameSquaredPoseErrorCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                               MatXRef Hxx) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);

    Mat4 d     = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(d);

    MatX &J     = _error_jac_scratch;
    J.noalias() = Jlog6 * _jac_scratch;

    MatX &JtW                              = _weighted_jac_scratch;
    JtW.noalias()                          = J.transpose() * _weight;
    Hxx.topLeftCorner(_nv, _nv).noalias() += JtW * J;
}

// FrameSquaredVelocityErrorCost
FrameSquaredVelocityErrorCost::FrameSquaredVelocityErrorCost(
  const Model &model, const std::string &frame, const Vec6 &desired_velocity,
  const double weight, const ReferenceFrame &reference_frame,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_velocity(desired_velocity)
  , _jac_scratch(MatX::Zero(6, _nv))
  , _dVdq_scratch(MatX::Zero(6, _nv))
  , _dVdv_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameSquaredVelocityErrorCost with a custom "
           "Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    _weight.setConstant(weight);
    set_reference_frame(reference_frame);
}

FrameSquaredVelocityErrorCost::FrameSquaredVelocityErrorCost(
  const Model &model, const std::string &frame, const Vec6 &desired_velocity,
  const Vec6 &weight, const ReferenceFrame &reference_frame,
  const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _desired_velocity(desired_velocity)
  , _weight(weight)
  , _jac_scratch(MatX::Zero(6, _nv))
  , _dVdq_scratch(MatX::Zero(6, _nv))
  , _dVdv_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameSquaredPoseErrorCost with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    set_reference_frame(reference_frame);
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

void FrameSquaredVelocityErrorCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FrameSquaredVelocityErrorCost",
                            _model->floatingBase());
    _reference_frame = reference_frame;
}

void FrameSquaredVelocityErrorCost::evaluate_into(ConstVecXRef x,
                                                  ConstVecXRef u,
                                                  VecXRef      out) const
{
    assert(x.size() >= _nq + _nv && "Invalid state passed (size < nq + nv)");

    const bool local = _reference_frame.type() == ReferenceFrame::Local;
    const Mat3 Rt    = _reference_frame.axes(x).transpose();

    Vec6 velocity =
      _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, local);
    velocity.head<3>() = Rt * velocity.head<3>();
    velocity.tail<3>() = Rt * velocity.tail<3>();

    out[0] = 0.5 * (_weight.array() * (velocity - _desired_velocity).array())
                     .matrix()
                     .squaredNorm();
}

void FrameSquaredVelocityErrorCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                                 VecXRef dx, VecXRef du) const
{
    assert(x.size() >= _nq + _nv && "Invalid state passed (size < nq + nv)");

    const Vec6 local_velocity =
      _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, true);
    _model->frameVelocityDerivatives(x.head(_nq),
                                     x.tail(_nv),
                                     _frame_id,
                                     true,
                                     _dVdq_scratch,
                                     _dVdv_scratch);

    Mat3 Rt = Mat3::Identity();
    if (_reference_frame.type() != ReferenceFrame::Local)
    {
        // The local velocity rotated by the frame's rotation, which turns
        // with the frame's angular Jacobian, the angular rows of dV/dv
        Mat3 R;
        std::tie(std::ignore, R) =
          _model->forwardKinematics(x.head(_nq), _frame_id);

        _dVdq_scratch.topRows<3>() +=
          _dVdv_scratch.bottomRows<3>().colwise().cross(
            local_velocity.head<3>());
        _dVdq_scratch.bottomRows<3>() +=
          _dVdv_scratch.bottomRows<3>().colwise().cross(
            local_velocity.tail<3>());
        Rt = _reference_frame.axes(x).transpose() * R;
    }

    Vec6 velocity;
    velocity.head<3>() = Rt * local_velocity.head<3>();
    velocity.tail<3>() = Rt * local_velocity.tail<3>();

    Vec6 diff =
      _weight.array().square() * (velocity - _desired_velocity).array();

    _dVdq_scratch.topRows<3>()    = Rt * _dVdq_scratch.topRows<3>();
    _dVdq_scratch.bottomRows<3>() = Rt * _dVdq_scratch.bottomRows<3>();
    _dVdv_scratch.topRows<3>()    = Rt * _dVdv_scratch.topRows<3>();
    _dVdv_scratch.bottomRows<3>() = Rt * _dVdv_scratch.bottomRows<3>();

    // Base axes turn with the base, which adds the velocity crossed with the
    // base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        _dVdq_scratch.block<3, 3>(0, 3) += skew(velocity.head<3>());
        _dVdq_scratch.block<3, 3>(3, 3) += skew(velocity.tail<3>());
    }

    dx.head(_nv) += _dVdq_scratch.transpose() * diff;
    dx.tail(_nv) += _dVdv_scratch.transpose() * diff;
}

void FrameSquaredVelocityErrorCost::add_hessian_xx(ConstVecXRef x,
                                                   ConstVecXRef u,
                                                   MatXRef      Hxx) const
{
    const bool local = _reference_frame.type() == ReferenceFrame::Local;
    const Mat3 Rt    = _reference_frame.axes(x).transpose();

    _model->frameJacobian(x.head(_nq), _frame_id, local, _jac_scratch);
    _jac_scratch.topRows<3>()    = Rt * _jac_scratch.topRows<3>();
    _jac_scratch.bottomRows<3>() = Rt * _jac_scratch.bottomRows<3>();

    Hxx.bottomRightCorner(_nv, _nv) +=
      _jac_scratch.transpose() *
      _weight.array().square().matrix().asDiagonal() * _jac_scratch;
}

// CenterOfMassTrackingCost
CenterOfMassTrackingCost::CenterOfMassTrackingCost(
  const Model &model, const Vec3 &cref, const double weight,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _cref(cref)
{
    assert(model.xml() &&
           "Cannot use CenterOfMassTrackingCost with a custom Model");
    _weight.setConstant(weight);
    set_reference_frame(reference_frame);
}

CenterOfMassTrackingCost::CenterOfMassTrackingCost(
  const Model &model, const Vec3 &cref, const Vec3 &weight,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _cref(cref)
  , _weight(weight)
{
    assert(model.xml() &&
           "Cannot use CenterOfMassTrackingCost with a custom Model");
    set_reference_frame(reference_frame);
}

void CenterOfMassTrackingCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void CenterOfMassTrackingCost::set_weight(const Vec3 &weight)
{
    _weight = weight;
}

void CenterOfMassTrackingCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require(
      { ReferenceFrame::World, ReferenceFrame::Fixed, ReferenceFrame::Base },
      "CenterOfMassTrackingCost",
      _model->floatingBase());
    _reference_frame = reference_frame;
}

void CenterOfMassTrackingCost::set_cref(const Vec3 &cref)
{
    _cref = cref;
}

void CenterOfMassTrackingCost::evaluate_into(ConstVecXRef x, ConstVecXRef u,
                                             VecXRef out) const
{
    Vec3 CoM = _model->centerOfMass(x.head(_nq));

    const Vec3 error = _reference_frame.axes(x).transpose() * (CoM - _cref);

    out[0] = 0.5 * (_weight.array() * error.array()).matrix().squaredNorm();
}

void CenterOfMassTrackingCost::add_gradient(ConstVecXRef x, ConstVecXRef u,
                                            VecXRef dx, VecXRef du) const
{
    ConstVecXRef q = x.head(_nq);

    const Mat3 Rt = _reference_frame.axes(x).transpose();

    Vec3 CoM  = _model->centerOfMass(q);
    MatX JCoM = Rt * _model->jacobianCenterOfMass(q);

    const Vec3 error = Rt * (CoM - _cref);

    // Base axes turn with the base, which adds the error crossed with the
    // base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        JCoM.middleCols<3>(3) += skew(error);
    }

    dx.head(_nv) +=
      JCoM.transpose() * (_weight.array().square() * error.array()).matrix();
}

void CenterOfMassTrackingCost::add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                              MatXRef Hxx) const
{
    const Mat3 Rt = _reference_frame.axes(x).transpose();

    MatX JCoM = Rt * _model->jacobianCenterOfMass(x.head(_nq));

    // Gauss-Newton Hessian through the turning base axes
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        JCoM.middleCols<3>(3) +=
          skew(Rt * (_model->centerOfMass(x.head(_nq)) - _cref));
    }

    Hxx.topLeftCorner(_nv, _nv) +=
      JCoM.transpose() * _weight.array().square().matrix().asDiagonal() * JCoM;
}

// FramesInverseSquaredDistanceCost
std::vector<uint> FramesInverseSquaredDistanceCost::_resolveFrameIds(
  const Model &model, const std::vector<std::string> &frames)
{
    std::vector<uint> ids;
    ids.reserve(frames.size());

    for (const std::string &frame : frames)
    {
        ids.push_back(model.frameId(frame));
    }

    return ids;
}

FramesInverseSquaredDistanceCost::FramesInverseSquaredDistanceCost(
  const Model &model, const std::vector<std::string> &frames,
  const double weight, const double distance_scale,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frames(frames)
  , _frame_ids(_resolveFrameIds(model, frames))
  , _weight(weight)
{
    assert(model.xml() && "Cannot use FramesInverseSquaredDistanceCost with a "
                          "custom Model");
#ifndef NDEBUG
    for (uint i = 0; i < frames.size(); ++i)
    {
        assert(model.hasFrame(frames[i]) && "Invalid frame for given model");
    }
#endif
    set_distance_scale(distance_scale);
    set_reference_frame(reference_frame);
    _jac_scratch.resize(6, _nv);
    _jac_cache.resize(_frames.size());
    for (MatX &jac : _jac_cache)
    {
        jac.resize(3, _nv);
    }
}

FramesInverseSquaredDistanceCost::FramesInverseSquaredDistanceCost(
  const Model &model, const std::vector<std::string> &frames,
  const double weight, const Vec3 &distance_scale,
  const ReferenceFrame &reference_frame, const std::string &name)
  : PredefinedStageCost(model, name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _frames(frames)
  , _frame_ids(_resolveFrameIds(model, frames))
  , _weight(weight)
{
    assert(model.xml() && "Cannot use FramesInverseSquaredDistanceCost with a "
                          "custom Model");
#ifndef NDEBUG
    for (uint i = 0; i < frames.size(); ++i)
    {
        assert(model.hasFrame(frames[i]) && "Invalid frame for given model");
    }
#endif
    set_distance_scale(distance_scale);
    set_reference_frame(reference_frame);
    _jac_scratch.resize(6, _nv);
    _jac_cache.resize(_frames.size());
    for (MatX &jac : _jac_cache)
    {
        jac.resize(3, _nv);
    }
}

void FramesInverseSquaredDistanceCost::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require(
      { ReferenceFrame::World, ReferenceFrame::Fixed, ReferenceFrame::Base },
      "FramesInverseSquaredDistanceCost",
      _model->floatingBase());
    _reference_frame = reference_frame;
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

void FramesInverseSquaredDistanceCost::evaluate_into(ConstVecXRef x,
                                                     ConstVecXRef u,
                                                     VecXRef      out) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    const Mat3 Rt = _reference_frame.axes(x).transpose();

    std::vector<Vec3> translations(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frame_ids[i]);
        translations[i] = Rt * translations[i];
    }

    out[0] = 0.0;

    for (uint i = 0; i < _frames.size(); ++i)
    {
        for (uint j = i + 1; j < _frames.size(); ++j)
        {
            out[0] +=
              _weight / (2 * (_distance_scale.array() *
                              (translations[i] - translations[j]).array())
                               .matrix()
                               .squaredNorm());
        }
    }
}

void FramesInverseSquaredDistanceCost::add_gradient(ConstVecXRef x,
                                                    ConstVecXRef u, VecXRef dx,
                                                    VecXRef du) const
{
    assert(x.size() >= _nq && "Invalid state passed (size < nq)");

    const Mat3 Rt = _reference_frame.axes(x).transpose();

    std::vector<Vec3> translations(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frame_ids[i]);
        translations[i] = Rt * translations[i];
        _model->frameJacobian(x.head(_nq), _frame_ids[i], false, _jac_scratch);
        _jac_cache[i] = Rt * _jac_scratch.topRows<3>();
        // Base axes turn with the base, which adds the translation crossed
        // with the base's angular tangent
        if (_reference_frame.type() == ReferenceFrame::Base)
        {
            _jac_cache[i].middleCols<3>(3) += skew(translations[i]);
        }
    }

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
              _weight * (_jac_cache[i] - _jac_cache[j]).transpose() * g_d;
        }
    }
}

void FramesInverseSquaredDistanceCost::add_hessian_xx(ConstVecXRef x,
                                                      ConstVecXRef u,
                                                      MatXRef      Hxx) const
{
    const Mat3 Rt = _reference_frame.axes(x).transpose();

    std::vector<Vec3> translations(_frames.size());

    for (uint i = 0; i < _frames.size(); ++i)
    {
        std::tie(translations[i], std::ignore) =
          _model->forwardKinematics(x.head(_nq), _frame_ids[i]);
        translations[i] = Rt * translations[i];
        _model->frameJacobian(x.head(_nq), _frame_ids[i], false, _jac_scratch);
        _jac_cache[i] = Rt * _jac_scratch.topRows<3>();
        // Base axes turn with the base, which adds the translation crossed
        // with the base's angular tangent
        if (_reference_frame.type() == ReferenceFrame::Base)
        {
            _jac_cache[i].middleCols<3>(3) += skew(translations[i]);
        }
    }

    Vec3 d2 = _distance_scale.array().square();

    for (uint i = 0; i < _frames.size(); ++i)
    {
        for (uint j = i + 1; j < _frames.size(); ++j)
        {
            Vec3 d  = translations[i] - translations[j];
            MatX Jd = _jac_cache[i] - _jac_cache[j];

            double s =
              (_distance_scale.array() * d.array()).matrix().squaredNorm();

            Vec3 Ad = d2.array() * d.array();

            Mat3 Hd  = (-d2 / (s * s)).asDiagonal();
            Hd      += 4.0 * (Ad * Ad.transpose()) / (s * s * s);

            Hxx.topLeftCorner(_nv, _nv) += _weight * Jd.transpose() * Hd * Jd;
        }
    }
}
}
}
}
}
}