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
SquaredJointAngleCost::SquaredJointAngleCost(const uint nq, const uint nv,
                                             const double weight,
                                             const bool   free_flyer)
  : _nq(nq)
  , _nv(nv)
  , _weight(free_flyer ? nq - 7 : nq)
  , _free_flyer(free_flyer)
{
    _weight.setConstant(weight);
}

SquaredJointAngleCost::SquaredJointAngleCost(const uint nq, const uint nv,
                                             const VecX &weight,
                                             const bool  free_flyer)
  : _nq(nq)
  , _nv(nv)
  , _weight(weight)
  , _free_flyer(free_flyer)
{
    assert(_weight.size() == (free_flyer ? nq - 7 : nq) &&
           "Invalid weight vector size");
}

void SquaredJointAngleCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredJointAngleCost::set_weight(const VecX &weight)
{
    assert(weight.size() == (_free_flyer ? _nq - 7 : _nq) &&
           "Invalid weight vector size");

    _weight = weight;
}

VecX SquaredJointAngleCost::evaluate(const VecX &x, const VecX &u) const
{
    VecX q = _free_flyer ? x.segment(7, _nq - 7) : x.head(_nq);
    return (Scalar() << 0.5 *
                          (_weight.array() * q.array()).matrix().squaredNorm())
      .finished();
}

void SquaredJointAngleCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                     VecX &du) const
{
    dx = VecX::Zero(2 * _nv);
    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) =
          _weight.array() * x.segment(7, _nq - 7).array();
    }
    else
    {
        dx.head(_nv) = _weight.array() * x.head(_nv).array();
    }

    du = VecX::Zero(u.size());
}
void SquaredJointAngleCost::hessian_xx(const VecX &x, const VecX &u,
                                       MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);

    if (_free_flyer)
    {
        Hxx.block(7, 7, _nq - 7, _nq - 7).diagonal() = _weight;
    }
    else
    {
        Hxx.topLeftCorner(_nv, _nv).diagonal() = _weight;
    }
}

void SquaredJointAngleCost::hessian_xu(const VecX &x, const VecX &u,
                                       MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void SquaredJointAngleCost::hessian_uu(const VecX &x, const VecX &u,
                                       MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// SquaredJointVelocityCost
SquaredJointVelocityCost::SquaredJointVelocityCost(const uint   nv,
                                                   const double weight,
                                                   const bool   free_flyer)
  : _nv(nv)
  , _weight(free_flyer ? nv - 6 : nv)
  , _njoints(free_flyer ? nv - 6 : nv)
{
    _weight.setConstant(weight);
}

SquaredJointVelocityCost::SquaredJointVelocityCost(const uint  nv,
                                                   const VecX &weight,
                                                   const bool  free_flyer)
  : _nv(nv)
  , _weight(weight)
  , _njoints(free_flyer ? nv - 6 : nv)
{
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
    return (Scalar() << 0.5 * (_weight.array() * x.tail(_njoints).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void SquaredJointVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    dx                = VecX::Zero(2 * _nv);
    dx.tail(_njoints) = _weight.array() * x.tail(_njoints).array();

    du = VecX::Zero(u.size());
}

void SquaredJointVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_njoints, _njoints).diagonal() = _weight;
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
SquaredControlCost::SquaredControlCost(const uint nv, const double weight)
  : _nv(nv)
  , _weight(nv)
{
    _weight.setConstant(weight);
}

SquaredControlCost::SquaredControlCost(const uint nv, const VecX &weight)
  : _nv(nv)
  , _weight(weight)
{
    assert(weight.size() == nv && "Invalid weight vector size");
}

void SquaredControlCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredControlCost::set_weight(const VecX &weight)
{
    assert(weight.size() == _nv && "Invalid weight vector size");

    _weight = weight;
}

VecX SquaredControlCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar()
            << 0.5 *
                 (_weight.array() * u.head(_nv).array()).matrix().squaredNorm())
      .finished();
}

void SquaredControlCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                  VecX &du) const
{
    dx           = VecX::Zero(2 * _nv);
    du           = VecX::Zero(u.size());
    du.head(_nv) = _weight.array() * u.head(_nv).array();
}

void SquaredControlCost::hessian_xx(const VecX &x, const VecX &u,
                                    MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
}

void SquaredControlCost::hessian_xu(const VecX &x, const VecX &u,
                                    MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void SquaredControlCost::hessian_uu(const VecX &x, const VecX &u,
                                    MatX &Huu) const
{
    Huu                                    = MatX::Zero(u.size(), u.size());
    Huu.topLeftCorner(_nv, _nv).diagonal() = _weight;
}

// SquaredForceCost
SquaredForceCost::SquaredForceCost(const uint nv, const double weight)
  : _nv(nv)
  , _weight(weight)
  , _single_weight(true)
{
}

SquaredForceCost::SquaredForceCost(const uint nv, const VecX &weight)
  : _nv(nv)
  , _weight_vec(weight)
  , _single_weight(false)
{
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
    Scalar res;

    if (_single_weight)
    {
        res[0] = (_weight * u.tail(u.size() - _nv)).squaredNorm();
    }
    else
    {
        res[0] = (_weight_vec.array() * u.tail(u.size() - _nv).array())
                   .matrix()
                   .squaredNorm();
    }

    return res;
}

void SquaredForceCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                VecX &du) const
{
    dx = VecX::Zero(2 * _nv);
    du = VecX::Zero(u.size());

    if (_single_weight)
    {
        du.tail(u.size() - _nv) = _weight * u.tail(u.size() - _nv);
    }
    else
    {
        du.tail(u.size() - _nv) =
          _weight_vec.array() * u.tail(u.size() - _nv).array();
    }
}

void SquaredForceCost::hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
}

void SquaredForceCost::hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void SquaredForceCost::hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
    if (_single_weight)
    {
        Huu.bottomRightCorner(u.size() - _nv, u.size() - _nv)
          .diagonal()
          .setConstant(_weight);
    }
    else
    {
        Huu.bottomRightCorner(u.size() - _nv, u.size() - _nv).diagonal() =
          _weight_vec;
    }
}

// SquaredConfigurationDifferenceCost
SquaredConfigurationDifferenceCost::SquaredConfigurationDifferenceCost(
  const VecX &desired_configuration, const uint nq, const uint nv,
  const double weight, const bool free_flyer)
  : _desired_configuration(desired_configuration)
  , _nq(nq)
  , _nv(nv)
  , _weight(free_flyer ? nq - 7 : nq)
  , _free_flyer(free_flyer)
{
    _weight.setConstant(weight);
}

SquaredConfigurationDifferenceCost::SquaredConfigurationDifferenceCost(
  const VecX &desired_configuration, const uint nq, const uint nv,
  const VecX &weight, const bool free_flyer)
  : _desired_configuration(desired_configuration)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
  , _free_flyer(free_flyer)
{
    assert(weight.size() == (free_flyer ? nq - 7 : nq) &&
           "Invalid weight vector size");
}

void SquaredConfigurationDifferenceCost::set_desired_configuration(
  const VecX &desired_configuration)
{
    _desired_configuration = desired_configuration;
}

void SquaredConfigurationDifferenceCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void SquaredConfigurationDifferenceCost::set_weight(const VecX &weight)
{
    assert(weight.size() == (_free_flyer ? _nq - 7 : _nq) &&
           "Invalid weight vector size");

    _weight = weight;
}

VecX SquaredConfigurationDifferenceCost::evaluate(const VecX &x,
                                                  const VecX &u) const
{
    return (Scalar() << 0.5 * (_weight.array() *
                               ((_free_flyer ? x.segment(7, _nq - 7)
                                             : x.head(_nq)) -
                                _desired_configuration)
                                 .array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void SquaredConfigurationDifferenceCost::gradient(const VecX &x, const VecX &u,
                                                  VecX &dx, VecX &du) const
{
    dx = VecX::Zero(2 * _nv);
    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) =
          _weight.array() *
          (x.segment(7, _nq - 7) - _desired_configuration).array();
    }
    else
    {
        dx.head(_nv) =
          _weight.array() * (x.head(_nv) - _desired_configuration).array();
    }

    du = VecX::Zero(u.size());
}

void SquaredConfigurationDifferenceCost::hessian_xx(const VecX &x,
                                                    const VecX &u,
                                                    MatX       &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);

    if (_free_flyer)
    {
        Hxx.block(7, 7, _nq - 7, _nq - 7).diagonal() = _weight;
    }
    else
    {
        Hxx.topLeftCorner(_nv, _nv).diagonal() = _weight;
    }
}

void SquaredConfigurationDifferenceCost::hessian_xu(const VecX &x,
                                                    const VecX &u,
                                                    MatX       &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void SquaredConfigurationDifferenceCost::hessian_uu(const VecX &x,
                                                    const VecX &u,
                                                    MatX       &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredTranslationErrorCost
FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const Vec3 &desired_translation, const std::string &frame,
  const uint nq, const uint nv, const double weight)
  : _model(std::make_shared<Model>(model))
  , _desired_translation(desired_translation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
{
    _weight.setConstant(weight);
}

FrameSquaredTranslationErrorCost::FrameSquaredTranslationErrorCost(
  const Model &model, const Vec3 &desired_translation, const std::string &frame,
  const uint nq, const uint nv, const Vec3 &weight)
  : _model(std::make_shared<Model>(model))
  , _desired_translation(desired_translation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
{
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
    VecX pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);

    Scalar res;

    return (Scalar() << 0.5 * (_weight.array() *
                               (pos - _desired_translation).array())
                                .matrix()
                                .squaredNorm())
      .finished();
}

void FrameSquaredTranslationErrorCost::gradient(const VecX &x, const VecX &u,
                                                VecX &dx, VecX &du) const
{
    VecX pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk                   = _model->frameJacobian(x.head(_nq), _frame);

    dx = VecX::Zero(2 * _nv);

    dx.head(_nv) =
      Jfk.topRows(3).transpose() *
      (_weight.array() * (pos - _desired_translation).array()).matrix();

    du = VecX::Zero(u.size());
}
void FrameSquaredTranslationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                                  MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) =
      Jfk.topRows(3).transpose() * _weight.asDiagonal() * Jfk.topRows(3);
}

void FrameSquaredTranslationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                                  MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void FrameSquaredTranslationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                                  MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredRotationErrorCost
FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const Mat3 &desired_rotation, const std::string &frame,
  const uint nq, const uint nv, const double weight)
  : _model(std::make_shared<Model>(model))
  , _desired_rotation(desired_rotation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
{
}

void FrameSquaredRotationErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_rotation = desired_rotation;
}

void FrameSquaredRotationErrorCost::set_weight(const double weight)
{
    _weight = weight;
}

VecX FrameSquaredRotationErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    MatX rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);

    return (Scalar() << 0.5 * _weight *
                          pinocchio_wrapper::logSO3(rot.transpose() *
                                                    _desired_rotation)
                            .squaredNorm())
      .finished();
}

void FrameSquaredRotationErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    MatX rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame, true);

    dx = VecX::Zero(2 * _nv);

    MatX d       = rot.transpose() * _desired_rotation;
    VecX phi     = pinocchio_wrapper::logSO3(d);
    MatX j_r_inv = pinocchio_wrapper::JlogSO3(d);
    MatX jac     = Jfk.bottomRows(3);
    dx.head(_nv) = _weight * (-j_r_inv * d.transpose() * jac).transpose() * phi;

    du = VecX::Zero(u.size());
}

void FrameSquaredRotationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    MatX rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk     = _model->frameJacobian(x.head(_nq), _frame, true);
    MatX d       = rot.transpose() * _desired_rotation;
    VecX phi     = pinocchio_wrapper::logSO3(d);
    MatX j_r_inv = pinocchio_wrapper::JlogSO3(d);
    MatX jac     = Jfk.bottomRows(3);

    MatX J = -j_r_inv * d.transpose() * jac;

    Hxx                         = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) = _weight * J.transpose() * J;
}

void FrameSquaredRotationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void FrameSquaredRotationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredPoseErrorCost
FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const Mat4 &desired_pose, const std::string &frame,
  const uint nq, const uint nv, const double translation_weight,
  const double rotation_weight)
  : _model(std::make_shared<Model>(model))
  , _desired_pose(desired_pose)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(Mat6::Zero())
{
    _weight.topLeftCorner<3, 3>()     = translation_weight * Mat3::Identity();
    _weight.bottomRightCorner<3, 3>() = rotation_weight * Mat3::Identity();
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const Vec3 &desired_translation,
  const Mat3 &desired_rotation, const std::string &frame, const uint nq,
  const uint nv, const double translation_weight, const double rotation_weight)
  : FrameSquaredPoseErrorCost(model,
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
                              frame, nq, nv, translation_weight,
                              rotation_weight)
{
}

void FrameSquaredPoseErrorCost::set_desired_pose(const Mat4 &desired_pose)
{
    _desired_pose = desired_pose;
}

void FrameSquaredPoseErrorCost::set_desired_translation(
  const Vec3 &desired_translation)
{
    _desired_pose.block<3, 1>(0, 3) = desired_translation.transpose();
}

void FrameSquaredPoseErrorCost::set_desired_rotation(
  const Mat3 &desired_rotation)
{
    _desired_pose.block<3, 3>(0, 0) = desired_rotation;
}

void FrameSquaredPoseErrorCost::set_translation_weight(
  const double translation_weight)
{
    _weight.topLeftCorner<3, 3>() = translation_weight * Mat3::Identity();
}

void FrameSquaredPoseErrorCost::set_rotation_weight(
  const double rotation_weight)
{
    _weight.bottomRightCorner<3, 3>() = rotation_weight * Mat3::Identity();
}

VecX FrameSquaredPoseErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);

    Mat4 pose_inv                  = Mat4::Zero();
    pose_inv.topLeftCorner<3, 3>() = pose.topLeftCorner<3, 3>().transpose();
    pose_inv.topRightCorner<3, 1>() =
      -pose.topLeftCorner<3, 3>().transpose() * pose.topRightCorner<3, 1>();
    pose_inv(3, 3) = 1.0;

    Vec6 diff = pinocchio_wrapper::logSE3(pose_inv * _desired_pose);
    return (Scalar() << 0.5 * diff.transpose() * _weight * diff).finished();
}

void FrameSquaredPoseErrorCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                         VecX &du) const
{
    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);
    MatX Jfk  = _model->frameJacobian(x.head(_nq), _frame, true);

    Mat4 pose_inv                  = Mat4::Zero();
    pose_inv.topLeftCorner<3, 3>() = pose.topLeftCorner<3, 3>().transpose();
    pose_inv.topRightCorner<3, 1>() =
      -pose.topLeftCorner<3, 3>().transpose() * pose.topRightCorner<3, 1>();
    pose_inv(3, 3) = 1.0;

    Mat4 d       = pose_inv * _desired_pose;
    Vec6 log6    = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6   = pinocchio_wrapper::JlogSE3(d);
    Mat6 adj_inv = pinocchio_wrapper::SE3ToActionMatrixInverse(d);

    dx           = VecX::Zero(2 * _nv);
    dx.head(_nv) = (-Jlog6 * adj_inv * Jfk).transpose() * _weight * log6;

    du = VecX::Zero(u.size());
}

void FrameSquaredPoseErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                           MatX &Hxx) const
{
    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);
    MatX Jfk  = _model->frameJacobian(x.head(_nq), _frame, true);

    Mat4 pose_inv                  = Mat4::Zero();
    pose_inv.topLeftCorner<3, 3>() = pose.topLeftCorner<3, 3>().transpose();
    pose_inv.topRightCorner<3, 1>() =
      -pose.topLeftCorner<3, 3>().transpose() * pose.topRightCorner<3, 1>();
    pose_inv(3, 3) = 1.0;

    Mat4 d       = pose_inv * _desired_pose;
    Vec6 log6    = pinocchio_wrapper::logSE3(d);
    Mat6 Jlog6   = pinocchio_wrapper::JlogSE3(d);
    Mat6 adj_inv = pinocchio_wrapper::SE3ToActionMatrixInverse(d);

    MatX J = -Jlog6 * adj_inv * Jfk;

    Hxx                         = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) = J.transpose() * _weight * J;
}

void FrameSquaredPoseErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                           MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void FrameSquaredPoseErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                           MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}

// FrameSquaredVelocityCost
FrameSquaredVelocityCost::FrameSquaredVelocityCost(const Model       &model,
                                                   const std::string &frame,
                                                   const uint nq, const uint nv,
                                                   const double weight)
  : _model(std::make_shared<Model>(model))
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
{
    _weight.setConstant(weight);
}

FrameSquaredVelocityCost::FrameSquaredVelocityCost(const Model       &model,
                                                   const std::string &frame,
                                                   const uint nq, const uint nv,
                                                   const Vec6 &weight)
  : _model(std::make_shared<Model>(model))
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
{
}

void FrameSquaredVelocityCost::set_weight(const double weight)
{
    _weight.setConstant(weight);
}

void FrameSquaredVelocityCost::set_weight(const Vec6 &weight)
{
    _weight = weight;
}

VecX FrameSquaredVelocityCost::evaluate(const VecX &x, const VecX &u) const
{
    LOGLN("TF?");
    return (Scalar()
            << 0.5 *
                 (_weight.array() *
                  _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true)
                    .array())
                   .matrix()
                   .squaredNorm())
      .finished();
}

void FrameSquaredVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    Vec6 vel =
      _weight.array() *
      _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true).array();

    MatX dVdq, dVdv;
    std::tie(dVdq, dVdv) = _model->frameVelocityDerivatives(x.head(_nq),
                                                            x.tail(_nv),
                                                            u.head(_nv),
                                                            _frame,
                                                            true);

    dx = VecX::Zero(2 * _nv);

    dx.head(_nv) = _weight * dVdq.transpose() * vel;
    dx.tail(_nv) = _weight * dVdv.transpose() * vel;

    du = VecX::Zero(u.size());
}
void FrameSquaredVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_nv, _nv) =
      Jfk.transpose() * _weight.asDiagonal() * Jfk;
}
void FrameSquaredVelocityCost::hessian_xu(const VecX &x, const VecX &u,
                                          MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}
void FrameSquaredVelocityCost::hessian_uu(const VecX &x, const VecX &u,
                                          MatX &Huu) const
{
    Huu = MatX::Zero(u.size(), u.size());
}
}
}
}
}
}