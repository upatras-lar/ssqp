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
// SquaredControlCost
SquaredControlCost::SquaredControlCost(const uint nv, const double weight)
  : _nv(nv)
  , _weight(weight)
{
}

VecX SquaredControlCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar() << _weight * u.squaredNorm()).finished();
}

void SquaredControlCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                  VecX &du) const
{
    dx = VecX::Zero(2 * _nv);
    du = 2 * _weight * u;
}

void SquaredControlCost::hessian_xx(const VecX &x, const VecX &u,
                                    MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
}

void SquaredControlCost::hessian_xu(const VecX &x, const VecX &u,
                                    MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, _nv);
}

void SquaredControlCost::hessian_uu(const VecX &x, const VecX &u,
                                    MatX &Huu) const
{
    Huu = MatX::Zero(_nv, _nv);
    Huu.diagonal().setConstant(2 * _weight);
}

// SquaredVelocityCost
SquaredVelocityCost::SquaredVelocityCost(const uint nv, const double weight)
  : _nv(nv)
  , _weight(weight)
{
}

VecX SquaredVelocityCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar() << _weight * x.tail(_nv).squaredNorm()).finished();
}

void SquaredVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                   VecX &du) const
{
    dx           = VecX::Zero(2 * _nv);
    dx.tail(_nv) = 2 * _weight * x.tail(_nv);

    du = VecX::Zero(_nv);
}

void SquaredVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                     MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_nv, _nv).diagonal().setConstant(2 * _weight);
}

void SquaredVelocityCost::hessian_xu(const VecX &x, const VecX &u,
                                     MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, _nv);
}

void SquaredVelocityCost::hessian_uu(const VecX &x, const VecX &u,
                                     MatX &Huu) const
{
    Huu = MatX::Zero(_nv, _nv);
}

// FrameSquaredPositionErrorCost
FrameSquaredPositionErrorCost::FrameSquaredPositionErrorCost(
  const Model &model, const VecX &desired_translation, const std::string &frame,
  const uint nq, const uint nv, const double weight)
  : _model(std::make_shared<Model>(model))
  , _desired_translation(desired_translation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
{
}

VecX FrameSquaredPositionErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    VecX pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);

    Scalar res;

    return (Scalar() << _weight * (pos - _desired_translation).squaredNorm())
      .finished();
}

void FrameSquaredPositionErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    VecX pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk                   = _model->frameJacobian(x.head(_nq), _frame);

    dx = VecX::Zero(2 * _nv);

    dx.head(_nv) =
      2 * _weight * Jfk.topRows(3).transpose() * (pos - _desired_translation);

    du = VecX::Zero(_nv);
}
void FrameSquaredPositionErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) =
      2 * _weight * Jfk.topRows(3).transpose() * Jfk.topRows(3);
}

void FrameSquaredPositionErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, _nv);
}

void FrameSquaredPositionErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(_nv, _nv);
}

// FrameSquaredRotationErrorCost
FrameSquaredRotationErrorCost::FrameSquaredRotationErrorCost(
  const Model &model, const MatX &desired_rotation, const std::string &frame,
  const uint nq, const uint nv, const double weight)
  : _model(std::make_shared<Model>(model))
  , _desired_rotation(desired_rotation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(weight)
{
}

VecX FrameSquaredRotationErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    MatX rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);

    return (Scalar() << _weight * pinocchio_wrapper::logSO3(
                                    _desired_rotation.transpose() * rot)
                                    .squaredNorm())
      .finished();
}

void FrameSquaredRotationErrorCost::gradient(const VecX &x, const VecX &u,
                                             VecX &dx, VecX &du) const
{
    MatX rot;
    std::tie(std::ignore, rot) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk                   = _model->frameJacobian(x.head(_nq), _frame);

    dx = VecX::Zero(2 * _nv);

    MatX rp      = _desired_rotation.transpose() * rot;
    dx.head(_nv) = 2 * _weight * Jfk.bottomRows(3).transpose() *
                   _desired_rotation * pinocchio_wrapper::JlogSO3(rp) *
                   pinocchio_wrapper::logSO3(rp);

    du = VecX::Zero(_nv);
}

void FrameSquaredRotationErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) =
      2 * _weight * Jfk.bottomRows(3).transpose() * Jfk.bottomRows(3);
}

void FrameSquaredRotationErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, _nv);
}

void FrameSquaredRotationErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                               MatX &Huu) const
{
    Huu = MatX::Zero(_nv, _nv);
}

// FrameSquaredPoseErrorCost
FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const VecX &desired_translation,
  const MatX &desired_rotation, const std::string &frame, const uint nq,
  const uint nv, const double position_weight, const double rotation_weight)
  : _model(std::make_shared<Model>(model))
  , _desired_pose_inv(Mat4::Zero())
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(Mat6::Zero())
{
    _desired_pose_inv.topLeftCorner<3, 3>() = desired_rotation.transpose();
    _desired_pose_inv.topRightCorner<3, 1>() =
      -desired_rotation.transpose() * desired_translation;
    _desired_pose_inv(3, 3) = 1.0;

    _weight.topLeftCorner<3, 3>()     = position_weight * Mat3::Identity();
    _weight.bottomRightCorner<3, 3>() = rotation_weight * Mat3::Identity();
}

VecX FrameSquaredPoseErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);

    return (Scalar() << (_weight *
                         pinocchio_wrapper::logSE3(_desired_pose_inv * pose))
                          .squaredNorm())
      .finished();
}

// TODO: Write this correctly
void FrameSquaredPoseErrorCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                         VecX &du) const
{
    Mat4 pose = _model->forwardKinematicsSE3(x.head(_nq), _frame);
    MatX Jfk  = _model->frameJacobian(x.head(_nq), _frame);

    Mat4 pd    = _desired_pose_inv * pose;
    Vec6 log6  = pinocchio_wrapper::logSE3(pd);
    Mat6 Jlog6 = pinocchio_wrapper::JlogSE3(pd);

    dx           = VecX::Zero(2 * _nv);
    dx.head(_nv) = 2 * Jfk.transpose() *
                   pinocchio_wrapper::SE3ToActionMatrixInverse(pd).transpose() *
                   Jlog6.transpose() * _weight * log6;

    du = VecX::Zero(_nv);
}

void FrameSquaredPoseErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                           MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx                         = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) = 2 * Jfk.transpose() * _weight * Jfk;
}

void FrameSquaredPoseErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                           MatX &Hxu) const
{
    ZeroHessian_xu(x, u, Hxu);
}

void FrameSquaredPoseErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                           MatX &Huu) const
{
    ZeroHessian_uu(x, u, Huu);
}

}
}
}
}
}