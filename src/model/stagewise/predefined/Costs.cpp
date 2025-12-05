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
                                             bool         free_flyer)
  : _nq(nq)
  , _nv(nv)
  , _weight(weight)
  , _free_flyer(free_flyer)
{
}

VecX SquaredJointAngleCost::evaluate(const VecX &x, const VecX &u) const
{
    VecX q = _free_flyer ? x.segment(7, _nq - 7) : x.head(_nq);
    return (Scalar() << 0.5 * _weight * q.squaredNorm()).finished();
}

void SquaredJointAngleCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                     VecX &du) const
{
    dx = VecX::Zero(2 * _nv);
    if (_free_flyer)
    {
        dx.segment(6, _nv - 6) = _weight * x.segment(7, _nq - 7);
    }
    else
    {
        dx.head(_nq) = _weight * x.head(_nq);
    }

    du = VecX::Zero(u.size());
}
void SquaredJointAngleCost::hessian_xx(const VecX &x, const VecX &u,
                                       MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);

    if (_free_flyer)
    {
        Hxx.block(7, 7, _nq - 7, _nq - 7).diagonal().setConstant(_weight);
    }
    else
    {
        Hxx.topLeftCorner(_nq, _nq).diagonal().setConstant(_weight);
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
                                                   bool         free_flyer)
  : _nv(nv)
  , _weight(weight)
  , _njoints(free_flyer ? nv - 6 : nv)
{
}

VecX SquaredJointVelocityCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar() << 0.5 * _weight * x.tail(_njoints).squaredNorm())
      .finished();
}

void SquaredJointVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    dx                = VecX::Zero(2 * _nv);
    dx.tail(_njoints) = _weight * x.tail(_njoints);

    du = VecX::Zero(u.size());
}

void SquaredJointVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_njoints, _njoints).diagonal().setConstant(_weight);
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
  , _weight(weight)
{
}

VecX SquaredControlCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar() << 0.5 * _weight * u.head(_nv).squaredNorm()).finished();
}

void SquaredControlCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                  VecX &du) const
{
    dx           = VecX::Zero(2 * _nv);
    du           = VecX::Zero(u.size());
    du.head(_nv) = _weight * u.head(_nv);
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
    Huu = MatX::Zero(u.size(), u.size());
    Huu.topLeftCorner(_nv, _nv).diagonal().setConstant(_weight);
}

// SquaredForceCost
SquaredForceCost::SquaredForceCost(const uint nv, const double weight)
  : _nv(nv)
  , _weight(weight)
{
}

VecX SquaredForceCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar() << 0.5 * _weight * u.tail(u.size() - _nv).squaredNorm())
      .finished();
}

void SquaredForceCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                VecX &du) const
{
    dx                      = VecX::Zero(2 * _nv);
    du                      = VecX::Zero(u.size());
    du.tail(u.size() - _nv) = _weight * u.tail(u.size() - _nv);
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
    Huu.bottomRightCorner(u.size() - _nv, u.size() - _nv)
      .diagonal()
      .setConstant(_weight);
}

// FrameSquaredPositionErrorCost
FrameSquaredPositionErrorCost::FrameSquaredPositionErrorCost(
  const Model &model, const Vec3 &desired_translation, const std::string &frame,
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

    return (Scalar() << 0.5 * _weight *
                          (pos - _desired_translation).squaredNorm())
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
      _weight * Jfk.topRows(3).transpose() * (pos - _desired_translation);

    du = VecX::Zero(u.size());
}
void FrameSquaredPositionErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                               MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) =
      _weight * Jfk.topRows(3).transpose() * Jfk.topRows(3);
}

void FrameSquaredPositionErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                               MatX &Hxu) const
{
    Hxu = MatX::Zero(2 * _nv, u.size());
}

void FrameSquaredPositionErrorCost::hessian_uu(const VecX &x, const VecX &u,
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
  const uint nq, const uint nv, const double position_weight,
  const double rotation_weight)
  : _model(std::make_shared<Model>(model))
  , _desired_pose(desired_pose)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _weight(Mat6::Zero())
{
    _weight.topLeftCorner<3, 3>()     = position_weight * Mat3::Identity();
    _weight.bottomRightCorner<3, 3>() = rotation_weight * Mat3::Identity();
}

FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const Vec3 &desired_position,
  const Mat3 &desired_rotation, const std::string &frame, const uint nq,
  const uint nv, const double position_weight, const double rotation_weight)
  : FrameSquaredPoseErrorCost(model,
                              Mat4 {
                                { desired_rotation(0, 0),
                                 desired_rotation(0, 1),
                                 desired_rotation(0, 2),
                                 desired_position(0) },
                                { desired_rotation(1, 0),
                                 desired_rotation(1, 1),
                                 desired_rotation(1, 2),
                                 desired_position(1) },
                                { desired_rotation(2, 0),
                                 desired_rotation(2, 1),
                                 desired_rotation(2, 2),
                                 desired_position(2) },
                                { 0.0, 0.0, 0.0, 1.0 }
},
                              frame, nq, nv, position_weight, rotation_weight)
{
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
  , _weight(weight)
{
}

VecX FrameSquaredVelocityCost::evaluate(const VecX &x, const VecX &u) const
{
    return (Scalar()
            << 0.5 *
                 _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true)
                   .squaredNorm())
      .finished();
}

void FrameSquaredVelocityCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                        VecX &du) const
{
    VecX vel = _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true);

    MatX dVdq, dVdv;
    std::tie(dVdq, dVdv) = _model->frameVelocityDerivatives(x.head(_nq),
                                                            x.tail(_nv),
                                                            u.head(_nv),
                                                            _frame,
                                                            true);

    dx = VecX::Zero(2 * _nv);

    dx.head(_nv) = _weight * dVdq.topRows<3>().transpose() * vel;
    dx.tail(_nv) = _weight * dVdv.topRows<3>().transpose() * vel;

    du = VecX::Zero(u.size());
}
void FrameSquaredVelocityCost::hessian_xx(const VecX &x, const VecX &u,
                                          MatX &Hxx) const
{
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = MatX::Zero(2 * _nv, 2 * _nv);
    Hxx.bottomRightCorner(_nv, _nv) =
      _weight * Jfk.topRows(3).transpose() * Jfk.topRows(3);
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
    ;
}

}
}
}
}
}