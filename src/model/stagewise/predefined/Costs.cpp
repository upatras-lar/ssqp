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
// FrameSquaredPoseErrorCost
FrameSquaredPoseErrorCost::FrameSquaredPoseErrorCost(
  const Model &model, const VecX &desired_translation,
  const MatX &desired_rotation, const std::string &frame, const uint nq,
  const uint nv, const double position_weight, const double rotation_weight,
  const double velocity_weight, const double control_weight)
  : _model(std::make_shared<Model>(model))
  , _desired_translation(desired_translation)
  , _desired_rotation(desired_rotation)
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _P_WGT(position_weight)
  , _R_WGT(rotation_weight)
  , _V_WGT(velocity_weight)
  , _U_WGT(control_weight)
{
}

VecX FrameSquaredPoseErrorCost::evaluate(const VecX &x, const VecX &u) const
{
    VecX pos;
    MatX rot;

    std::tie(pos, rot) = _model->forwardKinematics(x.head(_nq), _frame);

    Eigen::Vector<double, 1> res;

    res(0) =
      _P_WGT * (pos - _desired_translation).squaredNorm() +
      _R_WGT * pinocchio_wrapper::logSO3(_desired_rotation.transpose() * rot)
                 .squaredNorm() +
      _V_WGT * x.tail(_nv).squaredNorm() + _U_WGT * u.squaredNorm();
    return res;
}

// TODO: Rotation incorrect (??)
void FrameSquaredPoseErrorCost::gradient(const VecX &x, const VecX &u, VecX &dx,
                                         VecX &du) const
{
    VecX pos;
    MatX rot;
    std::tie(pos, rot)  = _model->forwardKinematics(x.head(_nq), _frame);
    Eigen::MatrixXd Jfk = _model->frameJacobian(x.head(_nq), _frame);

    dx = Eigen::VectorXd::Zero(2 * _nv);

    MatX rp = _desired_rotation.transpose() * rot;
    dx.head(_nv) =
      2 * _P_WGT * Jfk.topRows(3).transpose() * (pos - _desired_translation) +
      2 * _R_WGT * Jfk.bottomRows(3).transpose() * _desired_rotation *
        pinocchio_wrapper::JlogSO3(rp) * pinocchio_wrapper::logSO3(rp);

    dx.tail(_nv) = 2 * _V_WGT * x.tail(_nv);

    du = 2 * _U_WGT * u;
}
void FrameSquaredPoseErrorCost::hessian_xx(const VecX &x, const VecX &u,
                                           MatX &Hxx) const
{
    Eigen::MatrixXd Jfk = _model->frameJacobian(x.head(_nq), _frame);

    Hxx = Eigen::MatrixXd::Zero(2 * _nv, 2 * _nv);
    Hxx.topLeftCorner(_nv, _nv) =
      2 * _P_WGT * Jfk.topRows(3).transpose() * Jfk.topRows(3) +
      2 * _R_WGT * Jfk.bottomRows(3).transpose() * Jfk.bottomRows(3);
    Hxx.bottomRightCorner(_nv, _nv) =
      2 * _V_WGT * Eigen::MatrixXd::Identity(_nv, _nv);
}
void FrameSquaredPoseErrorCost::hessian_xu(const VecX &x, const VecX &u,
                                           MatX &Hxu) const
{
    Hxu = Eigen::MatrixXd::Zero(2 * _nv, _nv);
}
void FrameSquaredPoseErrorCost::hessian_uu(const VecX &x, const VecX &u,
                                           MatX &Huu) const
{
    Huu = Eigen::MatrixXd::Zero(_nv, _nv);
    Huu.diagonal().setConstant(2 * _U_WGT);
}
}
}
}
}
}