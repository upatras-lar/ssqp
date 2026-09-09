#include "model/stagewise/predefined/Constraints.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
namespace predefined
{
namespace constraints
{
// JointAngleConstraint
JointAngleConstraint::JointAngleConstraint(const Model &model, const VecX &lb,
                                           const VecX        &ub,
                                           const std::string &name)
  : StageConstraints(name)
  , _lb(lb)
  , _ub(ub)
  , _free_flyer(model.floatingBase())
  , _nv(model.nv())
  , _njoints(_free_flyer ? model.nq() - 7 : model.nq())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use JointAngleConstraint with a custom Model with no custom properties set!");
    // clang-format on
    assert(lb.size() == ub.size() && lb.size() == _njoints &&
           "Invalid bound size");
}

void JointAngleConstraint::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == _njoints && "Invalid bound size");
    _lb = lb;
}

void JointAngleConstraint::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == _njoints && "Invalid bound size");
    _ub = ub;
}

uint JointAngleConstraint::output_size() const
{
    return _njoints;
}

const VecX &JointAngleConstraint::lower_bound() const
{
    return _lb;
}

const VecX &JointAngleConstraint::upper_bound() const
{
    return _ub;
}

VecX JointAngleConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return _free_flyer ? x.segment(7, _njoints) : x.head(_njoints);
}

void JointAngleConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                    MatX &dU) const
{
    dX.setZero(_njoints, _ndx);
    if (_free_flyer)
    {
        dX.block(0, 6, _njoints, _njoints).setIdentity();
    }
    else
    {
        dX.topLeftCorner(_njoints, _njoints).setIdentity();
    }

    dU.setZero(_njoints, u.size());
}

void JointAngleConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void JointAngleConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void JointAngleConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// JointVelocityConstraint
JointVelocityConstraint::JointVelocityConstraint(const Model &model,
                                                 const VecX &lb, const VecX &ub,
                                                 const std::string &name)
  : StageConstraints(name)
  , _lb(lb)
  , _ub(ub)
  , _nv(model.nv())
  , _njoints(model.floatingBase() ? _nv - 6 : _nv)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use JointVelocityConstraint with a custom Model with no custom properties set!");
    // clang-format on
    assert(lb.size() == ub.size() && lb.size() == _njoints &&
           "Invalid bound size");
}

void JointVelocityConstraint::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == _njoints && "Invalid bound size");
    _lb = lb;
}

void JointVelocityConstraint::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == _njoints && "Invalid bound size");
    _ub = ub;
}

uint JointVelocityConstraint::output_size() const
{
    return _njoints;
}

const VecX &JointVelocityConstraint::lower_bound() const
{
    return _lb;
}

const VecX &JointVelocityConstraint::upper_bound() const
{
    return _ub;
}

VecX JointVelocityConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return x.tail(_njoints);
}

void JointVelocityConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                       MatX &dU) const
{
    dX.setZero(_njoints, 2 * _nv);
    dX.bottomRightCorner(_njoints, _njoints).setIdentity();

    dU.setZero(_njoints, u.size());
}

void JointVelocityConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(2 * _nv, 2 * _nv);
}
void JointVelocityConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(2 * _nv, u.size());
}
void JointVelocityConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// JointControlConstraint
JointControlConstraint::JointControlConstraint(const Model &model,
                                               const VecX &lb, const VecX &ub,
                                               const std::string &name)
  : StageConstraints(name)
  , _lb(lb)
  , _ub(ub)
  , _free_flyer(model.floatingBase())
  , _nv(model.nv())
  , _nu(model.nu())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
{
    // clang-format off
    assert((model.xml() || model.hasCustomProperties()) &&
           "Cannot use JointControlConstraint with a custom Model with no custom properties set!");
    // clang-format on
    assert(lb.size() == ub.size() && lb.size() == _nu && "Invalid bound size");
}

void JointControlConstraint::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == _nu && "Invalid bound size");
    _lb = lb;
}

void JointControlConstraint::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == _nu && "Invalid bound size");
    _ub = ub;
}

uint JointControlConstraint::output_size() const
{
    return _nu;
}

const VecX &JointControlConstraint::lower_bound() const
{
    return _lb;
}

const VecX &JointControlConstraint::upper_bound() const
{
    return _ub;
}

VecX JointControlConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return _free_flyer ? u.segment(6, _nu) : u.head(_nu);
}

void JointControlConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                      MatX &dU) const
{
    dX.setZero(_nu, _ndx);
    dU.setZero(_nu, u.size());
    if (_free_flyer)
    {
        dU.block(0, 6, _nu, _nu).setIdentity();
    }
    else
    {
        dU.topLeftCorner(_nu, _nu).setIdentity();
    }
}

void JointControlConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void JointControlConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void JointControlConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FloatingBaseTranslationBound
FloatingBaseTranslationBound::FloatingBaseTranslationBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
{
    assert(model.xml() && "Cannot use FloatingBaseTranslationBound "
                          "with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

uint FloatingBaseTranslationBound::output_size() const
{
    return 3;
}

void FloatingBaseTranslationBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FloatingBaseTranslationBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

const VecX &FloatingBaseTranslationBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBaseTranslationBound::upper_bound() const
{
    return _ub;
}

VecX FloatingBaseTranslationBound::evaluate(const VecX &x, const VecX &u) const
{
    return x.head<3>();
}

// TODO: is this valid? recheck
void FloatingBaseTranslationBound::gradient(const VecX &x, const VecX &u,
                                            MatX &dX, MatX &dU) const
{
    dX.setZero(output_size(), _ndx);

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    dX.leftCols<3>() = R;

    dU.setZero(output_size(), u.size());
}

void FloatingBaseTranslationBound::hessprod_xx(const VecX &x, const VecX &u,
                                               const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FloatingBaseTranslationBound::hessprod_xu(const VecX &x, const VecX &u,
                                               const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void FloatingBaseTranslationBound::hessprod_uu(const VecX &x, const VecX &u,
                                               const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FloatingBaseRotationBound
FloatingBaseRotationBound::FloatingBaseRotationBound(const Model       &model,
                                                     const Mat3        &Rref,
                                                     const Vec3        &lb,
                                                     const Vec3        &ub,
                                                     const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _Rref(Rref)
  , _lb(lb)
  , _ub(ub)
{
    assert(model.xml() &&
           "Cannot use FloatingBaseRotationBound with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

void FloatingBaseRotationBound::set_Rref(const Mat3 &Rref)
{
    _Rref = Rref;
}

void FloatingBaseRotationBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FloatingBaseRotationBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FloatingBaseRotationBound::output_size() const
{
    return 3;
}

const VecX &FloatingBaseRotationBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBaseRotationBound::upper_bound() const
{
    return _ub;
}

VecX FloatingBaseRotationBound::evaluate(const VecX &x, const VecX &u) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    return pinocchio_wrapper::logSO3(_Rref.transpose() * q.toRotationMatrix());
}

void FloatingBaseRotationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                         MatX &dU) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    dX.setZero(3, _ndx);

    dX.block<3, 3>(0, 3) =
      pinocchio_wrapper::JlogSO3(_Rref.transpose() * q.toRotationMatrix());

    dU.setZero(3, u.size());
}

void FloatingBaseRotationBound::hessprod_xx(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FloatingBaseRotationBound::hessprod_xu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FloatingBaseRotationBound::hessprod_uu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FloatingBasePoseBound
FloatingBasePoseBound::FloatingBasePoseBound(const Model &model,
                                             const Mat4 &Tref, const Vec6 &lb,
                                             const Vec6        &ub,
                                             const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _Tref(Tref)
  , _Tref_inv(inverseSE3(Tref))
  , _lb(lb)
  , _ub(ub)
{
    assert(model.xml() &&
           "Cannot use FloatingBasePoseBound with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
}

FloatingBasePoseBound::FloatingBasePoseBound(const Model &model,
                                             const Vec3 &tref, const Mat3 &Rref,
                                             const Vec6 &lb, const Vec6 &ub,
                                             const std::string &name)
  : FloatingBasePoseBound(model,
                          Mat4 {
                            { Rref(0, 0), Rref(0, 1), Rref(0, 2), tref(0) },
                            { Rref(1, 0), Rref(1, 1), Rref(1, 2), tref(1) },
                            { Rref(2, 0), Rref(2, 1), Rref(2, 2), tref(2) },
                            { 0.0, 0.0, 0.0, 1.0 }
},
                          lb, ub, name)
{
}

void FloatingBasePoseBound::set_Tref(const Mat4 &Tref)
{
    _Tref     = Tref;
    _Tref_inv = inverseSE3(Tref);
}

void FloatingBasePoseBound::set_tref(const Vec3 &tref)
{
    _Tref.block<3, 1>(0, 3) = tref.transpose();
    _Tref_inv               = inverseSE3(_Tref);
}

void FloatingBasePoseBound::set_Rref(const Mat3 &Rref)
{
    _Tref.block<3, 3>(0, 0) = Rref;
    _Tref_inv               = inverseSE3(_Tref);
}

void FloatingBasePoseBound::set_lower_bound(const Vec6 &lb)
{
    _lb = lb;
}

void FloatingBasePoseBound::set_upper_bound(const Vec6 &ub)
{
    _ub = ub;
}

uint FloatingBasePoseBound::output_size() const
{
    return 6;
}

const VecX &FloatingBasePoseBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBasePoseBound::upper_bound() const
{
    return _ub;
}

VecX FloatingBasePoseBound::evaluate(const VecX &x, const VecX &u) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    Mat4 Tfl              = Mat4::Zero();
    Tfl.block<3, 3>(0, 0) = q.toRotationMatrix();
    Tfl.block<3, 1>(0, 3) = x.head<3>();
    Tfl(3, 3)             = 1.0;

    return pinocchio_wrapper::logSE3(_Tref_inv * Tfl);
}

void FloatingBasePoseBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                     MatX &dU) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    Mat4 Tfl              = Mat4::Zero();
    Tfl.block<3, 3>(0, 0) = q.toRotationMatrix();
    Tfl.block<3, 1>(0, 3) = x.head<3>();
    Tfl(3, 3)             = 1.0;

    Mat6 Jlog = pinocchio_wrapper::JlogSE3(_Tref_inv * Tfl);

    dX.setZero(6, _ndx);
    dX.topLeftCorner<6, 6>() = Jlog;

    dU.setZero(6, u.size());
}

void FloatingBasePoseBound::hessprod_xx(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FloatingBasePoseBound::hessprod_xu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void FloatingBasePoseBound::hessprod_uu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FloatingBaseLinearVelocityBound
FloatingBaseLinearVelocityBound::FloatingBaseLinearVelocityBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
{
}

void FloatingBaseLinearVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FloatingBaseLinearVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FloatingBaseLinearVelocityBound::output_size() const
{
    return 3;
}

const VecX &FloatingBaseLinearVelocityBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBaseLinearVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FloatingBaseLinearVelocityBound::evaluate(const VecX &x,
                                               const VecX &u) const
{
    return x.segment<3>(_nq);
}

void FloatingBaseLinearVelocityBound::gradient(const VecX &x, const VecX &u,
                                               MatX &dX, MatX &dU) const
{
    dX.setZero(3, _ndx);
    dX.block<3, 3>(0, _nv).setIdentity();

    dU.setZero(3, u.size());
}

void FloatingBaseLinearVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                                  const VecX &mu,
                                                  MatX       &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FloatingBaseLinearVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                                  const VecX &mu,
                                                  MatX       &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void FloatingBaseLinearVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
                                                  const VecX &mu,
                                                  MatX       &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FloatingBaseAngularVelocityBound
FloatingBaseAngularVelocityBound::FloatingBaseAngularVelocityBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
{
}

void FloatingBaseAngularVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FloatingBaseAngularVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FloatingBaseAngularVelocityBound::output_size() const
{
    return 3;
}

const VecX &FloatingBaseAngularVelocityBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBaseAngularVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FloatingBaseAngularVelocityBound::evaluate(const VecX &x,
                                                const VecX &u) const
{
    return x.segment<3>(_nq + 3);
}
void FloatingBaseAngularVelocityBound::gradient(const VecX &x, const VecX &u,
                                                MatX &dX, MatX &dU) const
{
    dX.setZero(3, _ndx);
    dX.block<3, 3>(0, _nv + 3).setIdentity();

    dU.setZero(3, u.size());
}

void FloatingBaseAngularVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatX       &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FloatingBaseAngularVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatX       &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void FloatingBaseAngularVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatX       &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FrameTranslationBound
FrameTranslationBound::FrameTranslationBound(const Model       &model,
                                             const std::string &frame,
                                             const Vec3 &lb, const Vec3 &ub,
                                             const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _lb(lb)
  , _ub(ub)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameTranslationBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameTranslationBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameTranslationBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FrameTranslationBound::output_size() const
{
    return 3;
}

const VecX &FrameTranslationBound::lower_bound() const
{
    return _lb;
}
const VecX &FrameTranslationBound::upper_bound() const
{
    return _ub;
}

VecX FrameTranslationBound::evaluate(const VecX &x, const VecX &u) const
{
    Vec3 pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame_id);
    return pos;
}

void FrameTranslationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                     MatX &dU) const
{
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    dX.setZero(3, _ndx);
    dX.leftCols(_nv) = _jac_scratch.topRows(3);

    dU.setZero(3, u.size());
}

void FrameTranslationBound::hessprod_xx(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void FrameTranslationBound::hessprod_xu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FrameTranslationBound::hessprod_uu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FrameRotationBound
FrameRotationBound::FrameRotationBound(const Model       &model,
                                       const std::string &frame,
                                       const Mat3 &Rref, const Vec3 &lb,
                                       const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _Rref(Rref)
  , _lb(lb)
  , _ub(ub)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameRotatationBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameRotationBound::set_Rref(const Mat3 &Rref)
{
    _Rref = Rref;
}

void FrameRotationBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}
void FrameRotationBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FrameRotationBound::output_size() const
{
    return 3;
}

const VecX &FrameRotationBound::lower_bound() const
{
    return _lb;
}
const VecX &FrameRotationBound::upper_bound() const
{
    return _ub;
}

VecX FrameRotationBound::evaluate(const VecX &x, const VecX &u) const
{
    Mat3 Rf;
    std::tie(std::ignore, Rf) = _model->forwardKinematics(x.head(_nq), _frame_id);

    return pinocchio_wrapper::logSO3(_Rref.transpose() * Rf);
}

void FrameRotationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                  MatX &dU) const
{
    Mat3 Rf;
    std::tie(std::ignore, Rf) = _model->forwardKinematics(x.head(_nq), _frame_id);

    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);
    MatX Jw   = _jac_scratch.bottomRows(3);
    Mat3 Jlog = pinocchio_wrapper::JlogSO3(_Rref.transpose() * Rf);

    dX.setZero(3, _ndx);
    dX.leftCols(_nv) = Jlog * Jw;

    dU.setZero(3, u.size());
}

void FrameRotationBound::hessprod_xx(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void FrameRotationBound::hessprod_xu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FrameRotationBound::hessprod_uu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FramePoseBound
FramePoseBound::FramePoseBound(const Model &model, const std::string &frame,
                               const Mat4 &Tref, const Vec6 &lb, const Vec6 &ub,
                               const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _Tref(Tref)
  , _Tref_inv(inverseSE3(Tref))
  , _lb(lb)
  , _ub(ub)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use FramePoseBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

FramePoseBound::FramePoseBound(const Model &model, const std::string &frame,
                               const Vec3 &tref, const Mat3 &Rref,
                               const Vec6 &lb, const Vec6 &ub,
                               const std::string &name)
  : FramePoseBound(model, frame,
                   Mat4 {
                     { Rref(0, 0), Rref(0, 1), Rref(0, 2), tref(0) },
                     { Rref(1, 0), Rref(1, 1), Rref(1, 2), tref(1) },
                     { Rref(2, 0), Rref(2, 1), Rref(2, 2), tref(2) },
                     { 0.0, 0.0, 0.0, 1.0 }
},
                   lb, ub, name)
{
}

void FramePoseBound::set_Tref(const Mat4 &Tref)
{
    _Tref     = Tref;
    _Tref_inv = inverseSE3(Tref);
}

void FramePoseBound::set_tref(const Vec3 &tref)
{
    _Tref.block<3, 1>(0, 3) = tref.transpose();
    _Tref_inv               = inverseSE3(_Tref);
}

void FramePoseBound::set_Rref(const Mat3 &Rref)
{
    _Tref.block<3, 3>(0, 0) = Rref;
    _Tref_inv               = inverseSE3(_Tref);
}

void FramePoseBound::set_lower_bound(const Vec6 &lb)
{
    _lb = lb;
}

void FramePoseBound::set_upper_bound(const Vec6 &ub)
{
    _ub = ub;
}

uint FramePoseBound::output_size() const
{
    return 6;
}

const VecX &FramePoseBound::lower_bound() const
{
    return _lb;
}

const VecX &FramePoseBound::upper_bound() const
{
    return _ub;
}

VecX FramePoseBound::evaluate(const VecX &x, const VecX &u) const
{
    Mat4 Tf = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);
    return pinocchio_wrapper::logSE3(_Tref_inv * Tf);
}

void FramePoseBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                              MatX &dU) const
{
    Mat4 Tf = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);

    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);
    Mat6 Jlog = pinocchio_wrapper::JlogSE3(_Tref_inv * Tf);

    dX.setZero(6, _ndx);
    dX.leftCols(_nv) = Jlog * _jac_scratch;

    dU.setZero(6, u.size());
}

void FramePoseBound::hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                                 MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void FramePoseBound::hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                                 MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void FramePoseBound::hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                                 MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FrameLinearVelocityBound
FrameLinearVelocityBound::FrameLinearVelocityBound(const Model       &model,
                                                   const std::string &frame,
                                                   const Vec3        &lb,
                                                   const Vec3        &ub,
                                                   const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _lb(lb)
  , _ub(ub)
  , _dVdq_scratch(MatX::Zero(6, _nv))
  , _dVdv_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameLinearVelocityBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameLinearVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameLinearVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FrameLinearVelocityBound::output_size() const
{
    return 3;
}

const VecX &FrameLinearVelocityBound::lower_bound() const
{
    return _lb;
}
const VecX &FrameLinearVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FrameLinearVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    return _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, true)
      .head<3>();
}

void FrameLinearVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                        MatX &dU) const
{
    _model->frameVelocityDerivatives(x.head(_nq), x.tail(_nv), _frame_id, true,
                                     _dVdq_scratch, _dVdv_scratch);

    dX.setZero(3, _ndx);
    dX.leftCols(_nv)  = _dVdq_scratch.topRows<3>();
    dX.rightCols(_nv) = _dVdv_scratch.topRows<3>();

    dU.setZero(3, u.size());
}

void FrameLinearVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void FrameLinearVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FrameLinearVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// FrameAngularVelocityBound
FrameAngularVelocityBound::FrameAngularVelocityBound(const Model       &model,
                                                     const std::string &frame,
                                                     const Vec3        &lb,
                                                     const Vec3        &ub,
                                                     const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _lb(lb)
  , _ub(ub)
  , _dVdq_scratch(MatX::Zero(6, _nv))
  , _dVdv_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameAngularVelocityBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameAngularVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameAngularVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint FrameAngularVelocityBound::output_size() const
{
    return 3;
}

const VecX &FrameAngularVelocityBound::lower_bound() const
{
    return _lb;
}

const VecX &FrameAngularVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FrameAngularVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    return _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, true)
      .tail<3>();
}

void FrameAngularVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                         MatX &dU) const
{
    _model->frameVelocityDerivatives(x.head(_nq), x.tail(_nv), _frame_id, true,
                                     _dVdq_scratch, _dVdv_scratch);

    dX.setZero(3, _ndx);
    dX.leftCols(_nv)  = _dVdq_scratch.bottomRows<3>();
    dX.rightCols(_nv) = _dVdv_scratch.bottomRows<3>();

    dU.setZero(3, u.size());
}

void FrameAngularVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void FrameAngularVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FrameAngularVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// CenterOfMassBound
CenterOfMassBound::CenterOfMassBound(const Model &model, const Vec3 &lb,
                                     const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
{
    assert(model.xml() && "Cannot use CenterOfMassBound with a custom Model");
}

void CenterOfMassBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void CenterOfMassBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

uint CenterOfMassBound::output_size() const
{
    return 3;
}

const VecX &CenterOfMassBound::lower_bound() const
{
    return _lb;
}

const VecX &CenterOfMassBound::upper_bound() const
{
    return _ub;
}

VecX CenterOfMassBound::evaluate(const VecX &x, const VecX &u) const
{
    return _model->centerOfMass(x.head(_nq));
}

void CenterOfMassBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                 MatX &dU) const
{
    dX.setZero(3, _ndx);
    dX.leftCols(_nv) = _model->jacobianCenterOfMass(x.head(_nq));

    dU.setZero(3, u.size());
}

void CenterOfMassBound::hessprod_xx(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}

void CenterOfMassBound::hessprod_xu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}

void CenterOfMassBound::hessprod_uu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

// ContactConstraint
ContactConstraint::ContactConstraint(const Model       &model,
                                     const std::string &frame, const double mu,
                                     const double       max_normal_force,
                                     const double       eq_tol,
                                     const std::string &name)
  : StageConstraints(name)
  , _model(std::make_shared<Model>(model))
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _mu(mu)
  , _max_normal_force(max_normal_force)
  , _eq_tol(eq_tol)
  , _lb((Vec6() << 0.0, -std::numeric_limits<double>::infinity(), 0.0,
         -std::numeric_limits<double>::infinity(), 0.0, -eq_tol)
          .finished())
  , _ub((Vec6() << max_normal_force, 0.0,
         std::numeric_limits<double>::infinity(), 0.0,
         std::numeric_limits<double>::infinity(), eq_tol)
          .finished())
  , _jac_world_scratch(MatX::Zero(6, _nv))
  , _jac_local_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use ContactConstraint with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    assert(max_normal_force >= 0.0 &&
           "Max normal force value must be positive");
}

void ContactConstraint::set_mu(const double mu)
{
    _mu = mu;
}

void ContactConstraint::set_enabled(const bool enabled)
{
    StageConstraints::set_enabled(enabled);
    _force->ignore = !enabled;
}

uint ContactConstraint::output_size() const
{
    return 6;
}

uint ContactConstraint::required_forces() const
{
    return 1;
}

std::string ContactConstraint::force_frame(const uint i) const
{
    return _frame;
}

void ContactConstraint::set_forces(
  const std::vector<std::shared_ptr<Force>> &forces)
{
    assert(_frame == forces[0]->frame &&
           "Force frame does not match constraint frame");
    _force = forces[0];
}

const VecX &ContactConstraint::lower_bound() const
{
    return _lb;
}

const VecX &ContactConstraint::upper_bound() const
{
    return _ub;
}

VecX ContactConstraint::evaluate(const VecX &x, const VecX &u) const
{
    std::shared_ptr<terrain::Terrain> terrain = _model->terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    Vec3 f_global = R * u.segment(_nu + 3 * _force->force_index, 3);

    double fn  = (f_global.transpose() * tp.normal)[0];
    double ft1 = (f_global.transpose() * tp.t1)[0];
    double ft2 = (f_global.transpose() * tp.t2)[0];

    Vec6 res;
    res[0] = fn;
    res[1] = ft1 - _mu * fn;
    res[2] = ft1 + _mu * fn;
    res[3] = ft2 - _mu * fn;
    res[4] = ft2 + _mu * fn;
    res[5] = pos[2] - tp.height;

    return res;
}

void ContactConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                 MatX &dU) const
{
    std::shared_ptr<terrain::Terrain> terrain = _model->terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    dX.setZero(6, 2 * _nv);
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_world_scratch);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_local_scratch);
    MatX Jfw = _jac_world_scratch.topRows<3>();
    MatX Jfl = _jac_local_scratch.bottomRows<3>();

    uint total_force_index = _nu + 3 * _force->force_index;

    Vec3 f_local = u.segment(total_force_index, 3);

    Mat3 f_skew;
    f_skew << 0, -f_local[2], f_local[1], f_local[2], 0, -f_local[0],
      -f_local[1], f_local[0], 0;

    MatX dfglobal_dq = -R * f_skew * Jfl;

    Vec3 dhdp = (Vec3() << -tp.dhdx, -tp.dhdy, 1).finished();

    VecX dfn_dq  = dfglobal_dq.transpose() * tp.normal;
    VecX dft1_dq = dfglobal_dq.transpose() * tp.t1;
    VecX dft2_dq = dfglobal_dq.transpose() * tp.t2;

    dX.row(0).head(_nv) = dfn_dq.transpose();
    dX.row(1).head(_nv) = (dft1_dq - _mu * dfn_dq).transpose();
    dX.row(2).head(_nv) = (dft1_dq + _mu * dfn_dq).transpose();
    dX.row(3).head(_nv) = (dft2_dq - _mu * dfn_dq).transpose();
    dX.row(4).head(_nv) = (dft2_dq + _mu * dfn_dq).transpose();

    dX.row(5).head(_nv) = dhdp.transpose() * Jfw;

    Vec3 f_global = R * f_local;

    double fn  = (f_global.transpose() * tp.normal)[0];
    double ft1 = (f_global.transpose() * tp.t1)[0];
    double ft2 = (f_global.transpose() * tp.t2)[0];

    dU.setZero(6, u.size());

    Vec3 dfn_dflocal  = R.transpose() * tp.normal;
    Vec3 dft1_dflocal = R.transpose() * tp.t1;
    Vec3 dft2_dflocal = R.transpose() * tp.t2;

    dU.block<1, 3>(0, total_force_index) = dfn_dflocal.transpose();
    dU.block<1, 3>(1, total_force_index) =
      (dft1_dflocal - _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(2, total_force_index) =
      (dft1_dflocal + _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(3, total_force_index) =
      (dft2_dflocal - _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(4, total_force_index) =
      (dft2_dflocal + _mu * dfn_dflocal).transpose();
}

void ContactConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(2 * _nv, 2 * _nv);
}

void ContactConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(2 * _nv, u.size());
}
void ContactConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

}
}
}
}
}