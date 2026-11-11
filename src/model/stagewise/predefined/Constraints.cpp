#include "model/stagewise/predefined/Constraints.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"

#include <numeric>

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
namespace
{
// Rows 0 to n - 1, for constraints whose rows all depend only on the state
std::vector<uint> all_rows(const uint n)
{
    std::vector<uint> rows(n);
    std::iota(rows.begin(), rows.end(), 0);
    return rows;
}
}

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

std::vector<uint> JointAngleConstraint::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

std::vector<BoxRow> JointAngleConstraint::box_rows() const
{
    const uint offset = _free_flyer ? 6 : 0;

    std::vector<BoxRow> rows;
    for (uint i = 0; i < _njoints; ++i)
    {
        rows.push_back({ i, offset + i, false });
    }

    return rows;
}

VecX JointAngleConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return _free_flyer ? x.segment(7, _njoints) : x.head(_njoints);
}

void JointAngleConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                    MatX &dU) const
{
    dX.setZero(_njoints, _ndx);
    dU.setZero(_njoints, u.size());
    add_gradient(x, u, dX, dU);
}

void JointAngleConstraint::add_gradient(const VecX &x, const VecX &u,
                                        MatXRef dX, MatXRef dU) const
{
    if (_free_flyer)
    {
        dX.block(0, 6, _njoints, _njoints).diagonal().array() += 1.0;
    }
    else
    {
        dX.topLeftCorner(_njoints, _njoints).diagonal().array() += 1.0;
    }
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

void JointAngleConstraint::add_hessprod_xx(const VecX &x, const VecX &u,
                                           const VecX &mu, MatXRef Pxx) const
{
}

void JointAngleConstraint::add_hessprod_xu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatXRef Pxu) const
{
}

void JointAngleConstraint::add_hessprod_uu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatXRef Puu) const
{
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

std::vector<uint> JointVelocityConstraint::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

std::vector<BoxRow> JointVelocityConstraint::box_rows() const
{
    std::vector<BoxRow> rows;
    for (uint i = 0; i < _njoints; ++i)
    {
        rows.push_back({ i, 2 * _nv - _njoints + i, false });
    }

    return rows;
}

VecX JointVelocityConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return x.tail(_njoints);
}

void JointVelocityConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                       MatX &dU) const
{
    dX.setZero(_njoints, 2 * _nv);
    dU.setZero(_njoints, u.size());
    add_gradient(x, u, dX, dU);
}

void JointVelocityConstraint::add_gradient(const VecX &x, const VecX &u,
                                           MatXRef dX, MatXRef dU) const
{
    dX.bottomRightCorner(_njoints, _njoints).diagonal().array() += 1.0;
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

void JointVelocityConstraint::add_hessprod_xx(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Pxx) const
{
}

void JointVelocityConstraint::add_hessprod_xu(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Pxu) const
{
}

void JointVelocityConstraint::add_hessprod_uu(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Puu) const
{
}

// JointControlConstraint
JointControlConstraint::JointControlConstraint(const Model &model,
                                               const VecX &lb, const VecX &ub,
                                               const std::string &name)
  : StageConstraints(name)
  , _lb(lb)
  , _ub(ub)
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

std::vector<BoxRow> JointControlConstraint::box_rows() const
{
    std::vector<BoxRow> rows;
    for (uint i = 0; i < _nu; ++i)
    {
        rows.push_back({ i, i, true });
    }

    return rows;
}

VecX JointControlConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return u.head(_nu);
}

void JointControlConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                      MatX &dU) const
{
    dX.setZero(_nu, _ndx);
    dU.setZero(_nu, u.size());
    add_gradient(x, u, dX, dU);
}

void JointControlConstraint::add_gradient(const VecX &x, const VecX &u,
                                          MatXRef dX, MatXRef dU) const
{
    dU.topLeftCorner(_nu, _nu).diagonal().array() += 1.0;
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

void JointControlConstraint::add_hessprod_xx(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Pxx) const
{
}

void JointControlConstraint::add_hessprod_xu(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Pxu) const
{
}

void JointControlConstraint::add_hessprod_uu(const VecX &x, const VecX &u,
                                             const VecX &mu, MatXRef Puu) const
{
}

// FloatingBaseTranslationBound
FloatingBaseTranslationBound::FloatingBaseTranslationBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const Vec3 &center,
  const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
  , _center(center)
{
    assert(model.xml() && "Cannot use FloatingBaseTranslationBound "
                          "with a custom Model");
    assert(model.floatingBase() && "Provided Model has no floating base");
    set_reference_frame(reference_frame);
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

void FloatingBaseTranslationBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FloatingBaseTranslationBound");
    // The base's own axes are the base axes
    _reference_frame = reference_frame.type() == ReferenceFrame::Local
                         ? ReferenceFrame(ReferenceFrame::Base)
                         : reference_frame;
}

void FloatingBaseTranslationBound::set_center(const Vec3 &center)
{
    _center = center;
}

const VecX &FloatingBaseTranslationBound::lower_bound() const
{
    return _lb;
}

const VecX &FloatingBaseTranslationBound::upper_bound() const
{
    return _ub;
}

std::vector<uint> FloatingBaseTranslationBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FloatingBaseTranslationBound::evaluate(const VecX &x, const VecX &u) const
{
    return _reference_frame.axes(x).transpose() * (x.head<3>() - _center);
}

// TODO: is this valid? recheck
void FloatingBaseTranslationBound::gradient(const VecX &x, const VecX &u,
                                            MatX &dX, MatX &dU) const
{
    dX.setZero(output_size(), _ndx);
    dU.setZero(output_size(), u.size());
    add_gradient(x, u, dX, dU);
}

void FloatingBaseTranslationBound::add_gradient(const VecX &x, const VecX &u,
                                                MatXRef dX, MatXRef dU) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    Mat3               R = q.toRotationMatrix();

    const Mat3 A = _reference_frame.axes(x);

    dX.leftCols<3>() += A.transpose() * R;

    // Base axes turn with the base, which adds the bound's value crossed with
    // the base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, 3) += skew(A.transpose() * (x.head<3>() - _center));
    }
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

void FloatingBaseTranslationBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatXRef     Pxx) const
{
}

void FloatingBaseTranslationBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatXRef     Pxu) const
{
}

void FloatingBaseTranslationBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                                   const VecX &mu,
                                                   MatXRef     Puu) const
{
}

// FloatingBaseRotationBound
FloatingBaseRotationBound::FloatingBaseRotationBound(const Model       &model,
                                                     const Mat3        &Rref,
                                                     const Vec3        &lb,
                                                     const Vec3        &ub,
                                                     const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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

std::vector<uint> FloatingBaseRotationBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FloatingBaseRotationBound::evaluate(const VecX &x, const VecX &u) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    return pinocchio_wrapper::logSO3(_Rref.transpose() * q.toRotationMatrix());
}

void FloatingBaseRotationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                         MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FloatingBaseRotationBound::add_gradient(const VecX &x, const VecX &u,
                                             MatXRef dX, MatXRef dU) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    dX.block<3, 3>(0, 3) +=
      pinocchio_wrapper::JlogSO3(_Rref.transpose() * q.toRotationMatrix());
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

void FloatingBaseRotationBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Pxx) const
{
}

void FloatingBaseRotationBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Pxu) const
{
}

void FloatingBaseRotationBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Puu) const
{
}

// FloatingBasePoseBound
FloatingBasePoseBound::FloatingBasePoseBound(const Model &model,
                                             const Mat4 &Tref, const Vec6 &lb,
                                             const Vec6        &ub,
                                             const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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

std::vector<uint> FloatingBasePoseBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
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
    dX.setZero(6, _ndx);
    dU.setZero(6, u.size());
    add_gradient(x, u, dX, dU);
}

void FloatingBasePoseBound::add_gradient(const VecX &x, const VecX &u,
                                         MatXRef dX, MatXRef dU) const
{
    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);

    Mat4 Tfl              = Mat4::Zero();
    Tfl.block<3, 3>(0, 0) = q.toRotationMatrix();
    Tfl.block<3, 1>(0, 3) = x.head<3>();
    Tfl(3, 3)             = 1.0;

    Mat6 Jlog = pinocchio_wrapper::JlogSE3(_Tref_inv * Tfl);

    dX.topLeftCorner<6, 6>() += Jlog;
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

void FloatingBasePoseBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Pxx) const
{
}

void FloatingBasePoseBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Pxu) const
{
}

void FloatingBasePoseBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Puu) const
{
}

// FloatingBaseLinearVelocityBound
FloatingBaseLinearVelocityBound::FloatingBaseLinearVelocityBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
  , _reference_frame(reference_frame)
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

void FloatingBaseLinearVelocityBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    _reference_frame = reference_frame;
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

std::vector<uint> FloatingBaseLinearVelocityBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

std::vector<BoxRow> FloatingBaseLinearVelocityBound::box_rows() const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type != ReferenceFrame::Local && type != ReferenceFrame::Base)
    {
        return {};
    }

    return {
        { 0,     _nv, false },
        { 1, _nv + 1, false },
        { 2, _nv + 2, false }
    };
}

VecX FloatingBaseLinearVelocityBound::evaluate(const VecX &x,
                                               const VecX &u) const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        return x.segment<3>(_nq);
    }

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    return _reference_frame.rotation().transpose() * q.toRotationMatrix() *
           x.segment<3>(_nq);
}

void FloatingBaseLinearVelocityBound::gradient(const VecX &x, const VecX &u,
                                               MatX &dX, MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FloatingBaseLinearVelocityBound::add_gradient(const VecX &x, const VecX &u,
                                                   MatXRef dX, MatXRef dU) const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, _nv).diagonal().array() += 1.0;
    }
    else
    {
        // The base velocity rotated by the base's rotation, which turns with
        // the base's angular tangent
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        const Mat3         A =
          _reference_frame.rotation().transpose() * q.toRotationMatrix();

        dX.block<3, 3>(0, _nv) += A;
        dX.block<3, 3>(0, 3)   -= A * skew(x.segment<3>(_nq));
    }
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

void FloatingBaseLinearVelocityBound::add_hessprod_xx(const VecX &x,
                                                      const VecX &u,
                                                      const VecX &mu,
                                                      MatXRef     Pxx) const
{
}

void FloatingBaseLinearVelocityBound::add_hessprod_xu(const VecX &x,
                                                      const VecX &u,
                                                      const VecX &mu,
                                                      MatXRef     Pxu) const
{
}

void FloatingBaseLinearVelocityBound::add_hessprod_uu(const VecX &x,
                                                      const VecX &u,
                                                      const VecX &mu,
                                                      MatXRef     Puu) const
{
}

// FloatingBaseAngularVelocityBound
FloatingBaseAngularVelocityBound::FloatingBaseAngularVelocityBound(
  const Model &model, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
  , _reference_frame(reference_frame)
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

void FloatingBaseAngularVelocityBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    _reference_frame = reference_frame;
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

std::vector<uint> FloatingBaseAngularVelocityBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

std::vector<BoxRow> FloatingBaseAngularVelocityBound::box_rows() const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type != ReferenceFrame::Local && type != ReferenceFrame::Base)
    {
        return {};
    }

    return {
        { 0, _nv + 3, false },
        { 1, _nv + 4, false },
        { 2, _nv + 5, false }
    };
}

VecX FloatingBaseAngularVelocityBound::evaluate(const VecX &x,
                                                const VecX &u) const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        return x.segment<3>(_nq + 3);
    }

    Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
    return _reference_frame.rotation().transpose() * q.toRotationMatrix() *
           x.segment<3>(_nq + 3);
}
void FloatingBaseAngularVelocityBound::gradient(const VecX &x, const VecX &u,
                                                MatX &dX, MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FloatingBaseAngularVelocityBound::add_gradient(const VecX &x,
                                                    const VecX &u, MatXRef dX,
                                                    MatXRef dU) const
{
    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, _nv + 3).diagonal().array() += 1.0;
    }
    else
    {
        // The base velocity rotated by the base's rotation, which turns with
        // the base's angular tangent
        Eigen::Quaterniond q(x[6], x[3], x[4], x[5]);
        const Mat3         A =
          _reference_frame.rotation().transpose() * q.toRotationMatrix();

        dX.block<3, 3>(0, _nv + 3) += A;
        dX.block<3, 3>(0, 3)       -= A * skew(x.segment<3>(_nq + 3));
    }
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

void FloatingBaseAngularVelocityBound::add_hessprod_xx(const VecX &x,
                                                       const VecX &u,
                                                       const VecX &mu,
                                                       MatXRef     Pxx) const
{
}

void FloatingBaseAngularVelocityBound::add_hessprod_xu(const VecX &x,
                                                       const VecX &u,
                                                       const VecX &mu,
                                                       MatXRef     Pxu) const
{
}

void FloatingBaseAngularVelocityBound::add_hessprod_uu(const VecX &x,
                                                       const VecX &u,
                                                       const VecX &mu,
                                                       MatXRef     Puu) const
{
}

// FrameTranslationBound
FrameTranslationBound::FrameTranslationBound(
  const Model &model, const std::string &frame, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const Vec3 &center,
  const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _lb(lb)
  , _ub(ub)
  , _center(center)
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameTranslationBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    set_reference_frame(reference_frame);
}

void FrameTranslationBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameTranslationBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

void FrameTranslationBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FrameTranslationBound",
                            _model->floatingBase());
    _reference_frame = reference_frame;
}

void FrameTranslationBound::set_center(const Vec3 &center)
{
    _center = center;
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

std::vector<uint> FrameTranslationBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FrameTranslationBound::evaluate(const VecX &x, const VecX &u) const
{
    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    const Mat3 A = _reference_frame.type() == ReferenceFrame::Local
                     ? R
                     : _reference_frame.axes(x);
    return A.transpose() * (pos - _center);
}

void FrameTranslationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                     MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FrameTranslationBound::add_gradient(const VecX &x, const VecX &u,
                                         MatXRef dX, MatXRef dU) const
{
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    const ReferenceFrame::Type type = _reference_frame.type();
    if (type == ReferenceFrame::Local || type == ReferenceFrame::Base)
    {
        // The axes turn with the frame or the base, which adds the bound's
        // value crossed with their angular tangent
        Vec3 pos;
        Mat3 R;
        std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

        const Mat3 A =
          type == ReferenceFrame::Local ? R : _reference_frame.axes(x);
        const Vec3 value = A.transpose() * (pos - _center);

        dX.leftCols(_nv).noalias() += A.transpose() * _jac_scratch.topRows(3);
        if (type == ReferenceFrame::Local)
        {
            dX.leftCols(_nv).noalias() +=
              skew(value) * R.transpose() * _jac_scratch.bottomRows(3);
        }
        else
        {
            dX.block<3, 3>(0, 3) += skew(value);
        }
    }
    else
    {
        dX.leftCols(_nv).noalias() +=
          _reference_frame.rotation().transpose() * _jac_scratch.topRows(3);
    }
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

void FrameTranslationBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Pxx) const
{
}

void FrameTranslationBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Pxu) const
{
}

void FrameTranslationBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                            const VecX &mu, MatXRef Puu) const
{
}

// FrameTerrainHeightBound
FrameTerrainHeightBound::FrameTerrainHeightBound(const Model       &model,
                                                 const std::string &frame,
                                                 const double       lb,
                                                 const double       ub,
                                                 const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _lb(Scalar::Constant(lb))
  , _ub(Scalar::Constant(ub))
  , _jac_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() &&
           "Cannot use FrameTerrainHeightBound with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
}

void FrameTerrainHeightBound::set_lower_bound(const double lb)
{
    _lb[0] = lb;
}

void FrameTerrainHeightBound::set_upper_bound(const double ub)
{
    _ub[0] = ub;
}

uint FrameTerrainHeightBound::output_size() const
{
    return 1;
}

const VecX &FrameTerrainHeightBound::lower_bound() const
{
    return _lb;
}
const VecX &FrameTerrainHeightBound::upper_bound() const
{
    return _ub;
}

std::vector<uint> FrameTerrainHeightBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FrameTerrainHeightBound::evaluate(const VecX &x, const VecX &u) const
{
    Vec3 pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), _frame_id);

    return Scalar::Constant(pos[2] -
                            _model->terrain()->at(pos[0], pos[1]).height);
}

void FrameTerrainHeightBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                       MatX &dU) const
{
    dX.setZero(1, _ndx);
    dU.setZero(1, u.size());
    add_gradient(x, u, dX, dU);
}

void FrameTerrainHeightBound::add_gradient(const VecX &x, const VecX &u,
                                           MatXRef dX, MatXRef dU) const
{
    Vec3 pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), _frame_id);
    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_scratch);

    terrain::Terrain::TerrainPoint tp = _model->terrain()->at(pos[0], pos[1]);

    dX.leftCols(_nv) +=
      Vec3(-tp.dhdx, -tp.dhdy, 1).transpose() * _jac_scratch.topRows(3);
}

void FrameTerrainHeightBound::hessprod_xx(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxx) const
{
    Pxx.setZero(_ndx, _ndx);
}
void FrameTerrainHeightBound::hessprod_xu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxu) const
{
    Pxu.setZero(_ndx, u.size());
}
void FrameTerrainHeightBound::hessprod_uu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Puu) const
{
    Puu.setZero(u.size(), u.size());
}

void FrameTerrainHeightBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Pxx) const
{
}

void FrameTerrainHeightBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Pxu) const
{
}

void FrameTerrainHeightBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                              const VecX &mu, MatXRef Puu) const
{
}

// FrameRotationBound
FrameRotationBound::FrameRotationBound(const Model       &model,
                                       const std::string &frame,
                                       const Mat3 &Rref, const Vec3 &lb,
                                       const Vec3 &ub, const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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

std::vector<uint> FrameRotationBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FrameRotationBound::evaluate(const VecX &x, const VecX &u) const
{
    Mat3 Rf;
    std::tie(std::ignore, Rf) =
      _model->forwardKinematics(x.head(_nq), _frame_id);

    return pinocchio_wrapper::logSO3(_Rref.transpose() * Rf);
}

void FrameRotationBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                  MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FrameRotationBound::add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                                      MatXRef dU) const
{
    Mat3 Rf;
    std::tie(std::ignore, Rf) =
      _model->forwardKinematics(x.head(_nq), _frame_id);

    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);
    MatX Jw   = _jac_scratch.bottomRows(3);
    Mat3 Jlog = pinocchio_wrapper::JlogSO3(_Rref.transpose() * Rf);

    dX.leftCols(_nv) += Jlog * Jw;
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

void FrameRotationBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                         const VecX &mu, MatXRef Pxx) const
{
}

void FrameRotationBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatXRef Pxu) const
{
}

void FrameRotationBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatXRef Puu) const
{
}

// FramePoseBound
FramePoseBound::FramePoseBound(const Model &model, const std::string &frame,
                               const Mat4 &Tref, const Vec6 &lb, const Vec6 &ub,
                               const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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

std::vector<uint> FramePoseBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FramePoseBound::evaluate(const VecX &x, const VecX &u) const
{
    Mat4 Tf = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);
    return pinocchio_wrapper::logSE3(_Tref_inv * Tf);
}

void FramePoseBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                              MatX &dU) const
{
    dX.setZero(6, _ndx);
    dU.setZero(6, u.size());
    add_gradient(x, u, dX, dU);
}

void FramePoseBound::add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                                  MatXRef dU) const
{
    Mat4 Tf = _model->forwardKinematicsSE3(x.head(_nq), _frame_id);

    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_scratch);
    Mat6 Jlog = pinocchio_wrapper::JlogSE3(_Tref_inv * Tf);

    dX.leftCols(_nv) += Jlog * _jac_scratch;
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

void FramePoseBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                     const VecX &mu, MatXRef Pxx) const
{
}

void FramePoseBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatXRef Pxu) const
{
}

void FramePoseBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatXRef Puu) const
{
}

// FrameLinearVelocityBound
FrameLinearVelocityBound::FrameLinearVelocityBound(
  const Model &model, const std::string &frame, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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
    set_reference_frame(reference_frame);
}

void FrameLinearVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameLinearVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

void FrameLinearVelocityBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FrameLinearVelocityBound",
                            _model->floatingBase());
    _reference_frame = reference_frame;
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

std::vector<uint> FrameLinearVelocityBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FrameLinearVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    const bool local = _reference_frame.type() == ReferenceFrame::Local;
    return _reference_frame.axes(x).transpose() *
           _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, local)
             .head<3>();
}

void FrameLinearVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                        MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FrameLinearVelocityBound::add_gradient(const VecX &x, const VecX &u,
                                            MatXRef dX, MatXRef dU) const
{
    _model->frameVelocityDerivatives(x.head(_nq),
                                     x.tail(_nv),
                                     _frame_id,
                                     true,
                                     _dVdq_scratch,
                                     _dVdv_scratch);

    Mat3 Rt    = Mat3::Identity();
    Vec3 value = Vec3::Zero();
    if (_reference_frame.type() != ReferenceFrame::Local)
    {
        // The local velocity rotated by the frame's rotation, which turns
        // with the frame's angular Jacobian, the angular rows of dV/dv
        Mat3 R;
        std::tie(std::ignore, R) =
          _model->forwardKinematics(x.head(_nq), _frame_id);
        const Vec3 velocity =
          _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, true)
            .head<3>();

        for (Eigen::Index j = 0; j < _dVdq_scratch.cols(); ++j)
        {
            _dVdq_scratch.col(j).head<3>() +=
              _dVdv_scratch.col(j).tail<3>().cross(velocity);
        }
        Rt    = _reference_frame.axes(x).transpose() * R;
        value = Rt * velocity;
    }

    dX.leftCols(_nv).noalias()  += Rt * _dVdq_scratch.topRows<3>();
    dX.rightCols(_nv).noalias() += Rt * _dVdv_scratch.topRows<3>();

    // Base axes turn with the base, which adds the bound's value crossed with
    // the base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, 3) += skew(value);
    }
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

void FrameLinearVelocityBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                               const VecX &mu,
                                               MatXRef     Pxx) const
{
}

void FrameLinearVelocityBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                               const VecX &mu,
                                               MatXRef     Pxu) const
{
}

void FrameLinearVelocityBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                               const VecX &mu,
                                               MatXRef     Puu) const
{
}

// FrameAngularVelocityBound
FrameAngularVelocityBound::FrameAngularVelocityBound(
  const Model &model, const std::string &frame, const Vec3 &lb, const Vec3 &ub,
  const ReferenceFrame &reference_frame, const std::string &name)
  : StageConstraints(name)
  , _model(&model)
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
    set_reference_frame(reference_frame);
}

void FrameAngularVelocityBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void FrameAngularVelocityBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

void FrameAngularVelocityBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require({ ReferenceFrame::World,
                              ReferenceFrame::Fixed,
                              ReferenceFrame::Local,
                              ReferenceFrame::Base },
                            "FrameAngularVelocityBound",
                            _model->floatingBase());
    _reference_frame = reference_frame;
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

std::vector<uint> FrameAngularVelocityBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX FrameAngularVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    const bool local = _reference_frame.type() == ReferenceFrame::Local;
    return _reference_frame.axes(x).transpose() *
           _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, local)
             .tail<3>();
}

void FrameAngularVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                         MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void FrameAngularVelocityBound::add_gradient(const VecX &x, const VecX &u,
                                             MatXRef dX, MatXRef dU) const
{
    const bool local = _reference_frame.type() == ReferenceFrame::Local;
    _model->frameVelocityDerivatives(x.head(_nq),
                                     x.tail(_nv),
                                     _frame_id,
                                     local,
                                     _dVdq_scratch,
                                     _dVdv_scratch);

    const Mat3 Rt = _reference_frame.axes(x).transpose();

    dX.leftCols(_nv)  += Rt * _dVdq_scratch.bottomRows<3>();
    dX.rightCols(_nv) += Rt * _dVdv_scratch.bottomRows<3>();

    // Base axes turn with the base, which adds the bound's value crossed with
    // the base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, 3) += skew(
          Rt * _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, false)
                 .tail<3>());
    }
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

void FrameAngularVelocityBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Pxx) const
{
}

void FrameAngularVelocityBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Pxu) const
{
}

void FrameAngularVelocityBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                                const VecX &mu,
                                                MatXRef     Puu) const
{
}

// CenterOfMassBound
CenterOfMassBound::CenterOfMassBound(const Model &model, const Vec3 &lb,
                                     const Vec3           &ub,
                                     const ReferenceFrame &reference_frame,
                                     const Vec3           &center,
                                     const std::string    &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _ndx(model.solverMode() == SolverMode::Kinematics ? _nv : 2 * _nv)
  , _lb(lb)
  , _ub(ub)
  , _center(center)
{
    assert(model.xml() && "Cannot use CenterOfMassBound with a custom Model");
    set_reference_frame(reference_frame);
}

void CenterOfMassBound::set_lower_bound(const Vec3 &lb)
{
    _lb = lb;
}

void CenterOfMassBound::set_upper_bound(const Vec3 &ub)
{
    _ub = ub;
}

void CenterOfMassBound::set_reference_frame(
  const ReferenceFrame &reference_frame)
{
    reference_frame.require(
      { ReferenceFrame::World, ReferenceFrame::Fixed, ReferenceFrame::Base },
      "CenterOfMassBound",
      _model->floatingBase());
    _reference_frame = reference_frame;
}

void CenterOfMassBound::set_center(const Vec3 &center)
{
    _center = center;
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

std::vector<uint> CenterOfMassBound::state_only_rows() const
{
    return all_rows(constraint_diff_size());
}

VecX CenterOfMassBound::evaluate(const VecX &x, const VecX &u) const
{
    return _reference_frame.axes(x).transpose() *
           (_model->centerOfMass(x.head(_nq)) - _center);
}

void CenterOfMassBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                 MatX &dU) const
{
    dX.setZero(3, _ndx);
    dU.setZero(3, u.size());
    add_gradient(x, u, dX, dU);
}

void CenterOfMassBound::add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                                     MatXRef dU) const
{
    const Mat3 Rt = _reference_frame.axes(x).transpose();

    dX.leftCols(_nv) += Rt * _model->jacobianCenterOfMass(x.head(_nq));

    // Base axes turn with the base, which adds the bound's value crossed with
    // the base's angular tangent
    if (_reference_frame.type() == ReferenceFrame::Base)
    {
        dX.block<3, 3>(0, 3) +=
          skew(Rt * (_model->centerOfMass(x.head(_nq)) - _center));
    }
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

void CenterOfMassBound::add_hessprod_xx(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Pxx) const
{
}

void CenterOfMassBound::add_hessprod_xu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Pxu) const
{
}

void CenterOfMassBound::add_hessprod_uu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Puu) const
{
}

// ContactConstraint
ContactConstraint::ContactConstraint(const Model       &model,
                                     const std::string &frame, const double mu,
                                     const double max_normal_force,
                                     const double eq_tol, const bool no_slip,
                                     const std::string &name)
  : StageConstraints(name)
  , _model(&model)
  , _nq(model.nq())
  , _nv(model.nv())
  , _nu(model.nu())
  , _frame(frame)
  , _frame_id(model.frameId(frame))
  , _mu(mu)
  , _max_normal_force(max_normal_force)
  , _eq_tol(eq_tol)
  , _no_slip(no_slip)
  , _lb((Vec6() << 0.0, -std::numeric_limits<double>::infinity(), 0.0,
         -std::numeric_limits<double>::infinity(), 0.0, -eq_tol)
          .finished())
  , _ub((Vec6() << max_normal_force, 0.0,
         std::numeric_limits<double>::infinity(), 0.0,
         std::numeric_limits<double>::infinity(), eq_tol)
          .finished())
  , _jac_world_scratch(MatX::Zero(6, _nv))
  , _jac_local_scratch(MatX::Zero(6, _nv))
  , _dVdq_scratch(MatX::Zero(6, _nv))
  , _dVdv_scratch(MatX::Zero(6, _nv))
{
    assert(model.xml() && "Cannot use ContactConstraint with a custom Model");
    assert(model.hasFrame(frame) && "Invalid frame for given model");
    assert(max_normal_force >= 0.0 &&
           "Max normal force value must be positive");

    // No-slip rows bound the frame's linear velocity to zero
    if (_no_slip)
    {
        _lb.conservativeResize(9);
        _ub.conservativeResize(9);
        _lb.tail<3>().setZero();
        _ub.tail<3>().setZero();
    }
}

void ContactConstraint::set_mu(const double mu)
{
    _mu = mu;
}

void ContactConstraint::set_max_normal_force(const double max_normal_force)
{
    _max_normal_force = max_normal_force;
    _ub[0]            = max_normal_force;
}

void ContactConstraint::set_eq_tol(const double eq_tol)
{
    _eq_tol = eq_tol;
    _lb[5]  = -eq_tol;
    _ub[5]  = eq_tol;
}

void ContactConstraint::set_terrain(
  const std::shared_ptr<terrain::Terrain> &terrain)
{
    _terrain = terrain;
}

std::shared_ptr<terrain::Terrain> ContactConstraint::contact_terrain() const
{
    return _terrain ? _terrain : _model->terrain();
}

void ContactConstraint::set_enabled(const bool enabled)
{
    StageConstraints::set_enabled(enabled);
    _force->ignore = !enabled;
}

uint ContactConstraint::output_size() const
{
    return _no_slip ? 9 : 6;
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

std::vector<uint> ContactConstraint::state_only_rows() const
{
    // The height and no-slip rows, since the others bound the contact force
    if (_no_slip)
    {
        return { 5, 6, 7, 8 };
    }

    return { 5 };
}

VecX ContactConstraint::evaluate(const VecX &x, const VecX &u) const
{
    std::shared_ptr<terrain::Terrain> terrain = contact_terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    Vec3 f_global = R * u.segment(_nu + 3 * _force->force_index, 3);

    double fn  = (f_global.transpose() * tp.normal)[0];
    double ft1 = (f_global.transpose() * tp.t1)[0];
    double ft2 = (f_global.transpose() * tp.t2)[0];

    VecX res(output_size());
    res[0] = fn;
    res[1] = ft1 - _mu * fn;
    res[2] = ft1 + _mu * fn;
    res[3] = ft2 - _mu * fn;
    res[4] = ft2 + _mu * fn;
    res[5] = pos[2] - tp.height;

    if (_no_slip)
    {
        res.tail<3>() =
          _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame_id, true)
            .head<3>();
    }

    return res;
}

void ContactConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                 MatX &dU) const
{
    dX.setZero(output_size(), 2 * _nv);
    dU.setZero(output_size(), u.size());
    add_gradient(x, u, dX, dU);
}

void ContactConstraint::add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                                     MatXRef dU) const
{
    std::shared_ptr<terrain::Terrain> terrain = contact_terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame_id);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    _model->frameJacobian(x.head(_nq), _frame_id, false, _jac_world_scratch);
    _model->frameJacobian(x.head(_nq), _frame_id, true, _jac_local_scratch);
    const auto Jfw = _jac_world_scratch.topRows<3>();
    const auto Jfl = _jac_local_scratch.bottomRows<3>();

    uint total_force_index = _nu + 3 * _force->force_index;

    Vec3 f_local = u.segment(total_force_index, 3);

    Mat3 f_skew;
    f_skew << 0, -f_local[2], f_local[1], f_local[2], 0, -f_local[0],
      -f_local[1], f_local[0], 0;

    MatX &dfglobal_dq     = _dforce_scratch;
    dfglobal_dq.noalias() = -R * f_skew * Jfl;

    Vec3 dhdp = (Vec3() << -tp.dhdx, -tp.dhdy, 1).finished();

    _dforce_components_scratch.resize(_nv, 3);
    auto dfn_dq       = _dforce_components_scratch.col(0);
    auto dft1_dq      = _dforce_components_scratch.col(1);
    auto dft2_dq      = _dforce_components_scratch.col(2);
    dfn_dq.noalias()  = dfglobal_dq.transpose() * tp.normal;
    dft1_dq.noalias() = dfglobal_dq.transpose() * tp.t1;
    dft2_dq.noalias() = dfglobal_dq.transpose() * tp.t2;

    // Terrain frame derivatives with respect to the frame's x and y position
    // The frame is built from the height gradient (see `TerrainPoint`)
    const double a = tp.dhdx, b = tp.dhdy;
    const double s = std::sqrt(1 + a * a + b * b), r = std::sqrt(1 + a * a);

    Vec3 dn_da  = -Vec3::UnitX() / s - tp.normal * a / (s * s);
    Vec3 dn_db  = -Vec3::UnitY() / s - tp.normal * b / (s * s);
    Vec3 dt1_da = Vec3::UnitZ() / r - tp.t1 * a / (r * r);
    Vec3 dt2_da = dn_da.cross(tp.t1) + tp.normal.cross(dt1_da);
    Vec3 dt2_db = dn_db.cross(tp.t1);

    Vec3 dn_dx  = dn_da * tp.d2hdx2 + dn_db * tp.d2hdxdy;
    Vec3 dn_dy  = dn_da * tp.d2hdxdy + dn_db * tp.d2hdy2;
    Vec3 dt1_dx = dt1_da * tp.d2hdx2;
    Vec3 dt1_dy = dt1_da * tp.d2hdxdy;
    Vec3 dt2_dx = dt2_da * tp.d2hdx2 + dt2_db * tp.d2hdxdy;
    Vec3 dt2_dy = dt2_da * tp.d2hdxdy + dt2_db * tp.d2hdy2;

    Vec3 f_global = R * f_local;

    dfn_dq +=
      (f_global.dot(dn_dx) * Jfw.row(0) + f_global.dot(dn_dy) * Jfw.row(1))
        .transpose();
    dft1_dq +=
      (f_global.dot(dt1_dx) * Jfw.row(0) + f_global.dot(dt1_dy) * Jfw.row(1))
        .transpose();
    dft2_dq +=
      (f_global.dot(dt2_dx) * Jfw.row(0) + f_global.dot(dt2_dy) * Jfw.row(1))
        .transpose();

    dX.row(0).head(_nv) += dfn_dq.transpose();
    dX.row(1).head(_nv) += (dft1_dq - _mu * dfn_dq).transpose();
    dX.row(2).head(_nv) += (dft1_dq + _mu * dfn_dq).transpose();
    dX.row(3).head(_nv) += (dft2_dq - _mu * dfn_dq).transpose();
    dX.row(4).head(_nv) += (dft2_dq + _mu * dfn_dq).transpose();

    dX.row(5).head(_nv).noalias() += dhdp.transpose() * Jfw;

    if (_no_slip)
    {
        _model->frameVelocityDerivatives(x.head(_nq),
                                         x.tail(_nv),
                                         _frame_id,
                                         true,
                                         _dVdq_scratch,
                                         _dVdv_scratch);

        dX.bottomLeftCorner(3, _nv)  += _dVdq_scratch.topRows<3>();
        dX.bottomRightCorner(3, _nv) += _dVdv_scratch.topRows<3>();
    }

    double fn  = (f_global.transpose() * tp.normal)[0];
    double ft1 = (f_global.transpose() * tp.t1)[0];
    double ft2 = (f_global.transpose() * tp.t2)[0];

    Vec3 dfn_dflocal  = R.transpose() * tp.normal;
    Vec3 dft1_dflocal = R.transpose() * tp.t1;
    Vec3 dft2_dflocal = R.transpose() * tp.t2;

    dU.block<1, 3>(0, total_force_index) += dfn_dflocal.transpose();
    dU.block<1, 3>(1, total_force_index) +=
      (dft1_dflocal - _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(2, total_force_index) +=
      (dft1_dflocal + _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(3, total_force_index) +=
      (dft2_dflocal - _mu * dfn_dflocal).transpose();
    dU.block<1, 3>(4, total_force_index) +=
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

void ContactConstraint::add_hessprod_xx(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Pxx) const
{
}

void ContactConstraint::add_hessprod_xu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Pxu) const
{
}

void ContactConstraint::add_hessprod_uu(const VecX &x, const VecX &u,
                                        const VecX &mu, MatXRef Puu) const
{
}

}
}
}
}
}