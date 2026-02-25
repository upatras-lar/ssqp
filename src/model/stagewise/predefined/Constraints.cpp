#include "model/stagewise/predefined/Constraints.hpp"

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
JointAngleConstraint::JointAngleConstraint(const VecX &lb, const VecX &ub,
                                           const uint nq, const uint nv,
                                           const bool free_flyer)
  : _lb(lb)
  , _ub(ub)
  , _nv(nv)
  , _njoints(free_flyer ? nq - 7 : nq)
  , _free_flyer(free_flyer)
{
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

VecX JointAngleConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return _free_flyer ? x.segment(7, _njoints) : x.head(_njoints);
}

void JointAngleConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                    MatX &dU) const
{
    dX = MatX::Zero(_njoints, 2 * _nv);
    if (_free_flyer)
    {
        dX.block(0, 6, _njoints, _njoints).setIdentity();
    }
    else
    {
        dX.topLeftCorner(_njoints, _njoints).setIdentity();
    }

    dU = MatX::Zero(_njoints, u.size());
}

void JointAngleConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void JointAngleConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void JointAngleConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                       const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

// JointVelocityConstraint
JointVelocityConstraint::JointVelocityConstraint(const VecX &lb, const VecX &ub,
                                                 const uint nv,
                                                 const bool free_flyer)
  : _lb(lb)
  , _ub(ub)
  , _nv(nv)
  , _njoints(free_flyer ? nv - 6 : nv)
{
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

VecX JointVelocityConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return x.tail(_njoints);
}

void JointVelocityConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                       MatX &dU) const
{
    dX = MatX::Zero(_njoints, 2 * _nv);
    dX.bottomRightCorner(_njoints, _njoints).setIdentity();

    dU = MatX::Zero(_njoints, u.size());
}

void JointVelocityConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void JointVelocityConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void JointVelocityConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                          const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

// JointControlConstraint
JointControlConstraint::JointControlConstraint(const VecX &lb, const VecX &ub,
                                               const uint nv,
                                               const bool free_flyer)
  : _lb(lb)
  , _ub(ub)
  , _nv(nv)
  , _njoints(free_flyer ? nv - 6 : nv)
  , _free_flyer(free_flyer)
{
    assert(lb.size() == ub.size() && lb.size() == _njoints &&
           "Invalid bound size");
}

void JointControlConstraint::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == _njoints && "Invalid bound size");
    _lb = lb;
}

void JointControlConstraint::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == _njoints && "Invalid bound size");
    _ub = ub;
}

uint JointControlConstraint::output_size() const
{
    return _njoints;
}

VecX JointControlConstraint::evaluate(const VecX &x, const VecX &u) const
{
    return _free_flyer ? u.segment(6, _njoints) : u.head(_njoints);
}

void JointControlConstraint::gradient(const VecX &x, const VecX &u, MatX &dX,
                                      MatX &dU) const
{
    dX = MatX::Zero(_njoints, 2 * _nv);
    dU = MatX::Zero(_njoints, u.size());
    if (_free_flyer)
    {
        dU.block(0, 6, _njoints, _njoints).setIdentity();
    }
    else
    {
        dU.topLeftCorner(_njoints, _njoints).setIdentity();
    }
}

void JointControlConstraint::hessprod_xx(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void JointControlConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void JointControlConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                         const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

// FramePositionBound
FramePositionBound::FramePositionBound(const Model       &model,
                                       const std::string &frame, const VecX &lb,
                                       const VecX &ub, const uint nq,
                                       const uint nv)
  : _model(std::make_shared<Model>(model))
  , _frame(frame)
  , _lb(lb)
  , _ub(ub)
  , _nq(nq)
  , _nv(nv)
{
    assert(lb.size() == 3 && ub.size() == 3 && "Bound vector size must be 3");
}

void FramePositionBound::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == 3 && "Bound vector size must be 3");
    _lb = lb;
}

void FramePositionBound::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == 3 && "Bound vector size must be 3");
    _ub = ub;
}

uint FramePositionBound::output_size() const
{
    return 3;
}

VecX FramePositionBound::lower_bound() const
{
    return _lb;
}
VecX FramePositionBound::upper_bound() const
{
    return _ub;
}

VecX FramePositionBound::evaluate(const VecX &x, const VecX &u) const
{
    Vec3 pos;
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);
    return pos;
}

void FramePositionBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                  MatX &dU) const
{
    Vec3 pos;
    Mat3 rot;
    std::tie(pos, rot) = _model->forwardKinematics(x.head(_nq), _frame);
    MatX Jfk           = _model->frameJacobian(x.head(_nq), _frame);

    dX               = MatX::Zero(3, 2 * _nv);
    dX.leftCols(_nv) = Jfk.topRows(3);

    dU = MatX::Zero(3, u.size());
}

void FramePositionBound::hessprod_xx(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void FramePositionBound::hessprod_xu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void FramePositionBound::hessprod_uu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

// FrameLinearVelocityBound
FrameLinearVelocityBound::FrameLinearVelocityBound(
  const Model &model, const std::string &frame, const VecX &lb, const VecX &ub,
  const uint nq, const uint nv, const bool terminal)
  : _model(std::make_shared<Model>(model))
  , _frame(frame)
  , _lb(lb)
  , _ub(ub)
  , _nq(nq)
  , _nv(nv)
  , _terminal(terminal)
{
    assert(lb.size() == 3 && ub.size() == 3 && "Bound vector size must be 3");
}

void FrameLinearVelocityBound::set_lower_bound(const VecX &lb)
{
    assert(lb.size() == 3 && "Bound vector size must be 3");
    _lb = lb;
}

void FrameLinearVelocityBound::set_upper_bound(const VecX &ub)
{
    assert(ub.size() == 3 && "Bound vector size must be 3");
    _ub = ub;
}

uint FrameLinearVelocityBound::output_size() const
{
    return 3;
}

VecX FrameLinearVelocityBound::lower_bound() const
{
    return _lb;
}
VecX FrameLinearVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FrameLinearVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    return _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true)
      .head(3);
}

void FrameLinearVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                        MatX &dU) const
{
    VecX a;

    if (_terminal)
    {
        a = VecX::Zero(_nv);
    }
    else
    {
        a = u.head(_nv);
    }

    MatX dVdq, dVdv;
    std::tie(dVdq, dVdv) = _model->frameVelocityDerivatives(x.head(_nq),
                                                            x.tail(_nv),
                                                            a,
                                                            _frame,
                                                            true);

    dX                = MatX::Zero(3, 2 * _nv);
    dX.leftCols(_nv)  = dVdq.topRows<3>();
    dX.rightCols(_nv) = dVdv.topRows<3>();

    dU = MatX::Zero(3, u.size());
}

void FrameLinearVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void FrameLinearVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void FrameLinearVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
                                           const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

// ContactConstraint
ContactConstraint::ContactConstraint(const Model       &model,
                                     const std::string &frame, const uint nq,
                                     const uint nv, const double mu)
  : _model(std::make_shared<Model>(model))
  , _frame(frame)
  , _nq(nq)
  , _nv(nv)
  , _mu(mu)
{
}

void ContactConstraint::set_mu(const double mu)
{
    _mu = mu;
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

void ContactConstraint::set_forces(const std::vector<Force> &forces)
{
    _forces = forces;
}

VecX ContactConstraint::lower_bound() const
{
    return (Vec6() << 0.0,
            -std::numeric_limits<double>::infinity(),
            0.0,
            -std::numeric_limits<double>::infinity(),
            0.0,
            0.0)
      .finished();
}

VecX ContactConstraint::upper_bound() const
{
    return (Vec6() << std::numeric_limits<double>::infinity(),
            0.0,
            std::numeric_limits<double>::infinity(),
            0.0,
            std::numeric_limits<double>::infinity(),
            0.0)
      .finished();
}

VecX ContactConstraint::evaluate(const VecX &x, const VecX &u) const
{
    static std::shared_ptr<terrain::Terrain> terrain = _model->terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    Vec3 f_global = R * u.segment(_nv + 3 * _forces[0].force_index, 3);

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
    static std::shared_ptr<terrain::Terrain> terrain = _model->terrain();

    Vec3 pos;
    Mat3 R;
    std::tie(pos, R) = _model->forwardKinematics(x.head(_nq), _frame);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    dX       = MatX::Zero(6, 2 * _nv);
    MatX Jfw = _model->frameJacobian(x.head(_nq), _frame).topRows<3>();
    MatX Jfl = _model->frameJacobian(x.head(_nq), _frame, true).bottomRows<3>();

    uint total_force_index = _nv + 3 * _forces[0].force_index;

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

    dU = MatX::Zero(6, u.size());

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
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}

void ContactConstraint::hessprod_xu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void ContactConstraint::hessprod_uu(const VecX &x, const VecX &u,
                                    const VecX &mu, MatX &Puu) const
{
    Puu = MatX::Zero(u.size(), u.size());
}

}
}
}
}
}