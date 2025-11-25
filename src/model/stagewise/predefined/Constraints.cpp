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

// FrameVelocityBound
FrameVelocityBound::FrameVelocityBound(const Model       &model,
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

uint FrameVelocityBound::output_size() const
{
    return 3;
}

VecX FrameVelocityBound::lower_bound() const
{
    return _lb;
}
VecX FrameVelocityBound::upper_bound() const
{
    return _ub;
}

VecX FrameVelocityBound::evaluate(const VecX &x, const VecX &u) const
{
    return _model->frameVelocity(x.head(_nq), x.tail(_nv), _frame, true);
}

void FrameVelocityBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                  MatX &dU) const
{
    MatX dVdq, dVdv;
    std::tie(dVdq, dVdv) = _model->frameVelocityDerivatives(x, u, _frame, true);

    dX                = MatX::Zero(3, 2 * _nv);
    dX.leftCols(_nv)  = dVdq.topRows<3>();
    dX.rightCols(_nv) = dVdv.topRows<3>();

    dU = MatX::Zero(3, u.size());
}

void FrameVelocityBound::hessprod_xx(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxx) const
{
    Pxx = MatX::Zero(2 * _nv, 2 * _nv);
}
void FrameVelocityBound::hessprod_xu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxu) const
{
    Pxu = MatX::Zero(2 * _nv, u.size());
}
void FrameVelocityBound::hessprod_uu(const VecX &x, const VecX &u,
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

    Vec3 f_local = u.segment(_nv + 3 * _forces[0].force_index, 3);

    double fn  = f_local[2];
    double ft1 = f_local[0];
    double ft2 = f_local[1];

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
    std::tie(pos, std::ignore) = _model->forwardKinematics(x.head(_nq), _frame);

    terrain::Terrain::TerrainPoint tp = terrain->at(pos[0], pos[1]);

    // TODO: Assuming terrain basis is constant during linearization
    dX       = MatX::Zero(6, 2 * _nv);
    MatX Jfk = _model->frameJacobian(x.head(_nq), _frame).topRows<3>();

    Vec3 dhdp = (Vec3() << -tp.dhdx, -tp.dhdy, 1).finished();

    dX.row(5).head(_nv) = dhdp.transpose() * Jfk;

    uint total_force_index = _nv + 3 * _forces[0].force_index;

    Vec3 f_local = u.segment(total_force_index, 3);

    dU                                   = MatX::Zero(6, u.size());
    dU.block<1, 3>(0, total_force_index) = tp.normal.transpose();
    dU.block<1, 3>(1, total_force_index) =
      (tp.t1 - _mu * tp.normal).transpose();
    dU.block<1, 3>(2, total_force_index) =
      (tp.t1 + _mu * tp.normal).transpose();
    dU.block<1, 3>(3, total_force_index) =
      (tp.t2 - _mu * tp.normal).transpose();
    dU.block<1, 3>(4, total_force_index) =
      (tp.t2 + _mu * tp.normal).transpose();
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