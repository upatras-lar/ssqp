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
FramePositionBound::FramePositionBound(const Model &model, const VecX &lb,
                                       const VecX &ub, const uint nq,
                                       const uint nv)
  : _model(std::make_shared<Model>(model))
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
    VecX pos;
    std::tie(pos, std::ignore) =
      _model->forwardKinematics(x.head(_nq), "panda_ee");

    return (Eigen::Vector3d() << pos(0), pos(1), pos(2)).finished();
}

void FramePositionBound::gradient(const VecX &x, const VecX &u, MatX &dX,
                                  MatX &dU) const
{
    Eigen::VectorXd pos;
    Eigen::MatrixXd rot;
    std::tie(pos, rot)  = _model->forwardKinematics(x.head(_nq), "panda_ee");
    Eigen::MatrixXd Jfk = _model->frameJacobian(x.head(_nq), "panda_ee");

    dX = Eigen::MatrixXd::Zero(3, x.size());
    // TODO: should we be taking the top rows?
    // nq or nv?
    dX.leftCols(_nq) = Jfk.topRows(3);

    dU = Eigen::MatrixXd::Zero(3, _nv);
}

void FramePositionBound::hessprod_xx(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxx) const
{
    ZeroHessprod_xx(x, u, mu, Pxx);
}
void FramePositionBound::hessprod_xu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Pxu) const
{
    ZeroHessprod_xu(x, u, mu, Pxu);
}
void FramePositionBound::hessprod_uu(const VecX &x, const VecX &u,
                                     const VecX &mu, MatX &Puu) const
{
    ZeroHessprod_uu(x, u, mu, Puu);
}

}
}
}
}
}