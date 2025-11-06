#ifndef SSQP_MODEL_STAGEWISE_PREDEFINED_COSTS_HPP
#define SSQP_MODEL_STAGEWISE_PREDEFINED_COSTS_HPP

#include <memory>

#include "types.hpp"
#include "pinocchio_wrapper/PinocchioWrapper.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageCost.hpp"

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
class SquaredControlCost : public StageCost
{
  public:
    SquaredControlCost(const uint nv, const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint   _nv;
    const double _weight;
};

class SquaredVelocityCost : public StageCost
{
  public:
    SquaredVelocityCost(const uint nv, const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint   _nv;
    const double _weight;
};

class FrameSquaredPositionErrorCost : public StageCost
{
  public:
    FrameSquaredPositionErrorCost(const Model       &model,
                                  const VecX        &desired_translation,
                                  const std::string &frame, const uint nq,
                                  const uint nv, const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const VecX                   _desired_translation;
    const std::string            _frame;

    const uint   _nq, _nv;
    const double _weight;
};

class FrameSquaredRotationErrorCost : public StageCost
{
  public:
    FrameSquaredRotationErrorCost(const Model       &model,
                                  const MatX        &desired_rotation,
                                  const std::string &frame, const uint nq,
                                  const uint nv, const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const MatX                   _desired_rotation;
    const std::string            _frame;

    const uint   _nq, _nv;
    const double _weight;
};

class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(const Model       &model,
                              const VecX        &desired_translation,
                              const MatX        &desired_rotation,
                              const std::string &frame, const uint nq,
                              const uint nv, const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    MatX                         _desired_pose, _desired_pose_inv;
    const std::string            _frame;

    const uint   _nq, _nv;
    const double _weight;
};
}
}
}
}
}

#endif