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
class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(
      const Model &model, const VecX &desired_translation,
      const MatX &desired_rotation, const std::string &frame, const uint nq,
      const uint nv, const double position_weight, const double rotation_weight,
      const double velocity_weight, const double control_weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const VecX                   _desired_translation;
    const MatX                   _desired_rotation;
    const std::string            _frame;

    const uint   _nq, _nv;
    const double _P_WGT, _R_WGT, _V_WGT, _U_WGT;
};

}
}
}
}
}

#endif