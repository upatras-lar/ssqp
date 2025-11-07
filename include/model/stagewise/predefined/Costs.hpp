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

/* Predefined squared control cost
 * Defines the cost term $l(\boldsymbol{x}, \boldsymbol{u}) = w_u \boldsymbol{u}^2$
 * where $w_u$ is the weight on the control term
 */
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

/* Predefined squared velocity cost
 * Defines the cost term $l(\boldsymbol{x}, \boldsymbol{u}) = w_v \boldsymbol{x}_{vel}^2$
 * where:
 *  - $\boldsymbol{x}_{vel}$ is the velocity part of the state
 *  - $w_v$ is the weight on the velocity term
 */
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

/* Predefined squared frame position error cost
 * Defines the cost term $l(\boldsymbol{x}, \boldsymbol{u}) = w_p (\boldsymbol{p}_f - \boldsymbol{p}_d)^2$
 * where:
 *  - $\boldsymbol{p}_f$ is the frame position for a given frame of the model (in $\mathcal{R}^3$)
 *  - $\boldsymbol{p}_d$ is the desired position for that frame (in $\mathcal{R}^3$)
 *  - $w_p$ is the weight on the position error
 */
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

/* Predefined squared frame rotation error cost
 * Defines the cost term $l(\boldsymbol{x}, \boldsymbol{u}) = w_r \log^2_{\mathit{SO3}}R_dR$
 * where:
 *  - $R$ is the frame rotation for a given frame of the model (in $\mathit{SO3}$)
 *  - $R_d$ is the desired rotation for that frame (in $\mathit{SO3}$)
 *  - $w_r$ is the weight on the rotation error
 */
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

/* Predefined squared frame pose error cost
 * Defines the cost term $l(\boldsymbol{x}, \boldsymbol{u}) = W_t \log^2_{\mathit{SE3}}T_dT$
 * where:
 *  - $T$ is the frame pose for a given frame of the model (in $\mathit{SE3}$)
 *  - $T_d$ is the desired pose for that frame (in $\mathit{SE3}$)
 *  - $W_t$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
 *    for rotation
 *  TODO: doesn't work
 */
class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(const Model       &model,
                              const VecX        &desired_translation,
                              const MatX        &desired_rotation,
                              const std::string &frame, const uint nq,
                              const uint nv, const double position_weight,
                              const double rotation_weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    MatX                         _desired_pose_inv;
    const std::string            _frame;

    const uint _nq, _nv;
    Mat6       _weight;
};
}
}
}
}
}

#endif