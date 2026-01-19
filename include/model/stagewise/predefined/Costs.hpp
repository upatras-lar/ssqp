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
/* Predefined squared joint angle cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_a \boldsymbol{x}_{ang}^2$
 * where:
 *  - $\boldsymbol{x}_{ang}$ is the angular (position) part of the state
 *  - $w_a$ is the weight on the angular term
 */
class SquaredJointAngleCost : public StageCost
{
  public:
    SquaredJointAngleCost(const uint nq, const uint nv, const double weight,
                          const bool free_flyer = false);

    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nq, _nv;
    double     _weight;
    const bool _free_flyer;
};

/* Predefined squared joint velocity cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_v \boldsymbol{x}_{vel}^2$
 * where:
 *  - $\boldsymbol{x}_{vel}$ is the velocity part of the state
 *  - $w_v$ is the weight on the velocity term
 */
class SquaredJointVelocityCost : public StageCost
{
  public:
    SquaredJointVelocityCost(const uint nv, const double weight,
                             const bool free_flyer = false);

    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv, _njoints;
    double     _weight;
};

/* Predefined squared control cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_u \boldsymbol{u}^2$
 * where $w_u$ is the weight on the control term
 */
class SquaredControlCost : public StageCost
{
  public:
    SquaredControlCost(const uint nv, const double weight);

    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv;
    double     _weight;
};

/* Predefined squared force cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u})=w_f \boldsymbol{u}_f$
 * where $\boldsymbol{u}_f$ is the force part of the controls
 * and $w_f$ is the weight on the term
 */
class SquaredForceCost : public StageCost
{
  public:
    SquaredForceCost(const uint nv, const double weight);

    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv;
    double     _weight;
};

/* Predefined squared configuration difference cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_q (\boldsymbol{q} - \boldsymbol{q}_d)^2$
 * where:
 *  - $\boldsymbol{q}$ is the configuration part of the state
 *  - $\boldsymbol{q}_d$ is the desired configuration
 *  - $w_q$ is the weight for the difference
 */
class SquaredConfigurationDifferenceCost : public StageCost
{
  public:
    SquaredConfigurationDifferenceCost(const VecX &desired_configuration,
                                       const uint nq, const uint nv,
                                       const double weight,
                                       const bool   free_flyer = false);

    void set_desired_configuration(const VecX &desired_configuration);
    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    VecX       _desired_configuration;
    const uint _nq, _nv;
    double     _weight;
    const bool _free_flyer;
};

/* Predefined squared frame translation error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_p (\boldsymbol{p}_f - \boldsymbol{p}_d)^2$
 * where:
 *  - $\boldsymbol{p}_f$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
 *  - $\boldsymbol{p}_d$ is the desired translation for that frame (in $\mathcal{R}^3$)
 *  - $w_p$ is the weight on the translation error
 */
class FrameSquaredTranslationErrorCost : public StageCost
{
  public:
    FrameSquaredTranslationErrorCost(const Model       &model,
                                     const Vec3        &desired_translation,
                                     const std::string &frame, const uint nq,
                                     const uint nv, const double weight);

    void set_desired_translation(const Vec3 &desired_translation);
    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    Vec3                         _desired_translation;
    const std::string            _frame;

    const uint _nq, _nv;
    double     _weight;
};

/* Predefined squared frame rotation error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = w_r \log^2_{\mathit{SO3}}R_dR$
 * where:
 *  - $R$ is the frame rotation for a given frame of the model (in $\mathit{SO3}$)
 *  - $R_d$ is the desired rotation for that frame (in $\mathit{SO3}$)
 *  - $w_r$ is the weight on the rotation error
 */
class FrameSquaredRotationErrorCost : public StageCost
{
  public:
    FrameSquaredRotationErrorCost(const Model       &model,
                                  const Mat3        &desired_rotation,
                                  const std::string &frame, const uint nq,
                                  const uint nv, const double weight);

    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    Mat3                         _desired_rotation;
    const std::string            _frame;

    const uint _nq, _nv;
    double     _weight;
};

/* Predefined squared frame pose error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = W_t \log^2_{\mathit{SE3}}T_dT$
 * where:
 *  - $T$ is the frame pose for a given frame of the model (in $\mathit{SE3}$)
 *  - $T_d$ is the desired pose for that frame (in $\mathit{SE3}$)
 *  - $W_t$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
 *    for rotation
 */
class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(const Model &model, const Mat4 &desired_pose,
                              const std::string &frame, const uint nq,
                              const uint nv, const double translation_weight,
                              const double rotation_weight);

    FrameSquaredPoseErrorCost(const Model       &model,
                              const Vec3        &desired_translation,
                              const Mat3        &desired_rotation,
                              const std::string &frame, const uint nq,
                              const uint nv, const double translation_weight,
                              const double rotation_weight);

    void set_desired_pose(const Mat4 &desired_pose);
    void set_desired_translation(const Vec3 &desired_translation);
    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_translation_weight(const double translation_weight);
    void set_rotation_weight(const double rotation_weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    Mat4                         _desired_pose;
    const std::string            _frame;

    const uint _nq, _nv;
    Mat6       _weight;
};

class FrameSquaredVelocityCost : public StageCost
{
  public:
    FrameSquaredVelocityCost(const Model &model, const std::string &frame,
                             const uint nq, const uint nv, const double weight);

    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const std::string            _frame;

    const uint _nq, _nv;
    double     _weight;
};
}
}
}
}
}

#endif