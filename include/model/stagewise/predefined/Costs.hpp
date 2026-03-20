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
// Note: The $\odot$ symbol below denoted the element-wise product

/* Predefined squared joint angle cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{ang} \rVert_2^2$
 * where:
 *  - $\boldsymbol{x}_{ang}$ is the angular (position) part of the state
 *    Note that this will ignore the first 7 elements if the model has a 
 *    floating base
 *  - $\boldsymbol{w}$ is the weight on the angular term
 */
class SquaredJointAngleCost : public StageCost
{
  public:
    SquaredJointAngleCost(const uint nq, const uint nv, const double weight,
                          const bool free_flyer = false);
    SquaredJointAngleCost(const uint nq, const uint nv, const VecX &weight,
                          const bool free_flyer = false);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nq, _nv;
    VecX       _weight;
    const bool _free_flyer;
};

/* Predefined squared joint velocity cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{vel} \rVert_2^2$
 * where:
 *  - $\boldsymbol{x}_{vel}$ is the velocity part of the state
 *  - $\boldsymbol{w}$ is the weight on the velocity term
 */
class SquaredJointVelocityCost : public StageCost
{
  public:
    SquaredJointVelocityCost(const uint nv, const double weight,
                             const bool free_flyer = false);
    SquaredJointVelocityCost(const uint nv, const VecX &weight,
                             const bool free_flyer = false);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv, _njoints;
    VecX       _weight;
};

/* Predefined squared control cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{ctrl} \rVert_2^2$
 * where:
 *  - $\boldsymbol{u}_{ctrl}$ is the control part of the controls
 *  - $\boldsymbol{w}$ is the weight on the control term
 */
class SquaredControlCost : public StageCost
{
  public:
    SquaredControlCost(const uint nv, const uint nu, const double weight);
    SquaredControlCost(const uint nv, const uint nu, const VecX &weight);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv, _nu;
    VecX       _weight;
};

/* Predefined squared inverse dynamics torque cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{\tau} \rVert_2^2$
 * Note that this cost assumes inverse dynamics is being used (i.e. treats
 * controls as accelerations)
 * where:
 *  - $\boldsymbol{\tau}$ is the torque caused by the controls
 *  - $\boldsymbol{w}$ is the weight on the torque term
 */
// TODO: Take into account any joint forces
// TODO: Potential bug
// TODO: Fix for ReducedInverse
class SquaredIDTorqueCost : public StageCost
{
  public:
    SquaredIDTorqueCost(const Model &model, const uint nq, const uint nv,
                        const uint nu, const double weight);
    SquaredIDTorqueCost(const Model &model, const uint nq, const uint nv,
                        const uint nu, const VecX &weight);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _nu;
    VecX                         _weight;
};

/* Predefined squared force cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{f} \rVert_2^2$
 * where:
 *  - $\boldsymbol{u}_{f}$ is the force part of the controls
 *  - $\boldsymbol{w}$ is the weight on the force term
 */
class SquaredForceCost : public StageCost
{
  public:
    SquaredForceCost(const uint nv, const uint nu, const double weight);
    SquaredForceCost(const uint nv, const uint nu, const VecX &weight);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv, _nu;
    double     _weight;
    VecX       _weight_vec;
    bool       _single_weight;
};

/* Predefined squared configuration error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{q} - \boldsymbol{q}_d)\rVert_2^2$
 * where:
 *  - $\boldsymbol{q}$ is the configuration part of the state
 *  - $\boldsymbol{q}_d$ is the desired configuration
 *  - $\boldsymbol{w}$ is the weight for the configuration error
 */
class SquaredConfigurationErrorCost : public StageCost
{
  public:
    SquaredConfigurationErrorCost(const VecX &desired_configuration,
                                  const uint nq, const uint nv,
                                  const double weight,
                                  const bool   free_flyer = false);
    SquaredConfigurationErrorCost(const VecX &desired_configuration,
                                  const uint nq, const uint nv,
                                  const VecX &weight,
                                  const bool  free_flyer = false);

    void set_desired_configuration(const VecX &desired_configuration);
    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    VecX       _desired_configuration;
    const uint _nq, _nv;
    VecX       _weight;
    const bool _free_flyer;
};

/* Predefined squared frame translation error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{t} - \boldsymbol{t}_d)\rVert_2^2$
 * where:
 *  - $\boldsymbol{t}$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
 *  - $\boldsymbol{t}_d$ is the desired translation for that frame (in $\mathcal{R}^3$)
 *  - $\boldsymbol{w}$ is the weight on the translation error
 */
class FrameSquaredTranslationErrorCost : public StageCost
{
  public:
    FrameSquaredTranslationErrorCost(const Model       &model,
                                     const std::string &frame,
                                     const Vec3        &desired_translation,
                                     const uint nq, const uint nv,
                                     const double weight);
    FrameSquaredTranslationErrorCost(const Model       &model,
                                     const std::string &frame,
                                     const Vec3        &desired_translation,
                                     const uint nq, const uint nv,
                                     const Vec3 &weight);

    void set_desired_translation(const Vec3 &desired_translation);
    void set_weight(const double weight);
    void set_weight(const Vec3 &weight);

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
    Vec3       _weight;
};

/* Predefined squared frame rotation error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert w \log_{SO(3)}R^\top R_d\rVert_2^2$
 * where:
 *  - $R$ is the frame rotation for a given frame of the model (in $\mathit{SO3}$)
 *  - $R_d$ is the desired rotation for that frame (in $\mathit{SO3}$)
 *  - $\boldsymbol{w}$ is the weight on the rotation error
 */
class FrameSquaredRotationErrorCost : public StageCost
{
  public:
    FrameSquaredRotationErrorCost(const Model &model, const std::string &frame,
                                  const Mat3 &desired_rotation, const uint nq,
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
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} D^\top W D$
 * where:
 *  - $D = \log_{SE(3)} T^{-1}T_d$
 *  - $T$ is the frame pose for a given frame of the model (in $\mathit{SE3}$)
 *  - $T_d$ is the desired pose for that frame (in $\mathit{SE3}$)
 *  - $W$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
 *    for rotation
 */
class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Mat4 &desired_pose, const uint nq,
                              const uint nv, const double translation_weight,
                              const double rotation_weight);

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Vec3 &desired_translation,
                              const Mat3 &desired_rotation, const uint nq,
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

/* Predefined squared frame velocity error cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{v}_f - \boldsymbol{v}_d)\rVert_2^2$
 * where:
 *  - $\boldsymbol{v}_f$ is the frame velocity for a given frame of the model (in $\mathcal{R}^6$)
 *  - $\boldsymbol{v}_d$ is the desired velocity for that frame (in $\mathcal{R}^6$)
 *  - $\boldsymbol{w}$ is the weight on the velocity error
 */
class FrameSquaredVelocityErrorCost : public StageCost
{
  public:
    FrameSquaredVelocityErrorCost(const Model &model, const std::string &frame,
                                  const Vec6 &desired_velocity, const uint nq,
                                  const uint nv, const double weight);
    FrameSquaredVelocityErrorCost(const Model &model, const std::string &frame,
                                  const Vec6 &desired_velocity, const uint nq,
                                  const uint nv, const Vec6 &weight);

    void set_desired_velocity(const Vec6 &desired_velocity);
    void set_weight(const double weight);
    void set_weight(const Vec6 &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    Vec6                         _desired_velocity;
    const std::string            _frame;

    const uint _nq, _nv;
    Vec6       _weight;
};

/* Predefined squared frame velocity cost
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{v}_{f} \rVert_2^2$
 * where:
 *  - $\boldsymbol{v}_f$ is the velocity for a frame (in $\mathcal{R}^6$)
 *  - $\boldsymbol{w}$ is the weight on the velocity term
 */
class FrameSquaredVelocityCost : public StageCost
{
  public:
    FrameSquaredVelocityCost(const Model &model, const std::string &frame,
                             const uint nq, const uint nv, const double weight);
    FrameSquaredVelocityCost(const Model &model, const std::string &frame,
                             const uint nq, const uint nv, const Vec6 &weight);

    void set_weight(const double weight);
    void set_weight(const Vec6 &weight);

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
    Vec6       _weight;
};
}
}
}
}
}

#endif