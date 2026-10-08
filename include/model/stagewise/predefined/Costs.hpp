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

/** Predefined squared joint angle cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{ang} \rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{x}_{ang}$ is the angular (position) part of the state
 *    Note that this will ignore the first 7 elements if the model has a 
 *    floating base
 *  - $\boldsymbol{w}$ is the weight on the angular term
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`), appearing after the floating base (if present)
 */
class SquaredJointAngleCost : public StageCost
{
  public:
    SquaredJointAngleCost(const Model &model, const double weight,
                          const std::string &name = "");
    SquaredJointAngleCost(const Model &model, const VecX &weight,
                          const std::string &name = "");

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const bool _free_flyer;
    const uint _nq, _nv, _ndx;
    VecX       _weight;
};

/** Predefined squared joint velocity cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{vel} \rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{x}_{vel}$ is the velocity part of the state
 *  - $\boldsymbol{w}$ is the weight on the velocity term
 * 
 * Assumes:
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class SquaredJointVelocityCost : public StageCost
{
  public:
    SquaredJointVelocityCost(const Model &model, const double weight,
                             const std::string &name = "");
    SquaredJointVelocityCost(const Model &model, const VecX &weight,
                             const std::string &name = "");

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

/** Predefined squared control cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{ctrl} \rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{u}_{ctrl}$ is the control part of the controls
 *  - $\boldsymbol{w}$ is the weight on the control term
 * 
 * Assumes:
 *  - The model control is the first part of the control vector (size 
 *    `model.nu()`)
 */
class SquaredControlCost : public StageCost
{
  public:
    SquaredControlCost(const Model &model, const double weight,
                       const std::string &name = "");
    SquaredControlCost(const Model &model, const VecX &weight,
                       const std::string &name = "");

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nu, _ndx;
    VecX       _weight;
};

/** Predefined squared inverse dynamics torque cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{\tau} \rVert_2^2$
 * where:
 *  - $\boldsymbol{\tau}$ is the actuated-joint torque caused by the controls
 *    (see Model::getActuatedTorqueVector)
 *  - $\boldsymbol{w}$ is the weight on the torque term
 *
 * Note that this cost assumes inverse dynamics is being used (i.e. treats
 * controls as accelerations)
 *
 * Assumes:
 *  - The model is in Inverse Dynamics or Reduced Inverse Dynamics mode, such
 *    that the state is made up of configuration and velocity and the
 *    controls are made up of accelerations (including base, for Inverse
 *    Dynamics; actuated joints only, for Reduced Inverse Dynamics) and
 *    potential forces
 *
 * `k` (the model stage) only matters when there are forces present at that
 * stage; it selects which of those forces this cost accounts for, since a
 * cost added at multiple stages otherwise has no way to know which one it's
 * currently being evaluated at. Construct one instance per stage (rather
 * than sharing a single instance across stages via Model::addCost) whenever
 * forces vary between the stages a cost is added to
 */
class SquaredIDTorqueCost : public StageCost
{
  public:
    SquaredIDTorqueCost(const Model &model, const double weight,
                        const std::string &name = "", const uint k = 0);
    SquaredIDTorqueCost(const Model &model, const VecX &weight,
                        const std::string &name = "", const uint k = 0);

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                 *_model;
    const uint                   _nq, _nv, _nu, _k;
    VecX                         _weight;
};

/** Predefined squared force cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{f} \rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{u}_{f}$ is the force part of the controls
 *  - $\boldsymbol{w}$ is the weight on the force term
 * 
 * Assumes:
 *  - The model forces make up the second and final part of the control vector,
 *    following the control part of size `model.nu()`, for any remaining
 *    elements
 */
class SquaredForceCost : public StageCost
{
  public:
    SquaredForceCost(const Model &model, const double weight,
                     const std::string &name = "");
    SquaredForceCost(const Model &model, const VecX &weight,
                     const std::string &name = "");

    void set_weight(const double weight);
    void set_weight(const VecX &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const uint _nv, _nu, _ndx;
    double     _weight;
    VecX       _weight_vec;
    bool       _single_weight;
};

/** Predefined squared configuration error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{q} - \boldsymbol{q}_d)\rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{q}$ is the configuration part of the state
 *  - $\boldsymbol{q}_d$ is the desired configuration
 *  - $\boldsymbol{w}$ is the weight for the configuration error
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class SquaredConfigurationErrorCost : public StageCost
{
  public:
    SquaredConfigurationErrorCost(const Model       &model,
                                  const VecX        &desired_configuration,
                                  const double       weight,
                                  const std::string &name = "");
    SquaredConfigurationErrorCost(const Model       &model,
                                  const VecX        &desired_configuration,
                                  const VecX        &weight,
                                  const std::string &name = "");

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
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    VecX                         _desired_configuration;
    VecX                         _full_desired_configuration;
    const bool                   _free_flyer;
    VecX                         _weight;
};

/** Predefined squared floating base translation error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{t}_f - \boldsymbol{t}_d)\rVert_2^2$
 *
 * where:
 *  - $\boldsymbol{t}_f$ is the floating base translation (in $\mathcal{R}^3$)
 *  - $\boldsymbol{t}_d$ is the desired translation of the floating base (in $\mathcal{R}^3$)
 *  - $\boldsymbol{w}$ is the weight on the translation error
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The floating base translation is extracted from the base pose of the model
 *    state representation
 */
class FloatingBaseSquaredTranslationErrorCost : public StageCost
{
  public:
    FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                            const Vec3  &desired_translation,
                                            const double weight,
                                            const std::string &name = "");

    FloatingBaseSquaredTranslationErrorCost(const Model &model,
                                            const Vec3  &desired_translation,
                                            const Vec3  &weight,
                                            const std::string &name = "");

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
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    Vec3                         _desired_translation;
    Vec3                         _weight;
};

/** Predefined squared floating base rotation error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \log_{SO(3)}(R_f^\top R_d)\rVert_2^2$
 *
 * where:
 *  - $R_f$ is the floating base rotation (in $\mathit{SO3}$)
 *  - $R_d$ is the desired rotation of the floating base (in $\mathit{SO3}$)
 *  - $\boldsymbol{w}$ is the weight on the rotation error
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The floating base orientation is extracted from the base pose of the model
 *    state representation
 */
class FloatingBaseSquaredRotationErrorCost : public StageCost
{
  public:
    FloatingBaseSquaredRotationErrorCost(const Model       &model,
                                         const Mat3        &desired_rotation,
                                         const double       weight,
                                         const std::string &name = "");

    FloatingBaseSquaredRotationErrorCost(const Model       &model,
                                         const Mat3        &desired_rotation,
                                         const Vec3        &weight,
                                         const std::string &name = "");

    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_weight(const double weight);
    void set_weight(const Vec3 &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;

    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    Mat3                         _desired_rotation;
    Vec3                         _weight;
};

/** Predefined squared floating base pose error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} D^\top W D$
 *
 * where:
 *  - $D = \log_{SE(3)} T^{-1}T_d$
 *  - $T$ is the floating base pose (in $\mathit{SE3}$)
 *  - $T_d$ is the desired floating base pose (in $\mathit{SE3}$)
 *  - $W$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
 *    for rotation
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 */
class FloatingBaseSquaredPoseErrorCost : public StageCost
{
  public:
    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Mat4        &desired_pose,
                                     const double       weight,
                                     const std::string &name = "");

    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Mat4        &desired_pose,
                                     const double       translation_weight,
                                     const double       rotation_weight,
                                     const std::string &name = "");

    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Mat4        &desired_pose,
                                     const Vec6        &weight,
                                     const std::string &name = "");

    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Vec3        &desired_translation,
                                     const Mat3        &desired_rotation,
                                     const double       weight,
                                     const std::string &name = "");

    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Vec3        &desired_translation,
                                     const Mat3        &desired_rotation,
                                     const double       translation_weight,
                                     const double       rotation_weight,
                                     const std::string &name = "");

    FloatingBaseSquaredPoseErrorCost(const Model       &model,
                                     const Vec3        &desired_translation,
                                     const Mat3        &desired_rotation,
                                     const Vec6        &weight,
                                     const std::string &name = "");

    void set_desired_pose(const Mat4 &desired_pose);
    void set_desired_translation(const Vec3 &desired_translation);
    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_translation_weight(const double translation_weight);
    void set_rotation_weight(const double rotation_weight);
    void set_weight(const double weight);
    void set_weight(const Vec6 &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;

    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    Mat4                         _desired_pose, _desired_pose_inv;
    Mat6                         _weight;
};

/** Predefined squared floating base velocity error cost
 *
 * Defines the cost term
 * $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{v}_b - \boldsymbol{v}_d)\rVert_2^2$
 *
 * where:
 *  - $\boldsymbol{v}_b$ is the floating base velocity (in $\mathcal{R}^6$)
 *  - $\boldsymbol{v}_d$ is the desired floating base velocity (in $\mathcal{R}^6$)
 *  - $\boldsymbol{w}$ is the weight on the velocity error
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FloatingBaseSquaredVelocityErrorCost : public StageCost
{
  public:
    FloatingBaseSquaredVelocityErrorCost(const Model       &model,
                                         const Vec6        &desired_velocity,
                                         const double       weight,
                                         const std::string &name = "");

    FloatingBaseSquaredVelocityErrorCost(const Model       &model,
                                         const Vec6        &desired_velocity,
                                         const Vec6        &weight,
                                         const std::string &name = "");

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
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    Vec6                         _desired_velocity;
    Vec6                         _weight;
};

/** Predefined squared frame translation error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{t} - \boldsymbol{t}_d)\rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{t}$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
 *  - $\boldsymbol{t}_d$ is the desired translation for that frame (in $\mathcal{R}^3$)
 *  - $\boldsymbol{w}$ is the weight on the translation error
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FrameSquaredTranslationErrorCost : public StageCost
{
  public:
    FrameSquaredTranslationErrorCost(const Model       &model,
                                     const std::string &frame,
                                     const Vec3        &desired_translation,
                                     const double       weight,
                                     const std::string &name = "");
    FrameSquaredTranslationErrorCost(const Model       &model,
                                     const std::string &frame,
                                     const Vec3        &desired_translation,
                                     const Vec3        &weight,
                                     const std::string &name = "");

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
    const Model                  *_model;
    const uint                   _nq, _nv, _ndx;
    Vec3                         _desired_translation;
    const std::string            _frame;
    const uint                    _frame_id;
    Vec3                         _weight;
    mutable MatX                 _jac_scratch;
};

/** Predefined squared frame terrain height error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} w^2 (p_z - h(p_x, p_y) - d_d)^2$
 *
 * where:
 *  - $\boldsymbol{p}$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
 *  - $h$ is the height of the model's terrain at the frame's horizontal position
 *  - $d_d$ is the desired height of the frame above the terrain
 *  - $w$ is the weight on the height error
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 */
class FrameSquaredTerrainHeightErrorCost : public StageCost
{
  public:
    FrameSquaredTerrainHeightErrorCost(const Model       &model,
                                       const std::string &frame,
                                       const double       desired_height,
                                       const double       weight,
                                       const std::string &name = "");

    void set_desired_height(const double desired_height);
    void set_weight(const double weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                 *_model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    double                       _desired_height;
    double                       _weight;
    mutable MatX                 _jac_scratch;
};

/** Predefined squared frame rotation error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \log_{SO(3)}R^\top R_d\rVert_2^2$
 * 
 * where:
 *  - $R$ is the frame rotation for a given frame of the model (in $\mathit{SO3}$)
 *  - $R_d$ is the desired rotation for that frame (in $\mathit{SO3}$)
 *  - $\boldsymbol{w}$ is the weight on the rotation error
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FrameSquaredRotationErrorCost : public StageCost
{
  public:
    FrameSquaredRotationErrorCost(const Model &model, const std::string &frame,
                                  const Mat3        &desired_rotation,
                                  const double       weight,
                                  const std::string &name = "");

    FrameSquaredRotationErrorCost(const Model &model, const std::string &frame,
                                  const Mat3        &desired_rotation,
                                  const Vec3        &weight,
                                  const std::string &name = "");

    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_weight(const double weight);
    void set_weight(const Vec3 &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                  *_model;
    const uint                   _nq, _nv, _ndx;
    Mat3                         _desired_rotation;
    const std::string            _frame;
    const uint                    _frame_id;
    Vec3                         _weight;
    mutable MatX                 _jac_scratch;
};

/** Predefined squared frame pose error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} D^\top W D$
 * 
 * where:
 *  - $D = \log_{SE(3)} T^{-1}T_d$
 *  - $T$ is the frame pose for a given frame of the model (in $\mathit{SE3}$)
 *  - $T_d$ is the desired pose for that frame (in $\mathit{SE3}$)
 *  - $W$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
 *    for rotation
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FrameSquaredPoseErrorCost : public StageCost
{
  public:
    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Mat4 &desired_pose, const double weight,
                              const std::string &name = "");

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Mat4        &desired_pose,
                              const double       translation_weight,
                              const double       rotation_weight,
                              const std::string &name = "");

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Mat4 &desired_pose, const Vec6 &weight,
                              const std::string &name = "");

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Vec3 &desired_translation,
                              const Mat3 &desired_rotation, const double weight,
                              const std::string &name = "");

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Vec3        &desired_translation,
                              const Mat3        &desired_rotation,
                              const double       translation_weight,
                              const double       rotation_weight,
                              const std::string &name = "");

    FrameSquaredPoseErrorCost(const Model &model, const std::string &frame,
                              const Vec3 &desired_translation,
                              const Mat3 &desired_rotation, const Vec6 &weight,
                              const std::string &name = "");

    void set_desired_pose(const Mat4 &desired_pose);
    void set_desired_translation(const Vec3 &desired_translation);
    void set_desired_rotation(const Mat3 &desired_rotation);
    void set_translation_weight(const double translation_weight);
    void set_rotation_weight(const double rotation_weight);
    void set_weight(const double weight);
    void set_weight(const Vec6 &weight);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    // Hessians use GN approximation
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model                  *_model;
    const uint                   _nq, _nv, _ndx;
    Mat4                         _desired_pose, _desired_pose_inv;
    const std::string            _frame;
    const uint                    _frame_id;
    Mat6                         _weight;
    mutable MatX                 _jac_scratch;
};

/** Predefined squared frame velocity error cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{v}_f - \boldsymbol{v}_d)\rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{v}_f$ is the frame velocity for a given frame of the model (in $\mathcal{R}^6$)
 *  - $\boldsymbol{v}_d$ is the desired velocity for that frame (in $\mathcal{R}^6$)
 *  - $\boldsymbol{w}$ is the weight on the velocity error
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FrameSquaredVelocityErrorCost : public StageCost
{
  public:
    FrameSquaredVelocityErrorCost(const Model &model, const std::string &frame,
                                  const Vec6        &desired_velocity,
                                  const double       weight,
                                  const std::string &name = "");
    FrameSquaredVelocityErrorCost(const Model &model, const std::string &frame,
                                  const Vec6        &desired_velocity,
                                  const Vec6        &weight,
                                  const std::string &name = "");

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
    const Model                  *_model;
    const uint                   _nq, _nv, _ndx;
    Vec6                         _desired_velocity;
    const std::string            _frame;
    const uint                    _frame_id;
    Vec6                         _weight;
    mutable MatX                 _jac_scratch;
    mutable MatX                 _dVdq_scratch, _dVdv_scratch;
};

/** Predefined center of mass (CoM) tracking cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{c} - \boldsymbol{c}_r)\rVert_2^2$
 * 
 * where:
 *  - $\boldsymbol{c}$ is the center of mass for the model (in $\mathcal{R}^3$)
 *  - $\boldsymbol{c}_r$ is the reference point for the center of mass (in $\mathcal{R}^3$)
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class CenterOfMassTrackingCost : public StageCost
{
  public:
    CenterOfMassTrackingCost(const Model &model, const Vec3 &cref,
                             const double weight, const std::string &name = "");
    CenterOfMassTrackingCost(const Model &model, const Vec3 &cref,
                             const Vec3 &weight, const std::string &name = "");

    void set_weight(const double weight);
    void set_weight(const Vec3 &weight);

    void set_cref(const Vec3 &cref);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    const Model *_model;

    Vec3       _cref;
    const uint _nq, _nv, _ndx;
    Vec3       _weight;
};

/** Predefined frames squared inverse distance cost
 *
 * Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \sum_{i < j} \frac{w}{2 \lVert \boldsymbol{\sigma} \odot (\boldsymbol{t}_i - \boldsymbol{t}_j)\rVert_2^2}$
 *
 * where:
 *  - $\boldsymbol{t}_i$ is the frame translation for frame $i$ of the model (in $\mathcal{R}^3$)
 *  - $w$ is the weight on the translation error
 *  - $\boldsymbol{\sigma}$ is the distance scale, a per-axis vector (in $\mathcal{R}^3$) determining the
 *    scaling factor for each axis; the factors work in an inverse manner, i.e.
 *    larger values place less importance on an axis, while smaller ones place
 *    more
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FramesInverseSquaredDistanceCost : public StageCost
{
  public:
    FramesInverseSquaredDistanceCost(const Model                    &model,
                                     const std::vector<std::string> &frames,
                                     const double                    weight,
                                     const double distance_scale = 100.0,
                                     const std::string &name     = "");
    FramesInverseSquaredDistanceCost(const Model                    &model,
                                     const std::vector<std::string> &frames,
                                     const double                    weight,
                                     const Vec3        &distance_scale,
                                     const std::string &name = "");

    void set_weight(const double weight);

    void set_distance_scale(const double distance_scale);
    void set_distance_scale(const Vec3 &distance_scale);

    VecX evaluate(const VecX &x, const VecX &u) const override;
    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override;
    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override;
    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override;
    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override;

  protected:
    // Resolves each frame name to its index once, at construction
    static std::vector<uint> _resolveFrameIds(
      const Model &model, const std::vector<std::string> &frames);

    const Model                   *_model;
    const uint                     _nq, _nv, _ndx;
    const std::vector<std::string> _frames;
    const std::vector<uint>        _frame_ids;
    double                         _weight;
    Vec3                           _distance_scale;
    mutable MatX                   _jac_scratch;
    mutable std::vector<MatX>      _jac_cache;
};
}
}
}
}
}

#endif