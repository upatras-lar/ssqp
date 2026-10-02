#ifndef SSQP_MODEL_STAGEWISE_PREDEFINED_CONSTRAINTS_HPP
#define SSQP_MODEL_STAGEWISE_PREDEFINED_CONSTRAINTS_HPP

#include "types.hpp"
#include "model/Model.hpp"
#include "model/stagewise/StageConstraints.hpp"

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
/** Predefined joint angle constraint
 *
 * Creates a constraint on the joint angles of the model
 * 
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{q}_l \leq \boldsymbol{q} \leq \boldsymbol{q}_u$
 * 
 * where:
 *  - $\boldsymbol{q}_l$, $\boldsymbol{q}_u$: Lower and upper configuration vectors
 *  - $\boldsymbol{q}$: The model's joint configuration vector
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`), appearing after the floating base (if present)
 */
class JointAngleConstraint : public StageConstraints
{
  public:
    JointAngleConstraint(const Model &model, const VecX &lb, const VecX &ub,
                         const std::string &name = "");
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    VecX       _lb, _ub;
    const bool _free_flyer;
    const uint _nv, _njoints, _ndx;
};

/** Predefined joint velocity constraint
 *
 * Creates a constraint on the joint velocities of the model
 * 
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{q}_{v,l} \leq \boldsymbol{q}_v \leq \boldsymbol{q}_{v, u}$
 * 
 * where:
 *  - $\boldsymbol{q}_{v,l}$, $\boldsymbol{q}_{v,u}$: Lower and upper joint velocity vectors
 *  - $\boldsymbol{q}_v$: The model's joint velocity vector
 * 
 * Assumes:
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class JointVelocityConstraint : public StageConstraints
{
  public:
    JointVelocityConstraint(const Model &model, const VecX &lb, const VecX &ub,
                            const std::string &name = "");
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    VecX       _lb, _ub;
    const uint _nv, _njoints;
};

/** Predefined joint control constraint
 *
 * Creates a constraint on the joint part of the control vector
 * 
 * This can either act as a torque constraint (FD mode) or an acceleration
 * constraint (ID mode)
 * 
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{u}_{c,l} \leq \boldsymbol{u}_c \leq \boldsymbol{u}_{c, u}$
 * 
 * where:
 *  - $\boldsymbol{u}_{c,l}$, $\boldsymbol{u}_{c,u}$: Lower and upper joint control vectors
 *  - $\boldsymbol{u}_c$: The joint part of the control vector
 * 
 * Assumes:
 *  - The model control is the first part of the control vector (size 
 *    `model.nu()`)
 */
class JointControlConstraint : public StageConstraints
{
  public:
    JointControlConstraint(const Model &model, const VecX &lb, const VecX &ub,
                           const std::string &name = "");
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    VecX       _lb, _ub;
    const bool _free_flyer;
    const uint _nv, _nu, _ndx;
};

/** Floating base position bound constraint
 *
 * Creates a linear bound constraint on the position of the floating base:
 * $\boldsymbol{b}_l \leq p_f \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the base position (in $\mathcal{R}^3$)
 *  - $p_f$: Floating base position extracted from the system state
 *
 * Assumes:
 *  - The Model contains a floating-base where the base position is the first
 *    part of the configuration vector (first 3 elements of size `model.nq()`)
 */
class FloatingBaseTranslationBound : public StageConstraints
{
  public:
    FloatingBaseTranslationBound(const Model &model, const Vec3 &lb,
                                 const Vec3 &ub, const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    VecX                         _lb, _ub;
};

/** Predefined floating base orientation bound
 *
 * Creates a nonlinear bound constraint on the orientation of the floating base:
 * $\boldsymbol{b}_l \leq \log_{SO3}(R_{ref}^\top R_f) \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
 *  - $R_f$: Floating base rotation matrix
 *  - $R_{ref}$: Reference rotation matrix
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The floating base orientation is represented as a rotation matrix in SE(3)
 *    and can be extracted from the base pose in the system state
 */
class FloatingBaseRotationBound : public StageConstraints
{
  public:
    FloatingBaseRotationBound(const Model &model, const Mat3 &Rref,
                              const Vec3 &lb          = Vec3::Zero(),
                              const Vec3 &ub          = Vec3::Zero(),
                              const std::string &name = "");

    void set_Rref(const Mat3 &Rref);

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    Mat3                         _Rref;
    VecX                         _lb, _ub;
};

/** Predefined floating base SE(3) pose bound
 *
 * Creates a nonlinear bound constraint on the full pose (position + orientation)
 * of the floating base:
 * $\boldsymbol{b}_l \leq \log_{SE3}(T_{ref}^{-1} T_f) \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the pose error (in $\mathcal{R}^6$)
 *  - $T_f$: Floating base pose in SE(3)
 *  - $T_{ref}$: Reference pose in SE(3)
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The floating base pose is represented as an SE(3) transform and can be
 *    extracted from the base state in the system model
 */
class FloatingBasePoseBound : public StageConstraints
{
  public:
    FloatingBasePoseBound(const Model &model, const Mat4 &Tref,
                          const Vec6 &lb          = Vec6::Zero(),
                          const Vec6 &ub          = Vec6::Zero(),
                          const std::string &name = "");

    FloatingBasePoseBound(const Model &model, const Vec3 &tref,
                          const Mat3 &Rref, const Vec6 &lb = Vec6::Zero(),
                          const Vec6 &ub          = Vec6::Zero(),
                          const std::string &name = "");

    void set_Tref(const Mat4 &Tref);
    void set_tref(const Vec3 &tref);
    void set_Rref(const Mat3 &Rref);

    void set_lower_bound(const Vec6 &lb);
    void set_upper_bound(const Vec6 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    Mat4                         _Tref, _Tref_inv;
    VecX                         _lb, _ub;
};

/** Predefined floating base linear velocity bound
 *
 * Creates a linear bound constraint on the linear velocity of the floating base:
 * $\boldsymbol{b}_l \leq \boldsymbol{v}_b \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{v}_b$: Floating base linear velocity vector (in $\mathcal{R}^3$)
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FloatingBaseLinearVelocityBound : public StageConstraints
{
  public:
    FloatingBaseLinearVelocityBound(const Model &model, const Vec3 &lb,
                                    const Vec3        &ub,
                                    const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    VecX                         _lb, _ub;
};

/** Predefined floating base angular velocity bound
 *
 * Creates a linear bound constraint on the angular velocity of the floating base:
 * $\boldsymbol{b}_l \leq \boldsymbol{\omega}_b \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{\omega}_b$: Floating base angular velocity vector (in $\mathcal{R}^3$)
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FloatingBaseAngularVelocityBound : public StageConstraints
{
  public:
    FloatingBaseAngularVelocityBound(const Model &model, const Vec3 &lb,
                                     const Vec3        &ub,
                                     const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    VecX                         _lb, _ub;
};

/** Predefined 3D position bound for a given frame
 *
 * Creates a linear bound constraint on the position of a frame of the given
 * model: $\boldsymbol{b}_l \leq \boldsymbol{p}_f \leq \boldsymbol{b}_u$
 * 
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FrameTranslationBound : public StageConstraints
{
  public:
    // Regular constructor
    FrameTranslationBound(const Model &model, const std::string &frame,
                          const Vec3 &lb, const Vec3 &ub,
                          const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    VecX                         _lb, _ub;
    mutable MatX                 _jac_scratch;
};

/** Predefined terrain height bound for a given frame
 *
 * Creates a bound constraint on the height of a frame above the model's
 * terrain: $b_l \leq p_z - h(p_x, p_y) \leq b_u$
 *
 * where:
 *  - $b_l$, $b_u$: Lower and upper bounds on the height above the terrain
 *  - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
 *  - $h$: Terrain height at the frame's horizontal position
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 */
class FrameTerrainHeightBound : public StageConstraints
{
  public:
    FrameTerrainHeightBound(const Model &model, const std::string &frame,
                            const double lb, const double ub,
                            const std::string &name = "");

    void set_lower_bound(const double lb);
    void set_upper_bound(const double ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    VecX                         _lb, _ub;
    mutable MatX                 _jac_scratch;
};

/** Predefined 3D orientation bound for a given frame
 *
 * Creates a nonlinear bound constraint on the orientation of a frame:
 * $\boldsymbol{b}_l \leq \log_{SO3}R_{ref}^\top R_f \leq \boldsymbol{b}_u$
 * 
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
 *  - $R_f$: Frame rotation matrix
 *  - $R_{ref}$: Reference rotation matrix
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class FrameRotationBound : public StageConstraints
{
  public:
    FrameRotationBound(const Model &model, const std::string &frame,
                       const Mat3 &Rref, const Vec3 &lb = Vec3::Zero(),
                       const Vec3 &ub          = Vec3::Zero(),
                       const std::string &name = "");

    void set_Rref(const Mat3 &Rref);

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    Mat3                         _Rref;
    VecX                         _lb, _ub;
    mutable MatX                 _jac_scratch;
};

/** Predefined SE(3) pose bound for a given frame
 *
 * Creates a nonlinear bound constraint on the pose of a frame:
 * $\boldsymbol{b}_l \leq \log_{SE3}(T_{ref}^{-1} T_f) \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the pose error (in $\mathcal{R}^6$)
 *  - $T_f$: Frame pose in SE(3)
 *  - $T_{ref}$: Reference pose in SE(3)
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 */
class FramePoseBound : public StageConstraints
{
  public:
    FramePoseBound(const Model &model, const std::string &frame,
                   const Mat4 &Tref, const Vec6 &lb = Vec6::Zero(),
                   const Vec6 &ub = Vec6::Zero(), const std::string &name = "");

    FramePoseBound(const Model &model, const std::string &frame,
                   const Vec3 &tref, const Mat3 &Rref,
                   const Vec6 &lb = Vec6::Zero(), const Vec6 &ub = Vec6::Zero(),
                   const std::string &name = "");

    void set_Tref(const Mat4 &Tref);
    void set_tref(const Vec3 &tref);
    void set_Rref(const Mat3 &Rref);

    void set_lower_bound(const Vec6 &lb);
    void set_upper_bound(const Vec6 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    Mat4                         _Tref, _Tref_inv;
    VecX                         _lb, _ub;
    mutable MatX                 _jac_scratch;
};

/** Predefined 3D velocity bound for a given frame
 *
 * Creates a linear bound constraint on the velocity of a frame of the given
 * model: $\boldsymbol{b}_l \leq \boldsymbol{v}_f \leq \boldsymbol{b}_u$
 * 
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{v}_f$: Frame velocity vector (in $\mathcal{R^3}$)
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FrameLinearVelocityBound : public StageConstraints
{
  public:
    // Regular constructor
    FrameLinearVelocityBound(const Model &model, const std::string &frame,
                             const Vec3 &lb, const Vec3 &ub,
                             const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    VecX                         _lb, _ub;
    mutable MatX                 _dVdq_scratch, _dVdv_scratch;
};

/** Predefined angular velocity bound for a given frame
 *
 * Creates a linear bound constraint on the angular velocity of a frame of the
 * given model:
 * $\boldsymbol{b}_l \leq \boldsymbol{\omega}_f \leq \boldsymbol{b}_u$
 *
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{\omega}_f$: Frame angular velocity vector (in $\mathcal{R}^3$)
 *
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size
 *    `model.nq()`)
 *  - The model joint velocity is the final part of the state vector (size
 *    `model.nv()`)
 */
class FrameAngularVelocityBound : public StageConstraints
{
  public:
    // Regular constructor
    FrameAngularVelocityBound(const Model &model, const std::string &frame,
                              const Vec3 &lb, const Vec3 &ub,
                              const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    const std::string            _frame;
    const uint                   _frame_id;
    VecX                         _lb, _ub;
    mutable MatX                 _dVdq_scratch, _dVdv_scratch;
};

/** Predefined center of mass bound
 *
 * Creates a linear bound constraint on the center of mass of the model:
 * $\boldsymbol{b}_l \leq \boldsymbol{c} \leq \boldsymbol{b}_u$
 * 
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{c}$ is the center of mass of the model
 * 
 * Assumes:
 *  - The model joint configuration is the first part of the state vector (size 
 *    `model.nq()`)
 */
class CenterOfMassBound : public StageConstraints
{
  public:
    CenterOfMassBound(const Model &model, const Vec3 &lb, const Vec3 &ub,
                      const std::string &name = "");

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _ndx;
    VecX                         _lb, _ub;
};

/** Predefined contact constraint for a given frame
 * Creates a set of constraints that amount to a contact force constraint for
 * a frame
 * This consists of the following 4-point linearized friction cone constraints:
 *  - $f_n \gt 0$
 *  - $|f_1| \leq \mu f_n$
 *  - $|f_2| \leq \mu f_n$
 * As well as the following constraint on the 3D position of the frame the
 * contact is on:
 *  - $p_z = h_z$
 * where:
 *  - $f_n$ is the normal of the contact force ($f_z$)
 *  - $f_1$ is the first component of the tangential contact force ($f_x$)
 *  - $f_2$ is the second component of the tangential contact force ($f_y$)
 *  - $\mu$ is the static friction coefficient at the contact
 *  - $p_z$ is the $z-$component of the translation of the frame the contact is on
 *  - $h_z$ is the height of the terrain at point $(p_x, p_y)$
 * Note that when manually adding this contact constraint to a model, an 
 * external force must be added to the model and the appropriate methods must 
 * be called to correctly assign that force to this constraint. To avoid this,
 * simply use the `Model`'s `addConstraints` function, instantiated with the
 * template type `ContactConstraint`. In that case, adding and assigning of the
 * force will be handled automatically.
 * Also note that this constraint should be coupled with a `FrameLinearVelocityBound`
 * on the frame to avoid slipping.
 */
class ContactConstraint : public StageConstraints
{
  public:
    ContactConstraint(
      const Model &model, const std::string &frame, const double mu = 0.6,
      const double max_normal_force = std::numeric_limits<double>::infinity(),
      const double eq_tol = 0.0, const std::string &name = "");

    void set_mu(const double mu);
    void set_max_normal_force(const double max_normal_force);
    void set_eq_tol(const double eq_tol);

    void set_enabled(const bool enabled) override;

    uint output_size() const override;

    // Required number of forces for this constraint
    // Used for automatic adding of forces by `addConstraints`
    // Each contact constraint requires 1 force
    uint required_forces() const override;

    // Name of the frame the contact force is on
    std::string force_frame(const uint i) const override;

    // Used to set the required force for this contact constraint
    void set_forces(const std::vector<std::shared_ptr<Force>> &forces) override;

    const VecX &lower_bound() const override;
    const VecX &upper_bound() const override;

    VecX evaluate(const VecX &x, const VecX &u) const override;

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override;

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxx) const override;
    void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Pxu) const override;
    void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                     MatX &Puu) const override;

  protected:
    const std::shared_ptr<Model> _model;
    const uint                   _nq, _nv, _nu;
    const std::string            _frame;
    const uint                   _frame_id;
    std::shared_ptr<Force>       _force;
    double                       _mu;
    double                       _max_normal_force;
    double                       _eq_tol;
    VecX                         _lb, _ub;
    mutable MatX                 _jac_world_scratch, _jac_local_scratch;
};

}
}
}
}
}

#endif