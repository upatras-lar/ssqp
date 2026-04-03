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
/* Predefined joint angle constraint
 * Creates a constraint on the joint angles of the model
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{q}_l \leq \boldsymbol{q} \leq \boldsymbol{q}_u$
 * where:
 *  - $\boldsymbol{q}_l$, $\boldsymbol{q}_u$: Lower and upper configuration vectors
 *  - $\boldsymbol{q}$: The model's joint configuration vector
 */
class JointAngleConstraint : public StageConstraints
{
  public:
    JointAngleConstraint(const VecX &lb, const VecX &ub, const uint nq,
                         const uint nv, const bool free_flyer = false);
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

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
    const bool _free_flyer;
};

/* Predefined joint velocity constraint
 * Creates a constraint on the joint velocities of the model
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{q}_{v,l} \leq \boldsymbol{q}_v \leq \boldsymbol{q}_{v, u}$
 * where:
 *  - $\boldsymbol{q}_{v,l}$, $\boldsymbol{q}_{v,u}$: Lower and upper joint velocity vectors
 *  - $\boldsymbol{q}_v$: The model's joint velocity vector
 */
class JointVelocityConstraint : public StageConstraints
{
  public:
    JointVelocityConstraint(const VecX &lb, const VecX &ub, const uint nv,
                            const bool free_flyer = false);
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

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

/* Predefined joint control constraint
 * Creates a constraint on the joint part of the control vector
 * This can either act as a torque constraint (FD mode) or an acceleration
 * constraint (ID mode)
 * Note: this constraint is separate to any potential innate constraints
 * $\boldsymbol{u}_{c,l} \leq \boldsymbol{u}_c \leq \boldsymbol{u}_{c, u}$
 * where:
 *  - $\boldsymbol{u}_{c,l}$, $\boldsymbol{u}_{c,u}$: Lower and upper joint control vectors
 *  - $\boldsymbol{u}_c$: The joint part of the control vector
 */
class JointControlConstraint : public StageConstraints
{
  public:
    JointControlConstraint(const VecX &lb, const VecX &ub, const uint nv,
                           const uint nu, const bool free_flyer = false);
    void set_lower_bound(const VecX &lb);
    void set_upper_bound(const VecX &ub);

    uint output_size() const override;

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
    const uint _nv, _nu;
    const bool _free_flyer;
};

/* Predefined 3D position bound for a given frame
 * Creates a linear bound constraint on the position of a frame of the given
 * model: $\boldsymbol{b}_l \leq \boldsymbol{p}_f \leq \boldsymbol{b}_u$
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{p}_f$: Frame position vector (in $\mathcal{R^3}$)
 */
class FramePositionBound : public StageConstraints
{
  public:
    // Regular constructor
    FramePositionBound(const Model &model, const std::string &frame,
                       const Vec3 &lb, const Vec3 &ub, const uint nq,
                       const uint nv);

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

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
    const std::string            _frame;
    Vec3                         _lb, _ub;
    const uint                   _nq, _nv;
};

/* Predefined 3D orientation bound for a given frame
 * Creates a nonlinear bound constraint on the orientation of a frame:
 * $\boldsymbol{b}_l \leq \log_{SO3}R_{ref}^\top R_f \leq \boldsymbol{b}_u$
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors on the orientation error (in $\mathcal{R}^3$)
 *  - $R_f$: Frame rotation matrix
 *  - $R_{ref}$: Reference rotation matrix
 */
class FrameRotationBound : public StageConstraints
{
  public:
    FrameRotationBound(const Model &model, const std::string &frame,
                       const Mat3 &Rref, const Vec3 &lb, const Vec3 &ub,
                       const uint nq, const uint nv);

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

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
    const std::string            _frame;
    Mat3                         _Rref;
    Vec3                         _lb, _ub;
    const uint                   _nq, _nv;
};

/* Predefined 3D velocity bound for a given frame
 * Creates a linear bound constraint on the velocity of a frame of the given
 * model: $\boldsymbol{b}_l \leq \boldsymbol{v}_f \leq \boldsymbol{b}_u$
 * where:
 *  - $\boldsymbol{b}_l$, $\boldsymbol{b}_u$: Lower and upper bound vectors (in $\mathcal{R}^3$)
 *  - $\boldsymbol{v}_f$: Frame velocity vector (in $\mathcal{R^3}$)
 */
class FrameLinearVelocityBound : public StageConstraints
{
  public:
    // Regular constructor
    FrameLinearVelocityBound(const Model &model, const std::string &frame,
                             const Vec3 &lb, const Vec3 &ub, const uint nq,
                             const uint nv, const bool terminal = false);

    void set_lower_bound(const Vec3 &lb);
    void set_upper_bound(const Vec3 &ub);

    uint output_size() const override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

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
    const std::string            _frame;
    Vec3                         _lb, _ub;
    const uint                   _nq, _nv;
    const bool                   _terminal;
};

/* Predefined contact constraint for a given frame
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
    ContactConstraint(const Model &model, const std::string &frame,
                      const uint nq, const uint nv, const uint nu,
                      const double mu = 0.6, const double eq_tol = 0.0);

    void set_mu(const double mu);

    uint output_size() const override;

    // Required number of forces for this constraint
    // Used for automatic adding of forces by `addConstraints`
    // Each contact constraint requires 1 force
    uint required_forces() const override;

    // Name of the frame the contact force is on
    std::string force_frame(const uint i) const override;

    // Used to set the required force for this contact constraint
    void set_forces(const std::vector<Force> &forces) override;

    VecX lower_bound() const override;
    VecX upper_bound() const override;

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
    const std::string            _frame;
    std::vector<Force>           _forces;
    const uint                   _nq, _nv, _nu;
    double                       _mu;
    const double                 _eq_tol;
};

}
}
}
}
}

#endif