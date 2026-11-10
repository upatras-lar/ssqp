#ifndef SSQP_MODEL_STAGEWISE_STAGE_CONSTRAINTS_HPP
#define SSQP_MODEL_STAGEWISE_STAGE_CONSTRAINTS_HPP

#include <limits>

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
/** General abstract constraint function for an optimization stage $k$: $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
 * Defines the required functions for stage constraints, while providing 
 * default implementations for some
 */
class StageConstraints : public Function
{
  public:
    StageConstraints();
    StageConstraints(const std::string &name);

    // Set enabled status for StageConstraints
    virtual void set_enabled(const bool enabled);

    // Whether this constraint is enabled
    virtual bool enabled() const;

    // Number of external forces involved in the constraint (default 0)
    virtual uint required_forces() const;

    // Frame for i-th external force
    virtual std::string force_frame(const uint i) const;

    // Set the model Forces associated with this constraint
    virtual void set_forces(const std::vector<std::shared_ptr<Force>> &forces);

    // Get constraint lower bound
    virtual const VecX &lower_bound() const;

    // Get constraint upper bound
    virtual const VecX &upper_bound() const;

    // Size of the difference vector: $\boldsymbol{c}_k(\boldsymbol{x}_1, \boldsymbol{u}_1) \ominus \boldsymbol{c}_k(\boldsymbol{x}_2, \boldsymbol{u}_2)$
    // Where $\ominus$: the defined right_minus operator
    virtual uint constraint_diff_size() const;

    // The $\ominus$ operator, which outputs the difference of two constraint
    // vectors in tangent space
    virtual VecX right_minus(const VecX &x, const VecX &y) const;

    // Get constraint Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
    // given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  dX: gradient with respect to $\boldsymbol{x}$, $n_c \times n_{dx}$
    //  dU: gradient with respect to $\boldsymbol{u}$, $n_c \times n_u$
    virtual void gradient(const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product $\nabla_{x} \mu_k^\top \nabla_{x} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Pxx: Product, $n_{dx} \times n_{dx}$
    virtual void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxx) const = 0;

    // Get product $\nabla_{x} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Pxu: Product, $n_{dx} \times n_u$
    virtual void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxu) const = 0;

    // Get product $\nabla_{u} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Puu: Product, $n_u \times n_u$
    virtual void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Puu) const = 0;

    // Add the constraint Jacobians to dX and dU, which must already be
    // $n_c \times n_{dx}$ and $n_c \times n_u$
    // Defaults to adding the output of gradient
    virtual void add_gradient(const VecX &x, const VecX &u, MatXRef dX,
                              MatXRef dU) const;

    // Add $\nabla_{x} \mu_k^\top \nabla_{x} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // to Pxx, which must already be $n_{dx} \times n_{dx}$
    // Defaults to adding the output of hessprod_xx
    virtual void add_hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                                 MatXRef Pxx) const;

    // Add $\nabla_{x} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // to Pxu, which must already be $n_{dx} \times n_u$
    // Defaults to adding the output of hessprod_xu
    virtual void add_hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                                 MatXRef Pxu) const;

    // Add $\nabla_{u} \mu_k^\top \nabla_{u} \boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$
    // to Puu, which must already be $n_u \times n_u$
    // Defaults to adding the output of hessprod_uu
    virtual void add_hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                                 MatXRef Puu) const;

  protected:
    bool        _enabled = true;
    static uint TOTAL;

  private:
    // Lazily-computed default bounds (+-infinity), backing the default
    // lower_bound()/upper_bound() implementations
    mutable VecX _default_lb, _default_ub;

    // Hold the outputs the default add methods add, one per output so that
    // each keeps its size between calls
    mutable MatX _add_scratch_dX, _add_scratch_dU;
    mutable MatX _add_scratch_xx, _add_scratch_xu, _add_scratch_uu;
};
}
}
}

#endif