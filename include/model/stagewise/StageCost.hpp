#ifndef SSQP_MODEL_STAGEWISE_STAGE_COST_HPP
#define SSQP_MODEL_STAGEWISE_STAGE_COST_HPP

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
/** General abstract constraint function for an optimization stage $k$: $\ell_k(\boldsymbol{x}, \boldsymbol{u})$
 * Defines the required functions for a stage cost, while providing 
 * default implementations for some
 */
class StageCost : public Function
{
  public:
    StageCost();
    StageCost(const std::string &name);

    // Set enabled status for StageCost
    // A stage always keeps at least one cost, so this has no effect if every
    // cost on the stage is disabled
    virtual void set_enabled(const bool enabled);

    // Whether this cost is enabled
    virtual bool enabled() const;

    // Set the forces and wrenches of the stage this cost is added to, in the
    // order they appear in the controls
    // Called by the model when it's finalized, once for every stage the cost
    // is added to; has no effect by default
    virtual void set_stage_forces(
      const std::vector<std::shared_ptr<Force>> &forces,
      const std::vector<std::shared_ptr<Force>> &wrenches);

    // Cost function output size, always $1$
    uint output_size() const final
    {
        return 1;
    }

    // Get cost Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
    // given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \ell_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  dx: gradient with respect to $\boldsymbol{x}$, $n_{dx} \times 1$
    //  du: gradient with respect to $\boldsymbol{u}$, $n_u \times 1$
    virtual void gradient(ConstVecXRef x, ConstVecXRef u, VecX &dx,
                          VecX &du) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{xx}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{xx}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Hxx: Hessian matrix, $n_{dx} \times n_{dx}$
    virtual void hessian_xx(ConstVecXRef x, ConstVecXRef u,
                            MatX &Hxx) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{xu}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{xu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Hxu: Hessian matrix, $n_{dx} \times n_u$
    virtual void hessian_xu(ConstVecXRef x, ConstVecXRef u,
                            MatX &Hxu) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{uu}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{uu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Huu: Hessian matrix, $n_u \times n_u$
    virtual void hessian_uu(ConstVecXRef x, ConstVecXRef u,
                            MatX &Huu) const = 0;

    // Add the cost gradients to dx and du, which must already be sized
    // $n_{dx}$ and $n_u$
    // Defaults to adding the output of gradient
    virtual void add_gradient(ConstVecXRef x, ConstVecXRef u, VecXRef dx,
                              VecXRef du) const;

    // Add $\nabla_{xx}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$ to Hxx, which
    // must already be $n_{dx} \times n_{dx}$
    // Defaults to adding the output of hessian_xx
    virtual void add_hessian_xx(ConstVecXRef x, ConstVecXRef u,
                                MatXRef Hxx) const;

    // Add $\nabla_{xu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$ to Hxu, which
    // must already be $n_{dx} \times n_u$
    // Defaults to adding the output of hessian_xu
    virtual void add_hessian_xu(ConstVecXRef x, ConstVecXRef u,
                                MatXRef Hxu) const;

    // Add $\nabla_{uu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$ to Huu, which
    // must already be $n_u \times n_u$
    // Defaults to adding the output of hessian_uu
    virtual void add_hessian_uu(ConstVecXRef x, ConstVecXRef u,
                                MatXRef Huu) const;

  protected:
    bool        _enabled = true;
    static uint TOTAL;

  private:
    // Hold the outputs the default add methods add, one per output so that
    // each keeps its size between calls
    mutable VecX _add_scratch_dx, _add_scratch_du;
    mutable MatX _add_scratch_xx, _add_scratch_xu, _add_scratch_uu;
};
}
}
}

#endif