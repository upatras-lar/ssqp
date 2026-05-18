#ifndef SSQP_MODEL_STAGEWISE_STAGE_COST_HPP
#define SSQP_MODEL_STAGEWISE_STAGE_COST_HPP

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
/* General abstract constraint function for an optimization stage $k$: $\ell_k(\boldsymbol{x}, \boldsymbol{u})$
 * Defines the required functions for a stage cost, while providing 
 * default implementations for some
 */
class StageCost : public Function
{
  public:
    StageCost();
    StageCost(const std::string &name);

    // Set enabled status for StageCost
    // Note that this will only be considered if disabling is valid,
    // meaning this cost is part of a merged cost and the other cost is enabled
    void set_enabled(const bool enabled);

    // Whether this cost is enabled
    bool enabled() const;

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
    virtual void gradient(const VecX &x, const VecX &u, VecX &dx,
                          VecX &du) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{xx}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{xx}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Hxx: Hessian matrix, $n_{dx} \times n_{dx}$
    virtual void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{xu}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{xu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Hxu: Hessian matrix, $n_{dx} \times n_u$
    virtual void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const = 0;

    // Get cost function hessian matrix with respect to $\boldsymbol{uu}$, evaluated at given
    // $(\boldsymbol{x}, \boldsymbol{u})$:
    // $\nabla_{uu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  Huu: Hessian matrix, $n_u \times n_u$
    virtual void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const = 0;

  protected:
    bool        _enabled = true;
    static uint TOTAL;
};
}
}
}

#endif