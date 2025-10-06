#ifndef SSQP_STAGE_COST_HPP
#define SSQP_STAGE_COST_HPP

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
// Cost function for stage k
class StageCost : public Function
{
  public:
    uint output_size() const override
    {
        return 1;
    }

    // Get cost k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (nx) x 1
    //  dU: gradient with respect to u, (nu) x 1
    virtual void gradient(const VecX &x, const VecX &u, VecX &dx,
                          VecX &du) const = 0;

    // Get cost function k hessian matrix with respect to xx, evaluated at given
    // (x, u)
    // Returns:
    //  Hxx: Hessian matrix, (nx) x (nx)
    virtual void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const = 0;

    // Get cost function k hessian matrix with respect to xu, evaluated at given
    // (x, u)
    // Returns:
    //  Hxu: Hessian matrix, (nx) x (nu)
    virtual void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const = 0;

    // Get cost function k hessian matrix with respect to uu, evaluated at given
    // (x, u)
    // Returns:
    //  Huu: Hessian matrix, (nu) x (nu)
    virtual void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const = 0;
};
}
}
}

#endif