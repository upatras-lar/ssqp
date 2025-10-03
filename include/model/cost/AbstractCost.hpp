#include "types.hpp"

#ifndef SSQP_ABSTRACT_COST_HPP
#define SSQP_ABSTRACT_COST_HPP

namespace ssqp
{
namespace cost
{
class AbstractCost
{
  public:
    // Get cost function output size (always 1 since cost function returns
    // scalar)
    uint output_size()
    {
        return 1;
    }

    // Evaluate cost function l_k(x, u)
    // Returns: scalar
    virtual double evaluate(const uint k, const VecX &x,
                            const VecX &u) const = 0;

    // Get cost function k gradients with respect to x and u, evaluated at given
    // (x, u)
    // Returns:
    //  dx: gradient with respect to x, (nx) x 1
    //  du: gradient with respect to u, (nu) x 1
    virtual void gradient(const uint k, const VecX &x, const VecX &u, VecX &dx,
                          VecX &du) const = 0;

    // Get cost function k hessian matrix with respect to xx, evaluated at given
    // (x, u)
    // Returns:
    //  Hxx: Hessian matrix, (nx) x (nx)
    virtual void hessian_xx(const uint k, const VecX &x, const VecX &u,
                            MatX &Hxx) const = 0;

    // Get cost function k hessian matrix with respect to xu, evaluated at given
    // (x, u)
    // Returns:
    //  Hxu: Hessian matrix, (nx) x (nu)
    virtual void hessian_xu(const uint k, const VecX &x, const VecX &u,
                            MatX &Hxu) const = 0;

    // Get cost function k hessian matrix with respect to uu, evaluated at given
    // (x, u)
    // Returns:
    //  Huu: Hessian matrix, (nu) x (nu)
    virtual void hessian_uu(const uint k, const VecX &x, const VecX &u,
                            MatX &Huu) const = 0;
};
}
}

#endif
