#ifndef SSQP_STAGE_DYNAMICS_HPP
#define SSQP_STAGE_DYNAMICS_HPP

#include "model/stagewise/Function.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
// Constraint function for stage k
class StageDynamics : public Function
{
  public:
    // Get dynamics k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (nx) x (nx)
    //  dU: gradient with respect to u, (nx) x (nu)
    virtual void gradient(const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product d(mu_k^T*d f(x, u)/dx)/dx
    // Returns:
    //  Pxx: Product, (nx) x (nx)
    virtual void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxx) const = 0;

    // Get product d(mu_k^T*d f(x, u)/du)/dx
    // Returns:
    //  Pxu: Product, (nx) x (nu)
    virtual void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxu) const = 0;

    // Get product d(mu_k^T*d f(x, u)/du)/du
    // Returns:
    //  Puu: Product, (nu) x (nu)
    virtual void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Puu) const = 0;
};
}
}
}

#endif