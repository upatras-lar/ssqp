#include "types.hpp"

#ifndef SSQP_ABSTRACT_DYNAMICS_HPP
#define SSQP_ABSTRACT_DYNAMICS_HPP

namespace ssqp
{
namespace dynamics
{
class AbstractDynamics
{
  public:
    // Get dynamics output size (if f: R^m -> R^n, this returns n)
    virtual uint output_size() const = 0;

    // Get control size
    virtual uint control_size() const = 0;

    // Evaluate dynamics function f_k(x, u)
    // Returns: output vector, (nx) x 1
    virtual VecX evaluate(const uint k, const VecX &x, const VecX &u) const = 0;

    // Get dynamics function k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (nx) x (nx)
    //  dX: gradient with respect to u, (nx) x (nu)
    virtual void gradient(const uint k, const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product d(lambda_{k+1}^T*d f(x, u)/dx)/dx
    // Returns:
    //  Pxx: Product, (nx) x (nx)
    virtual void hessprod_xx(const uint k, const VecX &x, const VecX &u,
                             const VecX &lambda, MatX &Pxx) const = 0;

    // Get product d(lambda_{k+1}^T*d f(x, u)/du)/dx
    // Returns:
    //  Pxx: Product, (nx) x (nu)
    virtual void hessprod_xu(const uint k, const VecX &x, const VecX &u,
                             const VecX &lambda, MatX &Pxu) const = 0;

    // Get product d(lambda_{k+1}^T*d f(x, u)/du)/du
    // Returns:
    //  Pxx: Product, (nu) x (nu)
    virtual void hessprod_uu(const uint k, const VecX &x, const VecX &u,
                             const VecX &lambda, MatX &Puu) const = 0;
};
}
}

#endif
