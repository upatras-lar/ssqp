#include "types.hpp"

#ifndef SSQP_ABSTRACT_CONSTRAINT_HPP
#define SSQP_ABSTRACT_CONSTRAINT_HPP

namespace ssqp
{
namespace constraint
{
class AbstractConstraint
{
  public:
    // Get constraint function output size (if f: R^m -> R^n, this returns n)
    virtual uint output_size() const = 0;

    // Evaluate constraint function c_k(x, u)
    // Returns: output vector, (nc) x 1
    virtual VecX evaluate(const uint k, const VecX &x, const VecX &u) const = 0;

    // Get constraint function k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (nc) x (nx)
    //  dU: gradient with respect to u, (nc) x (nu)
    virtual void gradient(const uint k, const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product d(mu_k^T*d c(x, u)/dx)/dx
    // Returns:
    //  Pxx: Product, (nx) x (nx)
    virtual void hessprod_xx(const uint k, const VecX &x, const VecX &u,
                             const VecX &mu, MatX &Pxx) const = 0;

    // Get product d(mu_k^T*d c(x, u)/du)/dx
    // Returns:
    //  Pxu: Product, (nx) x (nu)
    virtual void hessprod_xu(const uint k, const VecX &x, const VecX &u,
                             const VecX &mu, MatX &Pxu) const = 0;

    // Get product d(mu_k^T*d c(x, u)/du)/du
    // Returns:
    //  Puu: Product, (nu) x (nu)
    virtual void hessprod_uu(const uint k, const VecX &x, const VecX &u,
                             const VecX &mu, MatX &Puu) const = 0;
};
}
}

#endif
