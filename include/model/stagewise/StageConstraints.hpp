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
// Constraint function for stage k
class StageConstraints : public Function
{
  public:
    bool is_satisfied(const VecX &x, const VecX &u,
                      const double tolerance = 1e-3) const;

    // Get constraint bounds
    virtual VecX lower_bound() const;
    virtual VecX upper_bound() const;

    // Size of difference vector: c(x1, u1) (-) c(x2, u2)
    // Where (-): the defined right_minus operator
    virtual uint constraint_diff_size() const;

    virtual VecX right_minus(const VecX &x, const VecX &y) const;

    // Get constraint k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (nc) x (ndx)
    //  dU: gradient with respect to u, (nc) x (nu)
    virtual void gradient(const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product d(mu_k^T*d c(x, u)/dx)/dx
    // Returns:
    //  Pxx: Product, (ndx) x (ndx)
    virtual void hessprod_xx(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxx) const = 0;

    // Get product d(mu_k^T*d c(x, u)/du)/dx
    // Returns:
    //  Pxu: Product, (ndx) x (nu)
    virtual void hessprod_xu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Pxu) const = 0;

    // Get product d(mu_k^T*d c(x, u)/du)/du
    // Returns:
    //  Puu: Product, (nu) x (nu)
    virtual void hessprod_uu(const VecX &x, const VecX &u, const VecX &mu,
                             MatX &Puu) const = 0;
};
}
}
}

#endif