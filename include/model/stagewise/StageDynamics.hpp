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
    // Size of state vector x
    virtual uint state_size() const = 0;

    // Size of control vector u
    virtual uint control_size() const = 0;

    // Size of state difference dx (default: nx)
    virtual uint state_diff_size() const;

    // Initial state (nize nx) (default all 0)
    virtual VecX init_state() const;

    virtual void set_init_state(const VecX &init_state);

    // Operator definitions for states
    // Defines operators between states and state differences
    // If state is a euclidian vector, these operators reduce to simple
    // addition/subtraction
    // If state is a lie group element, these operators perform conversions
    // between elements of the group (manifold) and tangent space elements
    // The state can also be composite, i.e made up of one or more manifolds and
    // one or more euclidian vectors, in which case operators act segment-wise

    // Defines right plus operator between states
    // Inputs: State x, state difference dx (sizes nx, ndx)
    // Output: State y (size nx)
    // Default behavior: x + y
    virtual VecX state_right_plus(const VecX &x, const VecX &dx) const;

    // Defines right minus operator between states
    // Inputs: Composite states x, y (vectors/lie group elements, size nx)
    // Output: State difference vector dx (size ndx)
    // Default behavior: x - y
    virtual VecX state_right_minus(const VecX &x, const VecX &y) const;

    // Get dynamics k Jacobians with respect to x and u, evaluated at
    // given (x, u)
    // Returns:
    //  dX: gradient with respect to x, (ndx) x (ndx)
    //  dU: gradient with respect to u, (ndx) x (nu)
    virtual void gradient(const VecX &x, const VecX &u, MatX &dX,
                          MatX &dU) const = 0;

    // Get product d(lambda_k^T*d f(x, u)/dx)/dx
    // Returns:
    //  Pxx: Product, (ndx) x (ndx)
    virtual void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Pxx) const = 0;

    // Get product d(lambda_k^T*d f(x, u)/du)/dx
    // Returns:
    //  Pxu: Product, (ndx) x (nu)
    virtual void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Pxu) const = 0;

    // Get product d(lambda_k^T*d f(x, u)/du)/du
    // Returns:
    //  Puu: Product, (nu) x (nu)
    virtual void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                             MatX &Puu) const = 0;
};
}
}
}

#endif