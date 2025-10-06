#ifndef SSQP_FUNCTION_HPP
#define SSQP_FUNCTION_HPP

#include "types.hpp"

namespace ssqp
{
namespace model
{
namespace stagewise
{
// Function interface for stage k
class Function
{
  public:
    // Get function output size (if f: R^m -> R^n, this returns n)
    virtual uint output_size() const = 0;

    // Size of state vector x
    virtual uint state_size() const = 0;

    // Size of control vector u
    virtual uint control_size() const = 0;

    // Evaluate function f(x, u)
    // Returns: output vector, n x 1
    virtual VecX evaluate(const VecX &x, const VecX &u) const = 0;

    // Gradients
    // Derived classes should implement gradients as vectors (if n == 1) or
    // matrices (if n > 1)

    // Hessians
    // Derived classes should either implement hessians (if n == 1) or hessprods
    // (if n > 1)
};
}
}
}

#endif