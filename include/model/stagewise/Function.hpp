#ifndef SSQP_MODEL_STAGEWISE_FUNCTION_HPP
#define SSQP_MODEL_STAGEWISE_FUNCTION_HPP

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
    Function(const std::string &name);

    // Get function name
    virtual const std::vector<std::string> names() const;

    // Get function output size (if $f: R^m \to R^n$, this returns $n$)
    virtual uint output_size() const = 0;

    // Evaluate function $f(\boldsymbol{x}, \boldsymbol{u})$
    // Returns:
    //  out: output vector, $n \times 1$
    virtual void evaluate(ConstVecXRef x, ConstVecXRef u, VecX &out) const = 0;

    // Write $f(\boldsymbol{x}, \boldsymbol{u})$ to out, which must already be
    // sized $n$
    // Defaults to copying the output of evaluate
    virtual void evaluate_into(ConstVecXRef x, ConstVecXRef u,
                               VecXRef out) const;

    // Gradients
    // Derived classes should implement gradients as vectors (if $n = 1$) or
    // matrices (if $n > 1$)

    // Hessians
    // Derived classes should either implement hessians (if $n = 1$) or
    // hessprods (if $n > 1$)

  protected:
    std::vector<std::string> _names;

  private:
    // Holds the output the default evaluate_into copies, so that it keeps its
    // size between calls
    mutable VecX _evaluate_scratch;
};
}
}
}

#endif