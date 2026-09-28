#ifndef SSQP_SOLVERS_QP_HPP
#define SSQP_SOLVERS_QP_HPP

#include "types.hpp"

namespace ssqp
{
namespace solvers
{

class QPSolver
{
  public:
    struct Solution
    {
        FlatVec dx, du;
        FlatVec lambda, mu;

        Solution(const model::Dimensions &dims);
    };

    virtual ~QPSolver() = default;

    virtual void        reset()                  = 0;
    virtual void        increaseRegularization() = 0;
    virtual SolverState solve(Solution &sol)     = 0;
    virtual uint        iterations() const       = 0;

    // Feedback gain K (nu-by-nx) of the last solve at stage k, giving the
    // change du = K dx of the stage's controls for a change dx of its state
    virtual MatX feedbackGain(const uint k) const = 0;
};

}
}

#endif
