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
        FlatVec           dx, du;
        std::vector<MatX> P;
        std::vector<VecX> p;
        FlatVec           z, y;

        Solution(const model::Dimensions &dims);
    };

    virtual void        reset()                  = 0;
    virtual void        increaseRegularization() = 0;
    virtual SolverState solve(Solution &sol)     = 0;
    virtual uint        iterations() const       = 0;
};

}
}

#endif
