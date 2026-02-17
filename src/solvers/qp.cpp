#include "solvers/qp.hpp"

namespace ssqp
{
namespace solvers
{
QPSolver::Solution::Solution(const model::Dimensions &dims)
  : dx(dims.state, dims.stage_state)
  , du(dims.control, dims.stage_control)
  , lambda(dims.state, dims.stage_state)
  , mu(dims.constraint, dims.stage_constraint)
{
}
}
}