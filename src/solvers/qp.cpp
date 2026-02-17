#include "solvers/qp.hpp"

namespace ssqp
{
namespace solvers
{
QPSolver::Solution::Solution(const model::Dimensions &dims)
  : dx(dims.state, dims.stage_state)
  , du(dims.control, dims.stage_control)
  , z(dims.constraint, dims.stage_constraint)
  , y(z, false)
{
}
}
}