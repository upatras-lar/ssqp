#include <memory>

#include <Eigen/Core>

#ifndef SSQP_TYPES_HPP
#define SSQP_TYPES_HPP

namespace ssqp
{
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;

namespace dynamics
{
class AbstractDynamics;
}

namespace cost
{
class AbstractCost;
}

namespace constraint
{
class AbstractConstraint;
}

namespace solvers
{
struct OCP
{
    std::shared_ptr<dynamics::AbstractDynamics>     dynamics;
    std::shared_ptr<cost::AbstractCost>             costs;
    std::shared_ptr<constraint::AbstractConstraint> constraints;
};

struct Guess
{
    // Vector sequences indexed over time horizon for state and control guesses
    // n
    // Matrix x is of dimensions nx x (H + 1) (starting at x_0 and ending at
    // x_H)
    // Matrix u is of dimensions nu x H (starting at u_0 and ending at
    // u_(H-1))
    MatX x, u;
    // Equivalent vector sequences for lagrange multiplier guesses n
    // Matrix lambda is of dimensions nx x H (starting at 1 and ending at H)
    // Matrix mu is of dimensions nc x (H + 1) (starting at 0 and ending at H)
    // (nc: number of constraints)
    MatX lambda, mu;
};
}
}

#endif
