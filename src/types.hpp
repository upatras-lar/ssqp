#include <memory>
#include <vector>

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

// Defines an optimal control problem over a horizon for the given costs,
// dynamics and constraints
struct OCP
{
    std::shared_ptr<dynamics::AbstractDynamics>     dynamics;
    std::shared_ptr<cost::AbstractCost>             costs;
    std::shared_ptr<constraint::AbstractConstraint> constraints;
    uint                                            horizon;
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

// Holds data for non-terminal stages of the below QP
struct QPStage
{
    MatX A, B;          // A: (nx) x (nx), B: (nx) * (nu)
    VecX gamma;         // (nx) x 1
    MatX Qxx, Qxu, Quu; // Qxx: (nx) x (nx), Qxu: (nx) x (nu), Quu: (nu) x (nu)
    VecX qx, qu;        // qx: (nx) x 1, qu: (nu) x 1
    MatX D, E;          // D: (nc) x (nx), E: (nc) x (nu)
    VecX c;             // (nc) x 1
};

/* Defines a QP of the following form, to be solved using ADMM
 *  min_{x, u} sum_{k = 0}^{H-1} ( [x_k u_k] [Qxx Qxu; Qux Quu] [x_k u_k]^T +
 *  [qx_k qu_k] [x_k u_k] ) + 0.5 x_H^T Qxx_H x_H + x_H^T qx_H + 0.5*rho || D_H
 *  x_H - z_{H}^j + rho^{-1} y_{H}^j ||_{2}^2  + sum_{k=0}^{H-1} ( 0.5*rho ||
 *  D_k x_k + E_k u_k - z_{k}^j + rho^{-1} y_{k}^j ||_{2}^2 ) + sum_{k=0}^{H} (
 *  0.5*sigma || x_k - x_{k}^j ||_{2}^2 ) + sum_{k=0}^{H} ( 0.5*sigma || u_k -
 *  u_{k}^j ||_{2}^2 )
 *  such that: x_{k+1} = A_k x_k + B_k u_k + gamma_k
 * Terms with (^j) are parts of the current ADMM guess
 * )
 */
struct QPData
{
    uint                 horizon;
    uint                 nx, nu, nc;
    std::vector<QPStage> stages;

    MatX Qxx_t;
    VecX qx_t;
    MatX Dt;
    VecX ct;

    QPData(uint horizon, uint nx, uint nu, uint nc)
      : horizon(horizon)
      , nx(nx)
      , nu(nu)
      , nc(nc)
      , stages(horizon)
    {
    }
};

// Solution to QP solved with ADMM
struct admmQPSolution
{
    MatX              dX, dU;
    MatX              dX_hat;
    std::vector<MatX> P;
    std::vector<VecX> p;
    MatX              z, y;
};

}
}

#endif
