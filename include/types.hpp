#ifndef SSQP_TYPES_HPP
#define SSQP_TYPES_HPP

#include <memory>
#include <vector>
#include <string>

#include <Eigen/Core>

namespace ssqp
{
using VecX  = Eigen::VectorXd;
using MatX  = Eigen::MatrixXd;
using DMatX = Eigen::DiagonalMatrix<double, -1>;

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

// Defines an optimal control problem over a horizon for the given costs,
// dynamics and constraints
struct OCP
{
    std::shared_ptr<dynamics::AbstractDynamics>     dynamics;
    std::shared_ptr<cost::AbstractCost>             costs;
    std::shared_ptr<constraint::AbstractConstraint> constraints;
    uint                                            horizon;
};

namespace solvers
{
enum SolverState
{
    SUCCESS           = 1,
    LOOSE_SUCCESS     = 2,
    PRIMAL_INFEASIBLE = 10,
    DUAL_INFEASIBLE   = 11,
    MAX_ITER_REACHED  = 13,
    ONGOING           = 100,
};

const std::string to_string(const SolverState &state);

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
}
}

#endif
