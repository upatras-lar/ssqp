#ifndef SSQP_TYPES_HPP
#define SSQP_TYPES_HPP

#include <stdlib.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

#include <Eigen/Core>

namespace ssqp
{
using VecX  = Eigen::VectorXd;
using MatX  = Eigen::MatrixXd;
using DMatX = Eigen::DiagonalMatrix<double, -1>;

namespace model
{
struct Dimensions
{
    uint horizon;

    // Total values for states, controls and constraints
    uint state;
    uint control;
    uint constraint;

    // Total values per stage
    std::vector<uint> stage_state;
    std::vector<uint> stage_control;
    std::vector<uint> stage_constraint;
};
}

namespace solvers
{

// Flattened vector of all stage data
struct FlatVec
{
  public:
    FlatVec();
    // Initialize a 0 vector, with stage sizes given in dims
    FlatVec(const uint total_size, const std::vector<uint> &dims);
    // Initialize a flat vector using the vectors given in vectors
    FlatVec(const uint total_size, const std::vector<VecX> &vectors);
    // FlatVec
    FlatVec(const FlatVec &other, bool copy = true);

    Eigen::VectorBlock<VecX> stage(const uint k);

    const VecX stage(const uint k) const;

    VecX vec;

    FlatVec operator+(const FlatVec &p) const;

    FlatVec operator-(const FlatVec &p) const;

    FlatVec operator-() const;

    friend FlatVec operator*(double val, const FlatVec &p);

    FlatVec operator/(double val);

    FlatVec &operator+=(const FlatVec &p);

    friend std::ostream &operator<<(std::ostream &os, const FlatVec &v);

  protected:
    uint _max_dim;
    // Stage size vector
    std::vector<uint> _stage_sizes;
    // Accumulated indexing vector
    std::vector<uint> _acc_idx;
};

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
    model::Dimensions    dims;
    std::vector<QPStage> stages;

    MatX Qxx_t;
    VecX qx_t;
    MatX Dt;
    VecX ct;

    QPData(const model::Dimensions &dims);
};
}
}

#endif
