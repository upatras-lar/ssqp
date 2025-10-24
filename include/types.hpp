#ifndef SSQP_TYPES_HPP
#define SSQP_TYPES_HPP

#include <stdlib.h>
#include <memory>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <limits>

#include <Eigen/Core>

namespace ssqp
{
#ifdef INFO
#define LOG(...) (std::cout << __VA_ARGS__)
#else
#define LOG(...) ((void)0)
#endif

#ifdef DEBUG
#define DEBUG_LOG(...) (std::cout << __VA_ARGS__)
#define DEBUG_WAIT()   std::cin.get()
#else
#define DEBUG_LOG(...) ((void)0)
#define DEBUG_WAIT()   ((void)0)
#endif

using VecX  = Eigen::VectorXd;
using MatX  = Eigen::MatrixXd;
using DMatX = Eigen::DiagonalMatrix<double, -1>;

namespace model
{
using DimensionFunction  = std::function<uint()>;
using EvaluationFunction = std::function<VecX(const VecX &, const VecX &)>;
using GradientFunction =
  std::function<void(const VecX &, const VecX &, VecX &, VecX &)>;
using multiGradientFunction =
  std::function<void(const VecX &, const VecX &, MatX &, MatX &)>;
using HessianFunction = std::function<void(const VecX &, const VecX &, MatX &)>;
using HessprodFunction =
  std::function<void(const VecX &, const VecX &, const VecX &, MatX &)>;

inline std::function<void(const VecX &, const VecX &, MatX &)> ZeroHessian_xx =
  [](const VecX &x, const VecX &u, MatX &Hxx)
{ Hxx = MatX::Zero(x.rows(), x.rows()); };

inline std::function<void(const VecX &, const VecX &, MatX &)> ZeroHessian_xu =
  [](const VecX &x, const VecX &u, MatX &Hxu)
{ Hxu = MatX::Zero(x.rows(), u.rows()); };

inline std::function<void(const VecX &, const VecX &, MatX &)> ZeroHessian_uu =
  [](const VecX &x, const VecX &u, MatX &Huu)
{ Huu = MatX::Zero(u.rows(), u.rows()); };

inline std::function<void(const VecX &, const VecX &, const VecX &, MatX &)>
  ZeroHessprod_xx = [](const VecX &x, const VecX &u, const VecX &vec, MatX &Pxx)
{ Pxx = MatX::Zero(x.rows(), x.rows()); };

inline std::function<void(const VecX &, const VecX &, const VecX &, MatX &)>
  ZeroHessprod_xu = [](const VecX &x, const VecX &u, const VecX &vec, MatX &Pxu)
{ Pxu = MatX::Zero(x.rows(), u.rows()); };

inline std::function<void(const VecX &, const VecX &, const VecX &, MatX &)>
  ZeroHessprod_uu = [](const VecX &x, const VecX &u, const VecX &vec, MatX &Puu)
{ Puu = MatX::Zero(u.rows(), u.rows()); };

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

    // State + control
    std::vector<uint> stage_sc;
};
}

std::string format_double(double val, uint precision);

using ManifoldOperator = std::function<VecX(const VecX &, const VecX &)>;

// Operators for elements part of (composite) manifold states
struct ManifoldOperators
{
    ManifoldOperator right_plus;
    ManifoldOperator right_minus;
};

namespace solvers
{
// Flattened vector of all stage data
struct FlatVec
{
  public:
    // Initialize a 0 vector, with stage sizes given in dims
    FlatVec(const uint total_size, const std::vector<uint> &dims,
            const std::vector<ManifoldOperators> &ops = {});
    // Initialize a flat vector using the vectors given in vectors
    FlatVec(const uint total_size, const std::vector<VecX> &vectors,
            const std::vector<ManifoldOperators> &ops = {});
    // FlatVec copying
    // If copy is true, operators are also copied (argument ignored)
    FlatVec(const FlatVec &other, bool copy = true,
            const std::vector<ManifoldOperators> &ops = {});

    Eigen::VectorBlock<VecX> stage(const uint k);
    const VecX               stage(const uint k) const;
    const uint               stages() const;

    void set_ops(const std::vector<ManifoldOperators> &ops);

    VecX vec;

    void write_data(std::ofstream &stream);

    FlatVec  operator+(const FlatVec &p) const;
    FlatVec  operator-(const FlatVec &p) const;
    FlatVec  operator-() const;
    FlatVec  operator/(double val);
    FlatVec &operator+=(const FlatVec &p);

    friend FlatVec       operator*(double val, const FlatVec &p);
    friend std::ostream &operator<<(std::ostream &os, const FlatVec &v);

  protected:
    uint _max_dim;
    // Stage size vector
    std::vector<uint> _stage_sizes;
    // Accumulated indexing vector
    std::vector<uint> _acc_idx;
    // Manifold operators per stage
    std::vector<ManifoldOperators> _ops;
    // Whether custom operators exist
    bool _custom_ops;
    // Whether stages have different ops
    bool _per_stage_ops;
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
    VecX cl, cu;        // cl: (nc) x 1, cu: (nc x 1)
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
    VecX cl_t, cu_t;

    QPData(const model::Dimensions &dims);
};
}
}

#endif
