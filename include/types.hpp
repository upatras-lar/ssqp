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
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ssqp
{
#ifdef INFO
#define LOG(...)   (std::cout << __VA_ARGS__)
#define LOGLN(...) (std::cout << __VA_ARGS__ << std::endl)
#else
#define LOG(...)   ((void)0)
#define LOGLN(...) ((void)0)
#endif

#ifdef DEBUG
#define DEBUG_LOG(...)   (std::cout << __VA_ARGS__)
#define DEBUG_LOGLN(...) (std::cout << __VA_ARGS__ << std::endl)
#define DEBUG_WAIT()     std::cin.get()
#else
#define DEBUG_LOG(...)   ((void)0)
#define DEBUG_LOGLN(...) ((void)0)
#define DEBUG_WAIT()     ((void)0)
#endif

using Scalar = Eigen::Vector<double, 1>;
using Vec2   = Eigen::Vector2d;
using Vec3   = Eigen::Vector3d;
using Vec4   = Eigen::Vector4d;
using Vec6   = Eigen::Vector<double, 6>;
using VecX   = Eigen::VectorXd;

using Mat3  = Eigen::Matrix3d;
using Mat4  = Eigen::Matrix<double, 4, 4>;
using Mat6  = Eigen::Matrix<double, 6, 6>;
using MatX  = Eigen::MatrixXd;
using DMatX = Eigen::DiagonalMatrix<double, -1>;

void print_matrix_dimensions(const MatX &mat);

namespace model
{
// TODO: These use x.rows() and x.cols()
// But if n_x != n_{dx}, they will fail
inline auto ZeroHessian_xx = [](const VecX &x, const VecX &u, MatX &Hxx)
{ Hxx = MatX::Zero(x.rows(), x.rows()); };

inline auto ZeroHessian_xu = [](const VecX &x, const VecX &u, MatX &Hxu)
{ Hxu = MatX::Zero(x.rows(), u.rows()); };

inline auto ZeroHessian_uu = [](const VecX &x, const VecX &u, MatX &Huu)
{ Huu = MatX::Zero(u.rows(), u.rows()); };

inline auto ZeroHessprod_xx =
  [](const VecX &x, const VecX &u, const VecX &vec, MatX &Pxx)
{ Pxx = MatX::Zero(x.rows(), x.rows()); };

inline auto ZeroHessprod_xu =
  [](const VecX &x, const VecX &u, const VecX &vec, MatX &Pxu)
{ Pxu = MatX::Zero(x.rows(), u.rows()); };

inline auto ZeroHessprod_uu =
  [](const VecX &x, const VecX &u, const VecX &vec, MatX &Puu)
{ Puu = MatX::Zero(u.rows(), u.rows()); };

inline auto DefaultRightMinus = [](const VecX &x, const VecX &y)
{ return x - y; };

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
    std::vector<uint> nq, nv, njoints;
};

// An external force on some frame
struct Force
{
    std::string frame;
    uint        force_index;

    Force() = default;
    Force(const std::string &frame, const uint force_index);
};
}

std::string format_double(double val, uint precision);

template<typename ManifoldOperator1, typename ManifoldOperator2>
struct ManifoldOperators
{
    ManifoldOperator1 right_plus;
    ManifoldOperator2 right_minus;
};

namespace solvers
{
// Flattened vector of all stage data
struct FlatVec
{
  public:
    // Initialize a 0 vector, with stage sizes given in dims
    FlatVec(const uint total_size, const std::vector<uint> &dims);
    // Initialize a flat vector using the vectors given in vectors
    FlatVec(const uint total_size, const std::vector<VecX> &vectors);
    // FlatVec copying
    // If copy is true, operators are also copied (argument ignored)
    FlatVec(const FlatVec &other, bool copy = true);

    Eigen::VectorBlock<VecX> stage(const uint k);
    const VecX               stage(const uint k) const;
    const uint               stages() const;

    FlatVec operator-() const;
    FlatVec operator/(double val);

    FlatVec operator-(const FlatVec &p) const;

    VecX vec;

    friend FlatVec       operator*(double val, const FlatVec &p);
    friend std::ostream &operator<<(std::ostream &os, const FlatVec &v);

  protected:
    uint _max_dim;
    // Stage size vector
    std::vector<uint> _stage_sizes;
    // Accumulated indexing vector
    std::vector<uint> _acc_idx;
};

FlatVec       operator*(double val, const FlatVec &p);
std::ostream &operator<<(std::ostream &os, const FlatVec &v);

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

/* Holds data for non-terminal stages of the below QP
 * Note that all matrix or vector sizes refer to tangent space
 */
struct QPStage
{
    MatX A, B;  // A: $n_x \times n_x$, B: $n_x \times n_u$
    VecX gamma; // $n_x \times 1$
    MatX Qxx, Qxu,
      Quu; // Qxx: $n_x \times n_x$, Qxu: $n_x \times n_u$, Quu: $n_u \times n_u$
    VecX qx, qu; // qx: $n_x \times 1$, qu: $n_u \times 1$
    MatX D, E;   // D: $n_c \times n_x$, E: $n_c \times n_u$
    VecX cl, cu; // cl: $n_c \times 1$, cu: $n_c \times 1$
};

/* Defines a QP of the following form, to be solved using ADMM
$$\Large
\begin{equation}
\begin{split}
&\min_{\boldsymbol{x}, \boldsymbol{u}} \sum_{k=0}^{H-1} (
\frac{1}{2} \Bigl[\begin{smallmatrix}\boldsymbol{x}_k \\ \boldsymbol{u}_k\end{smallmatrix}\Bigr]^\top
\Bigl[\begin{smallmatrix}Q_{xx} \ Q_{xu} \\ Q_{ux} \ Q_{uu}\end{smallmatrix}\Bigr]
\Bigl[\begin{smallmatrix}\boldsymbol{x}_k \\ \boldsymbol{u}_k\end{smallmatrix}\Bigr] 
+
\Bigl[\begin{smallmatrix}\boldsymbol{q}_k \\ \boldsymbol{r}_k\end{smallmatrix}\Bigr]^\top
\Bigl[\begin{smallmatrix}\boldsymbol{x}_k \\ \boldsymbol{u}_k\end{smallmatrix}\Bigr] )
+
\frac{1}{2}\boldsymbol{x}_H^\top Q_H \boldsymbol{x}_H
+
\boldsymbol{x}_H^\top q_H
\\
&+
\sum_{k=0}^{H-1} (
\frac{\rho}{2} \norm{D_k \boldsymbol{x}_k + E_k \boldsymbol{u}_k - \boldsymbol{z}_k^j + \rho^{-1} \boldsymbol{y}_k^j}_2^2 )
+
\frac{\rho}{2} \norm{D_H \boldsymbol{x}_H - \boldsymbol{z}_H^j + \rho^{-1} \boldsymbol{y}_H^j}_2^2
\\
&+
\sum_{k=0}^{H} ( \frac{\sigma}{2} \norm{\boldsymbol{x}_k - \boldsymbol{x}_k^j}_2^2 )
+
\sum_{k=0}^{H-1} ( \frac{\sigma}{2} \norm{\boldsymbol{u}_k - \boldsymbol{u}_k^j}_2^2 )
\\
&\text{s.t.} \boldsymbol{x}_{k+1} = A_k \boldsymbol{x}_k + B_k \boldsymbol{u}_k + \gamma_k 
\end{split}
\end{equation}
$$
 * Terms with (^j) are constant parts of the current ADMM guess
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
