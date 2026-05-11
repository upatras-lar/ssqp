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
#include <random>
#include <cassert>

#include <Eigen/Core>
#include <Eigen/Dense>

namespace ssqp
{
/* The available log levels
 * Each is used for a different class of output, and is independent of the 
 * others
 */
enum LogLevel : int
{
    Info    = 1 << 0,
    Debug   = 1 << 1,
    Warning = 1 << 2,
    Error   = 1 << 3,
};

inline uint GlobalLogMask = 1;

inline void set_silent_mode()
{
    GlobalLogMask = 0;
}

inline void set_log_level(const LogLevel &log_level)
{
    GlobalLogMask |= log_level;
}

inline void set_log_level(const uint log_level_mask)
{
    GlobalLogMask |= log_level_mask;
}

inline void unset_log_level(const LogLevel &log_level)
{
    GlobalLogMask &= ~log_level;
}

inline void unset_log_level(const uint log_level_mask)
{
    GlobalLogMask &= ~log_level_mask;
}

#ifndef DISABLE_LOGGING
#define log_at(level, ...)                                                 \
    (level & GlobalLogMask ? (void)(std::cout << __VA_ARGS__) : ((void)0))
#define logln_at(level, ...)                                               \
    (level & GlobalLogMask ? (void)(std::cout << __VA_ARGS__ << std::endl) \
                           : ((void)0))
#define log_wait_at(level)                                       \
    (level & GlobalLogMask ? (void)(std::cin.get()) : ((void)0))
#else
#define log_at(level, ...)   ((void)0)
#define logln_at(level, ...) ((void)0)
#define log_wait_at(level)   ((void)0)
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

void   print_matrix_dimensions(const MatX &mat);
int    sgn(double val);
double softmax_max(const VecX &v, double alpha);

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

    // URDF-specific values
    uint nq;           // Configuration size
    uint nv;           // Velocity size
    uint nu;           // Non-force control size
    uint njoints;      // Number of actuated joints
    uint total_joints; // Number of total joints
};

// An external force on some frame
struct Force
{
    std::string frame;
    uint        force_index;

    Force() = default;
    Force(const std::string &frame, const uint force_index);
};

/* Integration Scheme
 * The type of integration used for the system dynamics
 * Options:
 *  - ExplicitEuler:
 *      $\boldsymbol{x}_{n+1}=\boldsymbol{x}_n+\boldsymbol{v}_ndt$
 *      $\boldsymbol{v}_{n+1}=\boldsymbol{v}_n+\boldsymbol{a}_ndt$
 *  - SemiImplicitEuler:
 *      $\boldsymbol{x}_{n+1}=\boldsymbol{x}_n+\boldsymbol{v}_{n+1}dt$
 *      $\boldsymbol{v}_{n+1}=\boldsymbol{v}_n+\boldsymbol{a}_ndt$
 */
enum IntegrationScheme
{
    ExplicitEuler     = 0,
    SemiImplicitEuler = 1
};

/* Solver Mode
 * The mode the solver uses for the system formulation, corresponding to the
 * equality constraint of the NLP
 * This can either be full dynamics, or a kinematics formulation
 * Options:
 *  - Kinematics: The solver optimizes in a kinematic mode. This means there are
 *    no system dynamics. Instead, the generated controls correspond to velocities,
 *    while the state corresponds to the system configuration.
 *  - ForwardDynamics: The solver optimizes with a forward dynamics approach. The 
 *    state corresponds to a stacked vector of the system configuration and velocity.
 *    The generated controls correspond to torques which, when applied over time
 *    to the system, generate the output trajectory.
 *  - InverseDynamics: The solver optimizes with an inverse dynamics approach. The 
 *    state corresponds to a stacked vector of the system configuration and velocity.
 *    The generated controls correspond to accelerations. This simplifies the
 *    LQR formulation and can sometimes lead to more robust solutions. Inverse
 *    dynamics is used as a constraint, ensuring the torques producing the
 *    generated acceleration controls are within the required effort limits.
 *  - ReducedInverseDynamics: This mode is similar to the Inverse dynamics mode. In 
 *    this mode, the floating base (if present) and/or some subset of the model's
 *    joints are unactuated. While regular Inverse mode enforces this via
 *    equality constraints, this mode instead enforces this implicitly through
 *    state propagation.
 */
enum SolverMode
{
    Kinematics             = 0,
    ForwardDynamics        = 1,
    InverseDynamics        = 2,
    ReducedInverseDynamics = 3,
};

inline const std::unordered_map<SolverMode, std::string> DM_to_str = {
    {             SolverMode::Kinematics,             "Kinematics" },
    {        SolverMode::ForwardDynamics,        "ForwardDynamics" },
    {        SolverMode::InverseDynamics,        "InverseDynamics" },
    { SolverMode::ReducedInverseDynamics, "ReducedInverseDynamics" }
};

inline const std::unordered_map<std::string, SolverMode> str_to_DM = {
    {             "Kinematics",             SolverMode::Kinematics },
    {        "ForwardDynamics",        SolverMode::ForwardDynamics },
    {        "InverseDynamics",        SolverMode::InverseDynamics },
    { "ReducedInverseDynamics", SolverMode::ReducedInverseDynamics }
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

    // Return a stage vector reference for setting
    Eigen::VectorBlock<VecX> stage(const uint k);
    // Return a const stage vector for reading
    const VecX stage(const uint k) const;
    // Return the total number of stages
    const uint stages() const;

    FlatVec operator-() const;
    FlatVec operator/(double val);

    FlatVec operator-(const FlatVec &p) const;

    VecX vec;

    friend FlatVec       operator*(double val, const FlatVec &p);
    friend std::ostream &operator<<(std::ostream &os, const FlatVec &v);

    void setPrintColumns(const uint cols);
    uint printColumns() const;

  protected:
    uint _max_dim;
    // Stage size vector
    std::vector<uint> _stage_sizes;
    // Accumulated indexing vector
    std::vector<uint> _acc_idx;

    uint _print_cols = 10;
};

FlatVec       operator*(double val, const FlatVec &p);
std::ostream &operator<<(std::ostream &os, const FlatVec &v);

enum SolverState
{
    SUCCESS            = 1,
    LOOSE_SUCCESS      = 2,
    CONSTRAINT_SUCCESS = 3,
    PRIMAL_INFEASIBLE  = 10,
    DUAL_INFEASIBLE    = 11,
    MAX_ITER_REACHED   = 13,
    ONGOING            = 100,
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
