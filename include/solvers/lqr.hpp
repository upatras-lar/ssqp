#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "solvers/admm_qp.hpp"

namespace ssqp
{
namespace solvers
{
namespace lqr
{
struct Output
{
    MatX              x, u;
    std::vector<MatX> P, K;
    std::vector<VecX> p, k;
};

Output lqr(const QPData &qp, const admm_qp::Solution &current_sol,
           const std::vector<DMatX> &rho, const double sigma = 10e-6);
}
}
}