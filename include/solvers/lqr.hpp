#ifndef SSQP_LQR_HPP
#define SSQP_LQR_HPP

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
    FlatVec           x, u;
    std::vector<MatX> P, K;
    std::vector<VecX> p, k;
};

Output lqr(const QPData &qp, const admm_qp::Solution &current_sol,
           const FlatVec &rho, const double sigma = 10e-6,
           bool constrained = true);
}
}
}

#endif