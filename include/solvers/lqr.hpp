#ifndef SSQP_SOLVERS_LQR_HPP
#define SSQP_SOLVERS_LQR_HPP

#include <vector>

#include <Eigen/Dense>

#include "types.hpp"
#include "solvers/admm_qp.hpp"

namespace ssqp
{
namespace solvers
{
class lqr
{
  public:
    struct Output
    {
        FlatVec           x, u;
        std::vector<MatX> P, K;
        std::vector<VecX> p, k;
    };

    lqr(const QPData &qp, const admm_qp::Solution &current_sol,
        const FlatVec &rho, const double sigma = 10e-6);

    // Returns output for its structure
    Output output_structure();

    Output unconstrained();

    // No rho change
    Output constrained(const admm_qp::Solution &current_sol);
    // Rho change
    Output constrained(const admm_qp::Solution &current_sol,
                       const FlatVec           &rho);

  protected:
    QPData _qp;

    std::vector<MatX> _Qxx, _Qxu, _Quu;
    std::vector<VecX> _qx, _qu;
    MatX              _Qt;
    VecX              _qt;

    FlatVec _rho;
    double  _sigma;

    std::vector<MatX>              _G, _H;
    std::vector<Eigen::LDLT<MatX>> _H_ldlt;

    Output _output;

    void _solve();
    void _solve_cache_rho();
};
}
}

#endif