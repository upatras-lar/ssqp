#ifndef SSQP_SOLVERS_LQR_HPP
#define SSQP_SOLVERS_LQR_HPP

#include <vector>
#include <deque>
#include <exception>

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
    struct Params
    {
        double sigma = 10e-6;

        uint   solve_window        = 10;
        double solve_success_ratio = 0.9;

        double reg_init   = 1e-6;
        double reg_factor = 10.0;
        double reg_max    = 1e10;

        static Params Default()
        {
            return {};
        }
    };

    struct Output
    {
        FlatVec           x, u;
        std::vector<MatX> P, K;
        std::vector<VecX> p, k;
    };

    lqr(const QPData &qp, const admm_qp::Solution &current_sol,
        const FlatVec &rho, const Params &params = Params::Default());

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

    std::vector<MatX>             _G, _H;
    std::vector<Eigen::LLT<MatX>> _H_llt;
    const uint                    _solve_success_sma_k;
    const double                  _solve_success_sma_ratio;
    uint                          _solve_success_sum;
    std::deque<bool>              _solve_successes;

    const double _reg_init;
    const double _reg_factor;
    const double _reg_max;
    double       _reg;

    Output _output;

    void _solve();
    void _solve_cache_rho();
};
}
}

#endif