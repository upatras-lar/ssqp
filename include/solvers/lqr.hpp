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

        double reg_init   = 1e-5;
        double reg_factor = 10.0;
        double reg_max    = 1e4;

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
    Output &output_structure();

    Output &unconstrained();

    // No rho change
    Output &constrained(const admm_qp::Solution &current_sol, const FlatVec &z,
                        const FlatVec &y);
    // Rho change
    Output &constrained(const admm_qp::Solution &current_sol, const FlatVec &z,
                        const FlatVec &y, const FlatVec &rho);

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
    std::vector<MatX>             _cache_A_BK;

    const uint       _solve_success_sma_k;
    const double     _solve_success_sma_ratio;
    uint             _solve_success_sum;
    std::deque<bool> _solve_successes;

    const double _reg_init;
    const double _reg_factor;
    const double _reg_max;
    double       _reg;

    Output _output;

    void _solve();
    void _solve_cache_rho();

    // Temporaries to prevent aliasing for Eigen operations
    struct
    {
        VecX c1, c2;
        MatX CX1;
        MatX CU1;
        VecX x1, x2, x3;
        VecX u1, u2;
        MatX XX1, XX2, XX3;
        MatX UU1;
        MatX XU1;
        MatX UX1;
    } _tmp;
};
}
}

#endif