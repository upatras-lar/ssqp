#ifndef SSQP_SOLVERS_ADMM_QP_HPP
#define SSQP_SOLVERS_ADMM_QP_HPP

#include "solvers/qp.hpp"

#include "nlohmann/json.hpp"

namespace ssqp
{
namespace solvers
{
class admm_qp : public QPSolver
{
  public:
    struct Params
    {
        Params() = default;
        Params(const nlohmann::ordered_json &admmj);

        // Parameters

        // Initial rho value
        double init_rho = 0.1;
        // Over-relaxation parameter
        double alpha = 1.6;
        // Proximal regularization parameter
        double sigma = 1e-6;

        // Rho update

        // Whether to perform rho update
        bool rho_update = true;
        // Minimum possible rho value
        double rho_min = 1e-6;
        // Maximum possible rho value
        double rho_max = 1e8;
        // Factor by which rho must change to actually perform rho update
        double adaptive_rho_tolerance = 5;
        // Number of iterations to wait before updating rho
        // Also affects checking for termination
        uint rho_update_iter_freq = 25;

        // Termination parameters

        // Maximum number of ADMM iterations
        uint max_iter = 1'000;
        // Absolute ADMM tolerance
        double abs_tolerance = 1e-3;
        // Final absolute ADMM tolerance
        // This value will be reached after `tolerance_update_steps` ADMM steps
        // Use -1.0 to ignore
        double abs_tolerance_final = -1.0;
        // Relative ADMM tolerance
        double rel_tolerance = 1e-4;
        // Final relative ADMM tolerance
        // This value will be reached after `tolerance_update_steps` ADMM steps
        // Use -1.0 to ignore
        double rel_tolerance_final = -1.0;
        // Number of steps to go from initial to final tolerances
        // Updates are done linearly per step
        // Ignored (per tolerance) if tolerance is -1.0
        uint   tolerance_update_steps = 0;
        double primal_infs_tolerance  = 1e-6; // Primal infeasibility tolerance

        // Whether to warm start ADMM with the unconstrained solution
        // Note: set to false if using Inverse Dynamics
        bool warm_start = true;
        // Whether to reset the rho parameter between each new ADMM formulation
        // Note: setting to false may often cause oscillations
        bool reset_rho = true;
        // Whether to reset the y parameter between each new ADMM formulation
        // Note: setting to false may often cause oscillations
        bool reset_y = true;

        static Params Default()
        {
            return {};
        }

        void print() const;

        void write(nlohmann::ordered_json &j) const;
    };

    admm_qp(const QPData &qp, const Params &params = Params::Default());

    void reset() override;

    void increaseRegularization() override;

    SolverState solve(QPSolver::Solution &sol) override;

    uint iterations() const override;

  protected:
    const QPData      &_qp;
    Params             _params;
    QPSolver::Solution _solution;

    FlatVec _z, _y;

    FlatVec      _rho, _rho_inv;
    double       _rho_common;
    const double _rho_min = 1e-6;
    bool         _rho_changed;

    // Norms of residuals
    double _norm_primal, _norm_dual;
    double _norm_primal_rel, _norm_dual_rel;

    // Primal infeasibility criteria
    double _primal_infs_norm, _primal_infs_support;
    double _primal_infs_tol_scaled;

    uint _iter;

    uint _reg_incr;

    double _abs_tolerance_init;
    double _rel_tolerance_init;

    // Calculates residual norms
    // r_prim = Dx - z
    // r_dual = Q(z - z_prev)
    void _calculate_residuals(const FlatVec &x_prev, const FlatVec &u_prev,
                              const FlatVec &z_prev, const FlatVec &y_prev);

    SolverState _check_termination() const;

    void _update_rho();

    void _step(const FlatVec &dx_hat, const FlatVec &du_hat);
};
}
}

#endif
