#ifndef SSQP_SOLVERS_PIQP_QP_HPP
#define SSQP_SOLVERS_PIQP_QP_HPP

#include "solvers/qp.hpp"

#include "nlohmann/json.hpp"

namespace ssqp
{
namespace solvers
{

class piqp_data;

class piqp_qp : public QPSolver
{
  public:
    struct Params
    {
        Params() = default;
        Params(const nlohmann::ordered_json &piqpj);

        // Maximum iterations
        int max_iter = 100;

        double eps_abs = 1e-3;
        double eps_rel = 1e-3;

        double rho_init   = 1e-6;
        double delta_init = 1e-4;

        // Regularization term added to the Hessian diagonal
        // Keeps steps bounded along directions with little or no curvature,
        // which PIQP would otherwise solve exactly
        double reg_prim = 1e-4;

        // Parallel multistage solver flag, requires building with
        // SSQP_PIQP_OPENMP
        bool parallel = false;
        // Threads of the parallel solver, 0 uses the OpenMP default
        uint num_threads = 0;

        static Params Default()
        {
            return {};
        }

        void print() const;

        void write(nlohmann::ordered_json &j) const;
    };

    piqp_qp(const QPData &qp, const Params &params = Params::Default());
    ~piqp_qp();

    piqp_qp(const piqp_qp &)            = delete;
    piqp_qp &operator=(const piqp_qp &) = delete;

    void        reset() override;
    void        increaseRegularization() override;
    SolverState solve(QPSolver::Solution &sol) override;
    uint        iterations() const override;

    // Not available, since PIQP does not expose a Riccati gain
    MatX feedbackGain(const uint k) const override;

  protected:
    const QPData &_qp;
    Params        _params;
    Solution      _solution;

    double       _reg;
    const double _reg_max = 1e8;

    int  _iter        = 0;
    bool _qp_uploaded = false;
    bool _setup_done  = false;

    // Sparse problem data, solver and variable/row offsets
    std::shared_ptr<piqp_data> _piqp_data;
};
}
}

#endif
