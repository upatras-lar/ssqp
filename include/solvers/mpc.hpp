#ifndef SSQP_SOLVERS_MPC_HPP
#define SSQP_SOLVERS_MPC_HPP

#include <memory>

#include "types.hpp"
#include "model/Model.hpp"
#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"
#include "solvers/hpipm_qp.hpp"

namespace ssqp
{
namespace solvers
{

/** MPC class
 * Helper class which utilizes SSQP to perform MPC for real-time SSQP-based 
 * control.
 * 
 * Repeatedly uses SSQP with low iterations to solve a problem for the current 
 * state of a system, applies the first of the generated controls, updates
 * the initial guess using feedback from the system and a warm start using the
 * previous solution  and repeats the process.
 */
class MPC
{
  public:
    // MPC-specific parameters
    struct Params
    {
        // Upper limit on SSQP iterations
        uint sqp_iters = 10;
        // Upper limit on QP solver iterations
        uint qp_iters = 50;

        // Whether to save iteration data
        bool keep_convergence_data = false;

        static Params Default()
        {
            return {};
        }
    };

    // Initialize an MPC helper for the given model, the given MPC and SSQP
    // parameters
    // Note that the max iterations used for SSQP and the QP solver will be
    // determined by the MPC parameters only.
    MPC(model::Model &model, const Params &params = Params::Default(),
        const ssqp::Params &ssqp_params = ssqp::Params::Default());

    // Sets MPC warm-start to the given state and zero controls
    void reinitialize(const VecX &state);

    // Sets MPC warm-start to the given states and controls
    void reinitialize(const FlatVec &states, const FlatVec &controls);
    void reinitialize(const std::vector<VecX> &states,
                      const std::vector<VecX> &controls);

    // Receding horizon initialization step
    // Uses the provided state as feedback and the previous solution for
    // warm-starting
    std::pair<FlatVec, FlatVec> step(const VecX &state);

    // Uses the provided states and controls for warm-starting
    std::pair<FlatVec, FlatVec> step(const FlatVec &states,
                                     const FlatVec &controls);

    // Uses the provided states and controls for warm-starting
    std::pair<FlatVec, FlatVec> step(const std::vector<VecX> &states,
                                     const std::vector<VecX> &controls);

    // Performs an MPC step with the current guess as-is
    std::pair<FlatVec, FlatVec> step();

    // Returns stored convergence data
    const std::vector<ssqp::ConvergenceData> getConvergenceData() const;

  protected:
    ssqp::Params          _ssqp_params;
    model::Model         &_model;
    std::unique_ptr<ssqp> _solver;
    model::Dimensions     _solver_dims, _solver_linear_dims;

    std::unique_ptr<FlatVec> _prev_states = nullptr, _prev_controls = nullptr;

    const bool                         _keep_convergence_data;
    std::vector<ssqp::ConvergenceData> _convergence_data;
};
}
}

#endif
