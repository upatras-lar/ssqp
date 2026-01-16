#ifndef SSQP_SOLVERS_MPC_HPP
#define SSQP_SOLVERS_MPC_HPP

#include "types.hpp"
#include "model/Model.hpp"

namespace ssqp
{
namespace solvers
{
class MPC
{
  public:
    struct Params
    {
        uint sqp_iters  = 10;
        uint admm_iters = 50;

        static Params Default()
        {
            return {};
        }
    };

    MPC(model::Model &model, const Params &params = Params::Default());

    // Receding horizon initialization
    VecX step(const VecX &state);

    std::pair<FlatVec, FlatVec> step(const FlatVec &states,
                                     const FlatVec &controls);

    std::pair<FlatVec, FlatVec> step(const std::vector<VecX> &states,
                                     const std::vector<VecX> &controls);

    std::pair<FlatVec, FlatVec> step();

  protected:
    Params        _params;
    model::Model &_model;

    std::unique_ptr<FlatVec> _prev_states = nullptr, _prev_controls = nullptr;
};
}
}

#endif
