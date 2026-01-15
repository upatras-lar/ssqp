#include "solvers/mpc.hpp"
#include "solvers/ssqp.hpp"
#include "solvers/admm_qp.hpp"

namespace ssqp
{
namespace solvers
{
MPC::MPC(model::Model &model, const Params &params)
  : _model(model)
  , _params(params)
{
    if (!model.finalized())
    {
        model.finalize();
    }
}

std::pair<FlatVec, FlatVec> MPC::step(const std::vector<VecX> &states,
                                      const std::vector<VecX> &controls)
{
    const uint H = _model.horizon();

    for (uint k = 0; k < H; ++k)
    {
        _model.setInitState(k, states[k]);
        _model.setInitControl(k, controls[k]);
    }
    _model.setInitState(H, states[H]);

    return step();
}

std::pair<FlatVec, FlatVec> MPC::step(const FlatVec &states,
                                      const FlatVec &controls)
{
    const uint H = _model.horizon();

    for (uint k = 0; k < H; ++k)
    {
        _model.setInitState(k, states.stage(k));
        _model.setInitControl(k, controls.stage(k));
    }
    _model.setInitState(H, states.stage(H));

    return step();
}

std::pair<FlatVec, FlatVec> MPC::step()
{
    ssqp::Params ssqp_params;
    ssqp_params.max_iter = _params.sqp_iters;

    admm_qp::Params admm_params;
    admm_params.max_iter = _params.admm_iters;

    ssqp        solver(_model, ssqp_params, admm_params);
    ssqp::Guess solution(_model.dims(), _model.linear_dims());
    SolverState state = solver.solve(solution);

    return { solution.x, solution.u };
}
}
}