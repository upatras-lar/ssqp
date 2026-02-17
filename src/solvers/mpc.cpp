#include "solvers/mpc.hpp"

namespace ssqp
{
namespace solvers
{
MPC::MPC(model::Model &model, const Params &params,
         const ssqp::Params &ssqp_params)
  : _model(model)
  , _ssqp_params(ssqp_params)
{
    _ssqp_params.max_iter = params.sqp_iters;

    switch (_ssqp_params.qp_solver_type)
    {
    case ssqp::QPSolverType::ADMM:
        _ssqp_params.admm_params.max_iter = params.qp_iters;
        break;
    case ssqp::QPSolverType::HPIPM:
        _ssqp_params.hpipm_params.max_iter = params.qp_iters;
        break;
    }

    if (!model.finalized())
    {
        model.finalize();
    }
}

VecX MPC::step(const VecX &state)
{
    if (_prev_states != nullptr)
    {
        const uint H = _model.horizon();

        _model.setInitState(0, state);

        for (uint k = 1; k < H; ++k)
        {
            _model.setInitState(k, _prev_states->stage(k + 1));
            _model.setInitControl(k - 1, _prev_controls->stage(k));
        }

        _model.setInitControl(H - 1, _prev_controls->stage(H - 1));
    }

    std::pair<FlatVec, FlatVec> p = step();
    _prev_states                  = std::make_unique<FlatVec>(p.first);
    _prev_controls                = std::make_unique<FlatVec>(p.second);

    return _prev_controls->stage(0);
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
    ssqp        solver(_model, _ssqp_params);
    ssqp::Guess solution(_model.dims(), _model.linear_dims());
    SolverState state = solver.solve(solution);

    return { solution.x, solution.u };
}
}
}