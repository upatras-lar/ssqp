#include "solvers/mpc.hpp"

namespace ssqp
{
namespace solvers
{
MPC::MPC(model::Model &model, const Params &params,
         const ssqp::Params &ssqp_params)
  : _model(model)
  , _ssqp_params(ssqp_params)
  , _keep_convergence_data(params.keep_convergence_data)
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

// Sets MPC warm-start to the given state and zero controls
void MPC::reinitialize(const VecX &state)
{
    const uint               H             = _model.horizon();
    const std::vector<uint> &control_sizes = _model.dims().stage_control;

    for (uint k = 0; k < H; ++k)
    {
        _model.setInitState(k, state);
        _model.setInitControl(k, VecX::Zero(control_sizes[k]));
    }

    _model.setInitState(H, state);
}

// Sets MPC warm-start to the given states and controls
void MPC::reinitialize(const FlatVec &states, const FlatVec &controls)
{
    const uint H = _model.horizon();

    for (uint k = 0; k < H; ++k)
    {
        _model.setInitState(k, states.stage(k));
        _model.setInitControl(k, controls.stage(k));
    }

    _model.setInitState(H, states.stage(H));
}

void MPC::reinitialize(const std::vector<VecX> &states,
                       const std::vector<VecX> &controls)
{
    const uint H = _model.horizon();

    for (uint k = 0; k < H; ++k)
    {
        _model.setInitState(k, states[k]);
        _model.setInitControl(k, controls[k]);
    }

    _model.setInitState(H, states[H]);
}

std::pair<FlatVec, FlatVec> MPC::step(const VecX &state)
{
    if (_prev_states != nullptr)
    {
        const uint H = _model.horizon();

        _model.setInitState(0, state);

        for (uint k = 0; k + 2 <= H; ++k)
        {
            _model.setInitState(k + 1, _prev_states->stage(k + 2));
            _model.setInitControl(k, _prev_controls->stage(k + 1));
        }

        _model.setInitState(
          H,
          _model.dynamics()[H - 1]->evaluate(_prev_states->stage(H),
                                             _prev_controls->stage(H - 1)));
        _model.setInitControl(H - 1, _prev_controls->stage(H - 1));
    }

    std::pair<FlatVec, FlatVec> p = step();
    _prev_states                  = std::make_unique<FlatVec>(p.first);
    _prev_controls                = std::make_unique<FlatVec>(p.second);

    return { p.first, p.second };
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
    _model.updateDimensions();
    ssqp        solver(_model, _ssqp_params);
    ssqp::Guess solution(_model.dims(), _model.linear_dims());
    SolverState state = solver.solve(solution);

    if (_keep_convergence_data)
    {
        _convergence_data.push_back(solver.convergenceData());
    }

    return { solution.x, solution.u };
}

const std::vector<ssqp::ConvergenceData> MPC::getConvergenceData() const
{
    if (!_keep_convergence_data)
    {
        logln_at(LogLevel::Warning,
                 "Warning: keep_convergence_data is false, so convergence data "
                 "will be empty!");
    }

    return _convergence_data;
}
}
}