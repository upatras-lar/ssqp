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

VecX MPC::step(VecX current_state)
{
    for (uint k = 0; k < _model.horizon(); ++k)
    {
        _model.setInitState(k, current_state);
        _model.setInitControl(k, VecX::Zero(_model.dims().stage_control[k]));
    }

    ssqp::Params ssqp_params;
    ssqp_params.max_iter = _params.sqp_iters;

    admm_qp::Params admm_params;
    admm_params.max_iter = _params.admm_iters;

    ssqp        solver(_model, ssqp_params, admm_params);
    ssqp::Guess solution(_model.dims(), _model.linear_dims());
    SolverState state = solver.solve(solution);

    return solution.u.stage(0);
}
}
}