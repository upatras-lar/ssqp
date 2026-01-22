#include "tests/include/ADMMInfeasibility.hpp"

namespace ssqp
{
namespace tests
{
ADMMInfeasibility::ADMMInfeasibility(const std::string &name)
  : Test(name)
  , _model(100, 0.01,
           model::Model::Structure { "examples/urdf/panda/panda.urdf" })
{
    Vec3 lb(0.1, 0.1, 0.1);
    Vec3 ub(-0.1, -0.1, -0.1);

    for (uint k = 0; k < 100; ++k)
    {
        _model.addCost<model::stagewise::predefined::costs::SquaredControlCost>(
          k,
          9,
          1e-3);

        _model.addConstraints<
          model::stagewise::predefined::constraints::FramePositionBound>(
          k,
          _model,
          "panda_ee",
          lb,
          ub,
          9,
          9);
    }

    _model.addCost<model::stagewise::predefined::costs::SquaredJointAngleCost>(
      100,
      9,
      9,
      1e-1);

    _model.finalize();
}

bool ADMMInfeasibility::run()
{
    ssqp::solvers::ssqp::Params params;
    params.max_iter  = 1;
    params.tolerance = 1e-2;

    ssqp::solvers::admm_qp::Params admm_params;
    admm_params.abs_tolerance         = 1e-3;
    admm_params.rel_tolerance         = 1e-3;
    admm_params.primal_infs_tolerance = 1e-4;
    admm_params.max_iter              = 1000;

    ssqp::solvers::ssqp        solver(_model, params, admm_params);
    ssqp::solvers::ssqp::Guess solution(_model.dims(), _model.linear_dims());
    ssqp::solvers::SolverState state = solver.solve(solution);

    return state == ssqp::solvers::SolverState::PRIMAL_INFEASIBLE;
}
}
}