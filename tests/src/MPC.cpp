#include "MPC.hpp"

#include <exception>

#include "model/stagewise/predefined/Constraints.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "solvers/mpc.hpp"

namespace ssqp
{
namespace tests
{
MPCStepTest::MPCStepTest(const std::string                &name,
                         const solvers::ssqp::QPSolverType qp_solver_type)
  : Test(name)
  , _qp_solver_type(qp_solver_type)
{
}

bool MPCStepTest::run()
{
    ssqp::set_silent_mode();
    try
    {
        constexpr uint horizon = 3;

        model::Model::Structure structure;
        structure.urdf_path =
          "examples/assets/urdf/double_pendulum/pendubot.urdf";
        structure.solver_mode     = model::SolverMode::ForwardDynamics;
        structure.actuated_joints = { 0 };

        auto model = std::make_shared<model::Model>(horizon, 0.01, structure);

        for (uint k = 0; k < horizon; ++k)
        {
            model->addCost<
              model::stagewise::predefined::costs::SquaredControlCost>(k,
                                                                       *model,
                                                                       1e-3);
            model->addCost<
              model::stagewise::predefined::costs::SquaredJointVelocityCost>(
              k,
              *model,
              1e-3);
        }
        model->addCost<
          model::stagewise::predefined::costs::SquaredJointVelocityCost>(
          horizon,
          *model,
          1e-3);

        auto control_constraint = model->addConstraints<
          model::stagewise::predefined::constraints::JointControlConstraint>(
          0,
          *model,
          VecX::Constant(model->nu(), -1.0),
          VecX::Constant(model->nu(), 1.0));

        model->finalize();

        solvers::MPC::Params mpc_params;
        mpc_params.sqp_iters             = 1;
        mpc_params.qp_iters              = 5;
        mpc_params.keep_convergence_data = true;

        solvers::ssqp::Params ssqp_params;
        ssqp_params.qp_solver_type            = _qp_solver_type;
        ssqp_params.tolerance                 = 1e-2;
        ssqp_params.admm_params.abs_tolerance = 1e-3;
        ssqp_params.admm_params.rel_tolerance = 1e-3;

        solvers::MPC mpc(*model, mpc_params, ssqp_params);
        VecX state = VecX::Zero(model->dims().stage_state.front());
        state[0] = 0.1;
        state[1] = -0.2;
        mpc.reinitialize(state);

        const auto valid_solution =
          [horizon, &state](
            const std::pair<solvers::FlatVec, solvers::FlatVec> &solution)
        {
            return solution.first.stages() == horizon + 1 &&
                   solution.second.stages() == horizon &&
                   (solution.first.stage(0) - state).norm() < 1e-10 &&
                   solution.first.vec.array().isFinite().all() &&
                   solution.second.vec.array().isFinite().all();
        };

        if (!valid_solution(mpc.step(state)))
        {
            _failed_logs.push_back(
              "First MPC step returned an invalid solution\n");
            return false;
        }

        state[0] += 0.03;
        if (!valid_solution(mpc.step(state)))
        {
            _failed_logs.push_back(
              "Second MPC step returned an invalid solution\n");
            return false;
        }

        // Changing the enabled state changes the QP dimensions. The next step
        // must rebuild the persistent solver before solving the new QP.
        control_constraint->set_enabled(false);
        if (!valid_solution(mpc.step(state)))
        {
            _failed_logs.push_back("MPC step after a dimension change returned "
                                   "an invalid solution\n");
            return false;
        }

        if (mpc.getConvergenceData().size() != 3)
        {
            _failed_logs.push_back(
              "MPC convergence data did not record all three steps\n");
            return false;
        }

        return true;
    }
    catch (const std::exception &exception)
    {
        _failed_logs.push_back(std::string("MPC step threw an exception: ") +
                               exception.what() + "\n");
        return false;
    }
}
}
}
