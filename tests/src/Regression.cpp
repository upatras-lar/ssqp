#include "Regression.hpp"
#include "LinearProblem.hpp"
#include "solvers/mpc.hpp"
#include "model/stagewise/predefined/Costs.hpp"
#include "model/stagewise/predefined/Constraints.hpp"
#include <limits>

namespace ssqp
{
namespace tests
{
namespace
{
void solverKnownSolution(solvers::ssqp::QPSolverType backend, uint horizon,
                         bool bounded)
{
    auto                 model  = integratorProblem(horizon, 1.0, bounded);
    auto                 params = accurateParams(backend);
    solvers::ssqp        solver(*model, params);
    solvers::ssqp::Guess guess(model->dims(), model->linear_dims());
    for (double initial : { 1.0, -1.0, 0.7 })
    {
        for (uint k = 0; k <= horizon; ++k)
            model->setInitState(k, VecX::Constant(1, initial));
        for (uint k = 0; k < horizon; ++k)
            model->setInitControl(k, VecX::Constant(1, 0.15));
        solver.reinitialize();
        const auto state = solver.solve(guess);
        require(state == solvers::SUCCESS,
                "SSQP must converge (" + solvers::to_string(state) +
                  ", iterations=" +
                  std::to_string(solver.convergenceData().iterations) +
                  ", gap=" +
                  std::to_string(solver.convergenceData().trajectory_stage_gap) +
                  ", constraint=" +
                  std::to_string(solver.convergenceData().trajectory_constraint_violation) +
                  ")");
        auto expected = integratorSolution(horizon, initial);
        if (bounded)
        {
            // H=2: both controls saturate for these initial states.
            double u = initial > 0 ? -0.2 : 0.2;
            expected.second.setConstant(u);
            for (uint k = 0; k <= horizon; ++k)
                expected.first[k] = initial + k * u;
        }
        checkTrajectory(guess.x, guess.u, expected.first, expected.second);
        const auto cd = solver.convergenceData();
        require(std::abs(cd.trajectory_cost -
                         0.5 * (expected.first.squaredNorm() +
                                expected.second.squaredNorm())) < 2e-5,
                "Reported objective");
        require(cd.trajectory_stage_gap < 2e-6 &&
                  cd.trajectory_constraint_violation < 2e-6,
                "Reported feasibility");
        require(cd.iteration_data.size() == cd.iterations,
                "Iteration history must describe this solve");
        require(guess.lambda.vec.allFinite() && guess.mu.vec.allFinite(),
                "Finite multipliers");
    }
}
void mpcKnownSolution(solvers::ssqp::QPSolverType backend, uint horizon)
{
    auto                 model  = integratorProblem(horizon, 0.0);
    auto                 params = accurateParams(backend);
    solvers::MPC::Params mp;
    mp.sqp_iters             = 20;
    mp.qp_iters              = 10000;
    mp.keep_convergence_data = true;
    solvers::MPC mpc(*model, mp, params);
    uint         calls = 0;
    for (double initial : { 1.0, -0.8, 0.4, 1.2 })
    {
        const auto expected = integratorSolution(horizon, initial);
        auto       result   = mpc.step(VecX::Constant(1, initial));
        checkTrajectory(result.first,
                        result.second,
                        expected.first,
                        expected.second);
        require(mpc.getConvergenceData().size() == ++calls,
                "One convergence record per step");
        require(mpc.getConvergenceData().back().iteration_data.size() <=
                  mp.sqp_iters,
                "No accumulated iterations");
    }
    // Exercise all initialization and step overloads with deliberately different
    // nonoptimal guesses. Compare the result to the same independent solution.
    std::vector<VecX> x(horizon + 1, VecX::Constant(1, -0.6));
    std::vector<VecX> u(horizon, VecX::Constant(1, 0.1));
    auto              expected = integratorSolution(horizon, -0.6);
    auto              result   = mpc.step(x, u);
    checkTrajectory(result.first,
                    result.second,
                    expected.first,
                    expected.second);
    solvers::FlatVec fx(horizon + 1, x), fu(horizon, u);
    result = mpc.step(fx, fu);
    checkTrajectory(result.first,
                    result.second,
                    expected.first,
                    expected.second);
    mpc.reinitialize(x, u);
    result = mpc.step();
    checkTrajectory(result.first,
                    result.second,
                    expected.first,
                    expected.second);
    mpc.reinitialize(fx, fu);
    result = mpc.step();
    checkTrajectory(result.first,
                    result.second,
                    expected.first,
                    expected.second);
    mpc.reinitialize(VecX::Constant(1, 0.9));
    result   = mpc.step();
    expected = integratorSolution(horizon, 0.9);
    checkTrajectory(result.first,
                    result.second,
                    expected.first,
                    expected.second);
    require(mpc.getConvergenceData().size() == calls + 5,
            "History for every overload");
}
void mpcDimensionChanges(solvers::ssqp::QPSolverType backend)
{
    auto                 model  = integratorProblem(2, 1.0, true);
    auto                 bounds = model->constraints();
    // Toggle components of merged constraints, as in robot MPC applications.
    for (uint k = 0; k < 2; ++k) model->addConstraints<ControlBound>(k, 2.0);
    model->updateDimensions();
    auto                 params = accurateParams(backend);
    solvers::MPC::Params mp;
    mp.sqp_iters = 20;
    mp.qp_iters  = 10000;
    // Repeated destruction and rebuilding is also exercised under LeakSanitizer.
    for (int lifetime = 0; lifetime < 8; ++lifetime)
    {
        solvers::MPC mpc(*model, mp, params);
        for (int cycle = 0; cycle < 6; ++cycle)
        {
            bool enabled = cycle % 2 == 0;
            for (uint k = 0; k < 2; ++k)
                bounds[k]->set_enabled(enabled);
            // Cover callers that update dimensions themselves before step().
            if (cycle % 3 == 0) model->updateDimensions();
            mpc.reinitialize(VecX::Ones(1));
            auto result   = mpc.step();
            auto expected = integratorSolution(2, 1.0);
            if (enabled)
            {
                expected.first << 1, 0.8, 0.6;
                expected.second << -0.2, -0.2;
            }
            checkTrajectory(result.first,
                            result.second,
                            expected.first,
                            expected.second);
            require(model->linear_dims().constraint == (enabled ? 4u : 2u),
                    "Constraint dimensions updated");
        }
        require(mpc.getConvergenceData().empty(), "History disabled");
    }
}
void flatVecOwnership()
{
    const std::vector<uint> sizes { 2, 0, 3, 1 };
    solvers::FlatVec        original(6, sizes);
    original.vec << 1, 2, 3, 4, 5, 6;
    require(original.stage(1).size() == 0, "Empty stage");
    requireClose(original.stage(2),
                 (VecX(3) << 3, 4, 5).finished(),
                 "Stage offsets",
                 0,
                 0);
    solvers::FlatVec copy(original);
    copy.stage(2).setConstant(-1);
    require(original.vec[2] == 3, "Copy must not alias original");
    solvers::FlatVec assigned(1, std::vector<uint> { 1 });
    assigned = original;
    original.vec.setZero();
    require(assigned.stages() == 4 && assigned.stage(3)[0] == 6,
            "Assignment copies shape and storage");
    solvers::FlatVec scratch(assigned, false);
    require(scratch.vec.isZero() && scratch.stages() == 4,
            "Shape-only copy initializes zeros");
    requireClose((-assigned).vec, -assigned.vec, "Negation", 0, 0);
    requireClose((2.0 * assigned).vec, 2.0 * assigned.vec, "Scaling", 0, 0);
}
void quadraticHessians()
{
    model::Model::Structure structure;
    structure.urdf_path = "examples/assets/urdf/double_pendulum/pendubot.urdf";
    auto model          = std::make_shared<model::Model>(2, 0.01, structure);
    using namespace model::stagewise::predefined::costs;
    for (uint k = 0; k <= 2; ++k)
    {
        model->addCost<SquaredJointAngleCost>(k, *model, 2.3);
        model->addCost<SquaredJointVelocityCost>(k, *model, 0.7);
        if (k < 2) model->addCost<SquaredControlCost>(k, *model, 1.9);
    }
    model->finalize();
    for (uint k : { 0u, 2u })
    {
        VecX x    = (VecX(4) << 0.4, -0.7, 1.2, -0.6).finished();
        VecX u    = VecX::Constant(k == 2 ? 0 : model->nu(), 0.3);
        auto cost = model->costs()[k];
        MatX hxx, hxu, huu;
        cost->hessian_xx(x, u, hxx);
        cost->hessian_xu(x, u, hxu);
        cost->hessian_uu(x, u, huu);
        MatX fdxx(x.size(), x.size()), fdxu(x.size(), u.size()),
          fduu(u.size(), u.size());
        constexpr double h = 1e-5;
        for (int j = 0; j < x.size(); ++j)
        {
            VecX xp = x, xm = x, dxp, dxm, dup, dum;
            xp[j] += h;
            xm[j] -= h;
            cost->gradient(xp, u, dxp, dup);
            cost->gradient(xm, u, dxm, dum);
            fdxx.col(j) = (dxp - dxm) / (2 * h);
        }
        for (int j = 0; j < u.size(); ++j)
        {
            VecX up = u, um = u, dxp, dxm, dup, dum;
            up[j] += h;
            um[j] -= h;
            cost->gradient(x, up, dxp, dup);
            cost->gradient(x, um, dxm, dum);
            fdxu.col(j) = (dxp - dxm) / (2 * h);
            fduu.col(j) = (dup - dum) / (2 * h);
        }
        requireClose(hxx, fdxx, "Cost Hxx", 1e-8, 1e-8);
        requireClose(hxu, fdxu, "Cost Hxu", 1e-8, 1e-8);
        requireClose(huu, fduu, "Cost Huu", 1e-8, 1e-8);
        requireClose(hxx, hxx.transpose(), "Hxx symmetry", 1e-12, 0);
    }
}
}
void addQPRegressionTests(TestContainer &tests);
void addRegressionTests(TestContainer &tests)
{
    tests.addTest(std::make_shared<FunctionTest>("Regression_FlatVecOwnership",
                                                 flatVecOwnership));
    tests.addTest(std::make_shared<FunctionTest>("Regression_QuadraticHessians",
                                                 quadraticHessians));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_NumericalChecks",
      []
      {
          VecX v = VecX::Ones(2), bad = v;
          bad[0] = std::numeric_limits<double>::quiet_NaN();
          require(!close(bad, v), "NaN must fail comparisons");
          bad[0] = std::numeric_limits<double>::infinity();
          require(!close(bad, v), "Infinity must fail comparisons");
          bad     = v;
          bad[1] += 1e-3;
          require(!close(bad, v), "Incorrect gradient must fail");
          double d =
            finiteDifference([](double h) { return std::abs(1e-6 + h); }, 1e-3);
          require(std::abs(d - 1.0) < 1e-7, "Refine across an abs kink");
      }));
    std::vector<solvers::ssqp::QPSolverType> backends { solvers::ssqp::ADMM };
#ifdef WITH_HPIPM
    backends.push_back(solvers::ssqp::HPIPM);
#endif
    for (auto backend : backends)
    {
        std::string suffix = backend == solvers::ssqp::ADMM ? "ADMM" : "HPIPM";
        for (uint horizon : { 1u, 2u, 8u })
        {
            tests.addTest(std::make_shared<FunctionTest>(
              "Regression_SSQP_Known_H" + std::to_string(horizon) + "_" +
                suffix,
              [=] { solverKnownSolution(backend, horizon, false); }));
            tests.addTest(std::make_shared<FunctionTest>(
              "Regression_MPC_Known_H" + std::to_string(horizon) + "_" + suffix,
              [=] { mpcKnownSolution(backend, horizon); }));
        }
        tests.addTest(std::make_shared<FunctionTest>(
          "Regression_SSQP_ActiveBounds_" + suffix,
          [=] { solverKnownSolution(backend, 2, true); }));
        tests.addTest(std::make_shared<FunctionTest>(
          "Regression_MPC_DimensionChanges_" + suffix,
          [=] { mpcDimensionChanges(backend); }));
    }
    addQPRegressionTests(tests);
}
}
}
