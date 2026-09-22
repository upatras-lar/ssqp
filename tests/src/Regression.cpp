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
        require(
          state == solvers::SUCCESS,
          "SSQP must converge (" + solvers::to_string(state) + ", iterations=" +
            std::to_string(solver.convergenceData().iterations) + ", gap=" +
            std::to_string(solver.convergenceData().trajectory_stage_gap) +
            ", constraint=" +
            std::to_string(
              solver.convergenceData().trajectory_constraint_violation) +
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
    auto model  = integratorProblem(2, 1.0, true);
    auto bounds = model->constraints();
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
            for (uint k = 0; k < 2; ++k) bounds[k]->set_enabled(enabled);
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

// Tests the initialization behavior of Config::init_controls (empty ->
// zero, size 1 -> broadcast to every stage, size horizon -> one per stage)
// against distinct dynamics objects per stage
void finalizeInitControlsContract()
{
    const uint horizon = 3;

    auto build = [&](const std::vector<VecX> &init_controls)
    {
        model::Model::Config config;
        for (uint k = 0; k <= horizon; ++k)
            config.dynamics.push_back(std::make_shared<Integrator>());
        config.stage_cost    = std::make_shared<Quadratic>();
        config.terminal_cost = std::make_shared<Quadratic>();
        config.init_states   = { VecX::Constant(1, 1.0) };
        config.init_controls = init_controls;
        return std::make_shared<model::Model>(horizon, 1.0, config);
    };

    auto zero_model = build({});
    for (uint k = 0; k < horizon; ++k)
        requireClose(zero_model->dynamics()[k]->init_control(),
                     VecX::Zero(1),
                     "An empty init_controls must default to zero at stage " +
                       std::to_string(k),
                     0,
                     0);

    const VecX shared_control  = VecX::Constant(1, 0.42);
    auto       broadcast_model = build({ shared_control });
    for (uint k = 0; k < horizon; ++k)
        requireClose(broadcast_model->dynamics()[k]->init_control(),
                     shared_control,
                     "A single init_controls entry must broadcast to stage " +
                       std::to_string(k),
                     0,
                     0);

    std::vector<VecX> per_stage_controls;
    for (uint k = 0; k < horizon; ++k)
        per_stage_controls.push_back(VecX::Constant(1, 0.1 * (k + 1)));
    auto per_stage_model = build(per_stage_controls);
    for (uint k = 0; k < horizon; ++k)
        requireClose(per_stage_model->dynamics()[k]->init_control(),
                     per_stage_controls[k],
                     "A per-stage init_controls vector must set stage " +
                       std::to_string(k),
                     0,
                     0);
}

// Verify dDifference against numerical differentiation of difference() for both
// a model without an XML description (Euclidean subtraction) and an XML-based
// one (manifold difference via Pinocchio)
void dDifferenceMatchesFiniteDifferenceOfDifference()
{
    auto check = [](const model::Model &model, const VecX &q0, const VecX &q1)
    {
        const uint n  = static_cast<uint>(q0.size());
        MatX       J0 = MatX::Zero(n, n), J1 = MatX::Zero(n, n);
        model.dDifference(q0, q1, J0, 0);
        model.dDifference(q0, q1, J1, 1);

        for (uint j = 0; j < n; ++j)
        {
            VecX fd0 = finiteDifferenceVector(
              [&](double h)
              {
                  VecX q0p = q0, q0m = q0;
                  q0p[j] += h;
                  q0m[j] -= h;
                  return VecX(
                    (model.difference(q0p, q1) - model.difference(q0m, q1)) /
                    (2 * h));
              },
              1e-4);
            VecX fd1 = finiteDifferenceVector(
              [&](double h)
              {
                  VecX q1p = q1, q1m = q1;
                  q1p[j] += h;
                  q1m[j] -= h;
                  return VecX(
                    (model.difference(q0, q1p) - model.difference(q0, q1m)) /
                    (2 * h));
              },
              1e-4);
            requireClose(J0.col(j),
                         fd0,
                         "d(difference)/dq0, column " + std::to_string(j),
                         1e-6,
                         1e-6);
            requireClose(J1.col(j),
                         fd1,
                         "d(difference)/dq1, column " + std::to_string(j),
                         1e-6,
                         1e-6);
        }
    };

    model::Model custom_model(1, 0.01);
    check(custom_model,
          (VecX(3) << 1.0, -2.0, 0.5).finished(),
          (VecX(3) << 0.2, 0.3, -0.7).finished());

    model::Model::Structure structure;
    structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    model::Model xml_model(1, 0.01, structure);
    check(xml_model,
          VecX::Constant(xml_model.nq(), 0.3),
          VecX::Constant(xml_model.nq(), -0.2));
}

// Evaluate a torque constraint at, just inside, and just outside +-tau_bound
void innateTorqueBoundsAreEnforced()
{
    using namespace model::stagewise::predefined::costs;

    model::Model::Structure structure;
    structure.urdf_path = "examples/assets/urdf/panda/panda.urdf";
    auto model          = std::make_shared<model::Model>(1, 0.01, structure);

    for (uint k = 0; k <= model->dims().horizon; ++k)
        model->addCost<SquaredJointVelocityCost>(k, *model, 1.0);

    // Widen the position and velocity bounds so the zero configuration used
    // below is unconditionally feasible for those parts of the constraint,
    // isolating the check to the torque part
    const double         tau_bound = 3.5;
    model::Model::Config config;
    config.custom_lower_position_bounds =
      VecX::Constant(model->njoints(), -100.0);
    config.custom_upper_position_bounds =
      VecX::Constant(model->njoints(), 100.0);
    config.custom_velocity_bounds = VecX::Constant(model->njoints(), 100.0);
    config.custom_torque_bounds   = VecX::Constant(model->njoints(), tau_bound);
    model->finalize(config);

    auto constraint = model->constraints()[0];
    require(constraint != nullptr, "Non-terminal stage has innate constraints");

    const VecX &lb        = constraint->lower_bound();
    const VecX &ub        = constraint->upper_bound();
    auto        satisfies = [&](const VecX &c)
    {
        return (c.array() >= lb.array() - 1e-9).all() &&
               (c.array() <= ub.array() + 1e-9).all();
    };

    const VecX x = VecX::Zero(model->nq() + model->nv());

    require(satisfies(
              constraint->evaluate(x, VecX::Constant(model->nu(), tau_bound))),
            "A control at +tau_bound must satisfy the torque constraint");
    require(satisfies(
              constraint->evaluate(x, VecX::Constant(model->nu(), -tau_bound))),
            "A control at -tau_bound must satisfy the torque constraint");
    require(
      !satisfies(
        constraint->evaluate(x, VecX::Constant(model->nu(), 1.1 * tau_bound))),
      "A control above +tau_bound must violate the torque constraint");
    require(
      !satisfies(
        constraint->evaluate(x, VecX::Constant(model->nu(), -1.1 * tau_bound))),
      "A control below -tau_bound must violate the torque constraint");
}

// A preset must build the same model as loading its URDF by path directly
void modelFromPresetMatchesUrdfPath()
{
    model::Model::Structure preset_structure;
    preset_structure.from_preset = model::Preset::Go2;
    model::Model preset_model(1, 0.01, preset_structure);

    model::Model::Structure path_structure;
    path_structure.urdf_path  = "examples/assets/urdf/go2/go2.urdf";
    path_structure.free_flyer = true;
    model::Model path_model(1, 0.01, path_structure);

    require(preset_model.nq() == path_model.nq(),
            "Preset model nq must match the equivalent urdf_path model");
    require(preset_model.nv() == path_model.nv(),
            "Preset model nv must match the equivalent urdf_path model");
}

// from_preset is mutually exclusive with urdf_path
void modelFromPresetRejectsConflictingFields()
{
    model::Model::Structure structure;
    structure.from_preset = model::Preset::Go2;
    structure.urdf_path   = "examples/assets/urdf/go2/go2.urdf";

    bool threw = false;
    try
    {
        model::Model model(1, 0.01, structure);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    require(threw,
            "from_preset combined with urdf_path must throw invalid_argument");
}

// A free-flying body with no joints and no linear forces can only be rotated
// via a wrench's moment; solving to a target orientation is a direct
// end-to-end check that the solver actually exploits wrench dynamics, not
// just that the dynamics gradient is correct in isolation
void wrenchRotatesFloatingBody()
{
    using namespace model::stagewise::predefined::costs;

    const uint   H  = 10;
    const double dt = 0.05;

    model::Model::Structure structure;
    structure.urdf_path   = "examples/assets/urdf/floating_body/floating_body.urdf";
    structure.free_flyer  = true;
    structure.solver_mode = model::SolverMode::ForwardDynamics;

    auto model = std::make_shared<model::Model>(H, dt, structure);

    for (uint k = 0; k < H; ++k)
    {
        model->addWrench(k, "base_link");
        model->addCost<SquaredControlCost>(k, *model, 1e-3);
        model->addCost<SquaredForceCost>(k, *model, 1e-3);
    }

    Mat3 desired_rotation =
      Eigen::AngleAxisd(M_PI / 2.0, Vec3::UnitZ()).toRotationMatrix();
    model->addCost<FloatingBaseSquaredRotationErrorCost>(H,
                                                         *model,
                                                         desired_rotation,
                                                         50.0);

    VecX init_state = VecX::Zero(model->nq() + model->nv());
    init_state[6]   = 1.0; // Identity quaternion (x, y, z, w)

    model::Model::Config config;
    config.init_states = { init_state };
    model->finalize(config);

    solvers::ssqp::Params params;
    params.max_iter             = 100;
    params.tolerance            = 1e-3;
    params.constraint_tolerance = 1e-4;
    params.loose_success        = true;

    solvers::ssqp        solver(*model, params);
    solvers::ssqp::Guess guess(model->dims(), model->linear_dims());
    solver.reinitialize();

    const auto state = solver.solve(guess);
    require(state == solvers::SUCCESS,
            "SQP must converge for the wrench-only rotation problem (" +
              solvers::to_string(state) + ")");

    const VecX final_state = guess.x.stage(H);
    Eigen::Quaterniond q_final(final_state[6],
                               final_state[3],
                               final_state[4],
                               final_state[5]);
    Mat3 R_final = q_final.toRotationMatrix();

    double cos_angle =
      ((R_final.transpose() * desired_rotation).trace() - 1.0) / 2.0;
    cos_angle          = std::clamp(cos_angle, -1.0, 1.0);
    double angle_error = std::acos(cos_angle);

    require(angle_error < 0.1,
            "Body must rotate to within ~5.7 degrees of the target "
            "orientation using only wrench torque (no joints, no linear "
            "forces); got " +
              std::to_string(angle_error) + " rad error");
}
}
void addQPRegressionTests(TestContainer &tests);
void addRegressionTests(TestContainer &tests)
{
    tests.addTest(std::make_shared<FunctionTest>("Regression_FlatVecOwnership",
                                                 flatVecOwnership));
    tests.addTest(std::make_shared<FunctionTest>("Regression_QuadraticHessians",
                                                 quadraticHessians));
    tests.addTest(
      std::make_shared<FunctionTest>("Regression_Finalize_InitControlsContract",
                                     finalizeInitControlsContract));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_DDifference_MatchesFiniteDifference",
      dDifferenceMatchesFiniteDifferenceOfDifference));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_InnateTorqueBounds_AreEnforced",
      innateTorqueBoundsAreEnforced));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_ModelFromPreset_MatchesUrdfPath",
      modelFromPresetMatchesUrdfPath));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_ModelFromPreset_RejectsConflictingFields",
      modelFromPresetRejectsConflictingFields));
    tests.addTest(std::make_shared<FunctionTest>(
      "Regression_WrenchRotatesFloatingBody", wrenchRotatesFloatingBody));
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
