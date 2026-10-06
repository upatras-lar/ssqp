---
hide:
  - navigation
---


# FAQ

Answers to common "how do I...?" questions, for getting things done quickly before diving into
[Concepts](concepts/model.md). Where the Python and C++ APIs differ, switch the tabs below to compare. If you haven't yet, read the [Quickstart](getting-started/quickstart.md)
first; these answers assume you already have an importable `ssqpy` (Python) or a linked `ssqp` (C++, see
[Installation](getting-started/installation.md)).

## How do I define a custom cost, dynamics, or constraint?

Subclass `StageCost`/`StageDynamics`/`StageConstraints` and override the abstract methods. Here's a minimal linear
dynamics example; see [Costs, Constraints & Dynamics](concepts/costs-constraints-dynamics.md) for the full
interface each one requires.

=== "Python"

    ```python
    import ssqpy
    import numpy as np


    class MyDynamics(ssqpy.model.dynamics.StageDynamics):
        def output_size(self):
            return 2

        def state_size(self):
            return 2

        def control_size(self):
            return 1

        def evaluate(self, x, u):
            return np.array([x[0] + x[1] * 0.1, x[1] + u[0] * 0.1])

        def gradient(self, x, u):
            dX = np.array([[1.0, 0.1], [0.0, 1.0]])
            dU = np.array([[0.0], [0.1]])
            return dX, dU
    ```

=== "C++"

    ```cpp
    #include "model/stagewise/StageDynamics.hpp"

    class MyDynamics : public ssqp::model::stagewise::StageDynamics
    {
      public:
        uint state_size() const override { return 2; }
        uint control_size() const override { return 1; }
        uint output_size() const override { return 2; }

        VecX evaluate(const VecX &x, const VecX &u) const override
        {
            VecX xn(2);
            xn << x(0) + x(1) * 0.1, x(1) + u(0) * 0.1;
            return xn;
        }

        void gradient(const VecX &x, const VecX &u, MatX &dX, MatX &dU) const override
        {
            dX = (MatX(2, 2) << 1.0, 0.1, 0.0, 1.0).finished();
            dU = (MatX(2, 1) << 0.0, 0.1).finished();
        }

        // Linear dynamics: all Hessian-vector products are zero.
        void hessprod_xx(const VecX &, const VecX &, const VecX &, MatX &Pxx) const override { Pxx.setZero(2, 2); }
        void hessprod_xu(const VecX &, const VecX &, const VecX &, MatX &Pxu) const override { Pxu.setZero(2, 1); }
        void hessprod_uu(const VecX &, const VecX &, const VecX &, MatX &Puu) const override { Puu.setZero(1, 1); }
    };
    ```

## How do I track a target pose vs. a target joint configuration?

Two different predefined costs cover these, and picking between them depends on what you actually care about
matching:

=== "Python"

    ```python
    # Joint-space: match a specific set of joint angles
    configuration_cost = ssqpy.model.costs.SquaredConfigurationErrorCost(model, desired_configuration, weight)
    model.addCost(k, configuration_cost)

    # Task-space: match a frame's position and orientation, regardless of joint angles
    pose_cost = ssqpy.model.costs.FrameSquaredPoseErrorCost(
        model, "frame_name", desired_pose, translation_weight, rotation_weight
    )
    model.addCost(k, pose_cost)
    ```

=== "C++"

    ```cpp
    #include "model/stagewise/predefined/Costs.hpp"

    model.addCost<ssqp::model::stagewise::predefined::costs::SquaredConfigurationErrorCost>(
        k, model, desired_configuration, weight);

    model.addCost<ssqp::model::stagewise::predefined::costs::FrameSquaredPoseErrorCost>(
        k, model, "frame_name", desired_pose, translation_weight, rotation_weight);
    ```

`SquaredConfigurationErrorCost` penalizes deviation from an exact set of joint angles; use it when the
configuration itself is what matters, as in [Pendubot](examples/pendubot.md)'s swing-up target.
`FrameSquaredPoseErrorCost` penalizes deviation of a named frame's pose (position and/or orientation) in task
space, ignoring which joint angles get there; use it for manipulation-style tasks specified by where something
should end up, as in [Panda Circle](examples/panda-circle-trajgen.md)'s end-effector circle.

The two aren't mutually exclusive: for a redundant arm (more joints than the 6 DOF a pose constrains), tracking
pose alone leaves the extra joint-space freedom unconstrained, which can drift over a long closed-loop run.
[Panda Circle (MPC)](examples/panda-circle-mpc.md#building-the-short-horizon-model) adds a small
`SquaredConfigurationErrorCost` toward a nominal configuration alongside the pose-tracking cost for exactly this
reason.

## How do I build a model from a URDF or MJCF file?

=== "Python"

    ```python
    model = ssqpy.model.Model(
        horizon=50,
        dt=0.01,
        urdf_path="robot.urdf",
        free_flyer=True,
    )
    model.finalize(stage_cost=my_stage_cost, terminal_cost=my_terminal_cost)
    ```

=== "C++"

    ```cpp
    #include "model/Model.hpp"

    ssqp::model::Model::Structure structure;
    structure.urdf_path  = "robot.urdf";
    structure.free_flyer = true;

    ssqp::model::Model model(50, 0.01, structure);

    ssqp::model::Model::Config config;
    config.stage_cost    = std::make_shared<MyStageCost>();
    config.terminal_cost = std::make_shared<MyTerminalCost>();
    model.finalize(config);
    ```

See [Model](concepts/model.md) for the full set of constructor and `finalize` arguments.

If your robot is one of the ones ssqp already bundles (Go2, A1, Panda, G1, Talos, or the double pendulum), you
can skip the file entirely and pass `from_preset` instead; see [Built-in presets](concepts/model.md#built-in-presets).

## How do I choose a good initial guess?

SSQP is a local solver: it improves on whatever guess it's given rather than searching globally, so the initial
guess affects both how fast it converges and which local solution it lands on. `init_states`/`init_controls`
passed to `finalize` do double duty: they fix $\boldsymbol{x}_0$ (which the solver cannot change) *and* seed the
guess for every other stage before the solver improves on it.

=== "Python"

    ```python
    model.finalize(init_states=[my_init_state], init_controls=[my_init_control])
    ```

=== "C++"

    ```cpp
    config.init_states   = {my_init_state};
    config.init_controls = {my_init_control};
    model.finalize(config);
    ```

There's no single right way to build one; which approach fits depends on the problem:

* No guess at all falls back to SSQP's own manifold-aware neutral configuration (e.g. angle $0$ on a joint's
  tangent/circle manifold rather than literal zero, which wouldn't be a valid point on it), which is enough for
  straightforward problems; see [Pendubot](examples/pendubot.md).
* A single configuration already close to the expected solution, repeated across every stage, helps a lot when you
  have domain knowledge about roughly where the solution should be; see
  [Panda Circle](examples/panda-circle-trajgen.md#building-the-model).
* For contact-rich problems, the *control* guess matters as much as the state one: seeding contact forces at a
  static-equilibrium value (e.g. each foot supporting its share of the robot's weight) avoids wasting early
  iterations discovering that a standing robot needs to push against the ground; see
  [Go2 Jump](examples/go2-jump.md#initial-state-and-a-static-equilibrium-initial-guess).

A guess that actively points the wrong way can be worse than no guess at all; see the swing-up scheduling trick in
[Pendubot](examples/pendubot.md#costs-and-why-the-tracking-cost-turns-on-late) for a case where the fix was
designing the costs around this rather than picking a better guess directly.

See [`finalize` arguments](concepts/model.md#finalize-arguments) in Model for the full set of initial-guess-related
options.

## How do I solve a trajectory optimization problem?

=== "Python"

    ```python
    solver = ssqpy.solvers.ssqp(model)
    state, x, u = solver.solve()

    x.print()
    u.print()
    ```

=== "C++"

    ```cpp
    #include "solvers/ssqp.hpp"

    ssqp::solvers::ssqp solver(model);

    ssqp::solvers::ssqp::Guess guess(model.dims(), model.linear_dims());
    ssqp::SolverState state = solver.solve(guess);

    std::cout << guess.x << std::endl;
    std::cout << guess.u << std::endl;
    ```

See [The SSQP Solver](concepts/solver.md) for termination criteria, line search and QP solver selection.

## How do I run MPC (real-time control)?

=== "Python"

    ```python
    mpc = ssqpy.solvers.MPC(model, sqp_iters=5, qp_iters=50)
    mpc.reinitialize(initial_state)

    while running:
        state = read_state()
        x, u = mpc.step(state)
        apply_controls(u.stage(0))
    ```

=== "C++"

    ```cpp
    #include "solvers/mpc.hpp"

    ssqp::solvers::MPC::Params mpc_params;
    mpc_params.sqp_iters = 5;
    mpc_params.qp_iters  = 50;

    ssqp::solvers::MPC mpc(model, mpc_params);
    mpc.reinitialize(initial_state);

    while (running)
    {
        VecX state  = read_state();
        auto [x, u] = mpc.step(state);
        apply_controls(u.stage(0));
    }
    ```

See [MPC](concepts/mpc.md) for warm-starting details and how `sqp_iters`/`qp_iters` bound per-step solve time.

## What happens if MPC fails to converge mid-run, and how do I detect it?

`mpc.step(...)` always returns an `(x, u)` trajectory, even if that step's SSQP iterations didn't fully converge
within the tight per-step budget (`sqp_iters`/`qp_iters`): it has to return *something* for the control loop to
apply, so it doesn't raise or signal failure on its own. To actually check whether a given step converged, keep
convergence data and inspect it after each step:

=== "Python"

    ```python
    mpc = ssqpy.solvers.MPC(model, sqp_iters=5, qp_iters=50, keep_convergence_data=True)

    while running:
        x, u = mpc.step(read_state())
        apply_controls(u.stage(0))

        data = mpc.getConvergenceData()[-1]
        if data.max_iter_reached and not data.loose_success:
            print("MPC step did not converge")
    ```

=== "C++"

    ```cpp
    ssqp::solvers::MPC::Params mpc_params;
    mpc_params.keep_convergence_data = true;
    ssqp::solvers::MPC mpc(model, mpc_params);

    while (running)
    {
        auto [x, u] = mpc.step(read_state());
        apply_controls(u.stage(0));

        const auto &data = mpc.getConvergenceData().back();
        if (data.max_iter_reached && !data.loose_success)
            std::cout << "MPC step did not converge" << std::endl;
    }
    ```

An occasional non-converged step is usually not fatal in a receding-horizon loop with good warm-starting (see
[Warm-starting](concepts/mpc.md#warm-starting) in MPC): the next step re-solves with fresh state feedback rather
than letting the error compound open-loop. A persistent run of failures is worth investigating the same way as an
offline non-converging solve (see
[How do I inspect solver convergence?](#how-do-i-inspect-solver-convergence-or-debug-why-it-isnt-converging)), and
is often a sign that `sqp_iters`/`qp_iters` is too tight for how far the state has drifted from the warm start, or
that the tracked target has left the problem's feasible region. Note this is a different failure mode from the
per-step solve simply running too *slowly* to keep up with real time. In those cases, trying different solver parameters
or attempting to simplify the model might instead be helpful. If that does not help or is not possible, the next solution
would be to use SSQPy's MPC as a high level plan, generated at a lower frequency and followed using a low-level controller
running at a higher frequency.

## How do I correct the controls between MPC steps?

When the control loop runs faster than MPC can solve, the first stage's controls go stale as the state moves away
from the one they were solved for. `mpc.feedbackGain(k)` returns the feedback gain $K_k$ of the last step at stage
$k$, which gives the change of the stage's controls for a change of its state. Apply it with the difference between
the measured state and the planned one, taken in the tangent space of the dynamics:

=== "Python"

    ```python
    x, u = mpc.step(read_state())
    K = mpc.feedbackGain(0)
    dynamics = model.getDynamics(0)

    while not solved_again():
        dx = dynamics.state_right_minus(read_state(), x.stage(0))
        apply_controls(u.stage(0) + K @ dx)
    ```

=== "C++"

    ```cpp
    auto [x, u] = mpc.step(read_state());
    const MatX K  = mpc.feedbackGain(0);
    const auto &dynamics = model.dynamics()[0];

    while (!solved_again())
    {
        const VecX dx = dynamics->state_right_minus(read_state(), x.stage(0));
        apply_controls(u.stage(0) + K * dx);
    }
    ```

The gain describes the last QP, so it only holds near the plan and for the plan's active constraints. With HPIPM
it is the Riccati gain of the last interior point iteration, where the barrier stiffens the active constraints (a
looser tolerance gives a smoother gain). With ADMM it is the gain of the final LQR, which penalizes inactive
constraints as well, so it is stiffer. The gains can be too aggressive to apply as they are when the loop adds
delay, so scaling them down may be needed.

## How do I switch between the ADMM and HPIPM QP solvers?

HPIPM requires building SSQP with `-DWITH_HPIPM=ON` (see
[Installation](getting-started/installation.md#cmake-options)), which is enabled by default.

=== "Python"

    ```python
    solver = ssqpy.solvers.ssqp(
        model,
        qp_solver_type=ssqpy.solvers.HPIPM,
        hpipm_mode=ssqpy.solvers.HPIPMMode.Speed,
    )
    ```

=== "C++"

    ```cpp
    ssqp::solvers::ssqp::Params params;
    params.qp_solver_type    = ssqp::solvers::ssqp::HPIPM;
    params.hpipm_params.mode = ssqp::solvers::SPEED;

    ssqp::solvers::ssqp solver(model, params);
    ```

See [QP Solvers](concepts/qp-solvers.md) for the tunable parameters of each backend.

## How do I add a contact constraint (e.g. for a robot foot)?

=== "Python"

    ```python
    model.addContactConstraint(k, "foot_frame", mu=0.6, max_normal_force=200.0)
    ```

=== "C++"

    ```cpp
    #include "model/stagewise/predefined/Constraints.hpp"

    namespace constraints = ssqp::model::stagewise::predefined::constraints;

    model.addConstraints<constraints::ContactConstraint>(k, model, "foot_frame", 0.6, 200.0);
    ```

The contact constraint restricts the contact force (friction cone), keeps the foot from penetrating the ground and,
by default, bounds the foot's linear velocity to zero so that a planted foot does not slide along the ground. Pass
`no_slip=False` to drop that velocity bound, e.g. when you want to bound or penalize the foot's motion yourself
instead.

In C++, the templated `addConstraints<C>` automatically creates and wires up the force the contact needs (it reads
`required_forces()`/`force_frame()` off the constraint); this is exactly the convenience the Python
`addContactConstraint` wraps. See [Contacts and forces](concepts/model.md#contacts-and-forces) in Model.

## How do I schedule contacts over a trajectory (a gait, a jump, ...)?

[How do I add a contact constraint?](#how-do-i-add-a-contact-constraint-eg-for-a-robot-foot) covers a single
contact at a single stage; a gait or a jump needs that decision to change per foot, per stage, over the whole
horizon. There isn't a dedicated scheduling API for this: it's the same `addContactConstraint`/`addConstraints`
calls as always, driven by a schedule you compute yourself and apply inside your own per-stage loop. Two ways to
turn a contact on/off for a given stage:

=== "Python"

    ```python
    for k in range(H):
        # Way 1: only add the constraints for the stages where the foot is grounded
        if in_stance(k, "foot_1"):
            model.addContactConstraint(k, "foot_1", mu=0.6, max_normal_force=200.0)

        # Way 2: add them at every stage, and disable them where the foot is airborne
        contact = model.addContactConstraint(k, "foot_2", mu=0.6, max_normal_force=200.0)
        if not in_stance(k, "foot_2"):
            contact.set_enabled(False)
    ```

=== "C++"

    ```cpp
    namespace constraints = ssqp::model::stagewise::predefined::constraints;

    for (uint k = 0; k < H; ++k)
    {
        // Way 1: only add the constraints for the stages where the foot is grounded
        if (in_stance(k, "foot_1"))
        {
            model.addConstraints<constraints::ContactConstraint>(k, model, "foot_1", 0.6, 200.0);
        }

        // Way 2: add them at every stage, and disable them where the foot is airborne
        auto contact = model.addConstraints<constraints::ContactConstraint>(k, model, "foot_2", 0.6, 200.0);
        if (!in_stance(k, "foot_2"))
        {
            contact->set_enabled(false);
        }
    }
    ```

The two approaches suit different situations: conditionally adding the contact constraint only inside the stages where a foot is grounded is simplest when the schedule is just a stage-index
check. `set_enabled` is more convenient when a constraint's applicability depends on something more complex than that, or when you want to
create the constraint object once upfront and reuse it. Neither approach requires all feet (or any other set of
contacts) to share the same schedule; per-foot, per-stage toggling is exactly what makes asymmetric gaits
representable with the same building blocks. Inside a fully airborne window, adding no contact constraints at all
for any foot lets the dynamics reduce to free-fall, which is usually what you want for a flight phase. See
[Go2 Jump](examples/go2-jump.md#the-per-stage-loop-stance-flight-stance) for a complete worked example, including
both toggling styles side by side.

## How do I choose a solver mode?

`ForwardDynamics`, `InverseDynamics`, and `ReducedInverseDynamics` all solve the same underlying problem, just
formulated differently (see [Solver mode](concepts/model.md#solver-mode) in Model for what each one actually
changes); which one converges best depends on the problem:

* **`ForwardDynamics`** (torque controls) is the most direct formulation and a reasonable default when there's no
  particular reason to prefer one of the others; see [Pendubot](examples/pendubot.md).
* **`InverseDynamics`** (acceleration controls) can ease convergence for fully-actuated systems: accelerations
  enter the dynamics linearly, so the QP built at each iteration is a better local approximation than it would be
  with torques, where that mapping is nonlinear in the configuration; see
  [Panda Circle](examples/panda-circle-trajgen.md#building-the-model).
* **`ReducedInverseDynamics`** (acceleration controls, unactuated DOFs propagated implicitly) is strictly an improvement
  over `InverseDynamics` for systems with manifold-valued states, and is used mainly for free-flyer
  systems, e.g. legged robots, where the base isn't directly actuated; the base's 6 degrees of freedom are
  enforced through the dynamics instead of being treated as free control inputs, which `InverseDynamics` alone
  wouldn't handle; see [Go2 Jump](examples/go2-jump.md).

If you're not sure and the problem is floating-base, `ReducedInverseDynamics` is worth trying first: across the
[Solver Mode benchmarks](benchmarks-solver-modes.md), it tends to converge in noticeably fewer iterations and less
time than the alternatives on legged problems specifically (see the
[general conclusions](benchmarks.md#general-conclusions) on the Benchmarks page). For a fully-actuated system,
comparing `ForwardDynamics` against `InverseDynamics` directly on your own problem is more reliable than assuming
one is always better; see the ["Things to try"](examples/panda-circle-trajgen.md#things-to-try) section of Panda
Circle for exactly this comparison.

See
[How do I change the solver mode or integration scheme?](#how-do-i-change-the-solver-mode-or-integration-scheme)
for the syntax to actually set it.

## How do I change the solver mode or integration scheme?

=== "Python"

    ```python
    model = ssqpy.model.Model(
        horizon, dt,
        solver_mode=ssqpy.model.SolverMode.InverseDynamics,
        integration=ssqpy.model.Integration.SemiImplicitEuler,
    )
    ```

=== "C++"

    ```cpp
    ssqp::model::Model::Structure structure;
    structure.solver_mode = ssqp::InverseDynamics;
    structure.integration = ssqp::SemiImplicitEuler;

    ssqp::model::Model model(horizon, dt, structure);
    ```

See [Solver mode](concepts/model.md#solver-mode) and
[Integration scheme](concepts/model.md#integration-scheme) in Model for what each option actually changes.

## How do I save a solved trajectory and load it back later?

=== "Python"

    ```python
    solver.write("trajectory.json")

    x, u = ssqpy.solvers.loadTrajectory("trajectory.json")
    ```

=== "C++"

    ```cpp
    solver.write("trajectory.json");

    auto [x, u] = ssqp::solvers::loadTrajectory("trajectory.json");
    ```

## How do I inspect solver convergence, or debug why it isn't converging?

=== "Python"

    ```python
    data = solver.getConvergenceData()
    data.print()
    ```

=== "C++"

    ```cpp
    const auto &data = solver.convergenceData();
    data.print();
    ```

`ConvergenceData`/`ssqpConvergenceData` includes per-iteration norms, step size and stagewise cost/gap/violation
breakdowns. See [Termination](concepts/solver.md#termination) in The SSQP Solver for what to look at first (e.g.
`first_violation_stage`, `norm_dx`/`norm_du` not decreasing, or `max_iter_reached`).

## How do I enable or disable a cost or constraint at runtime?

=== "Python"

    ```python
    my_cost.set_enabled(False)
    model.updateDimensions()  # only needed if a constraint's enabled state changed
    ```

=== "C++"

    ```cpp
    my_cost->set_enabled(false);
    model.updateDimensions();
    ```

Toggle the object you added to the model (the one `addCost`/`addConstraints` was given or returned). The entries
in `model.costs()`/`model.constraints()` are merged combinations of everything on a stage, and calling
`set_enabled` on one throws.

* **Costs:** nothing more is needed, since disabling a cost doesn't change any dimension. A stage always keeps at
  least one cost, so disabling every cost on a stage has no effect.
* **Constraints:** disabling one removes its rows from the stage's constraint vector, which changes the sizes in
  `model.dims()`/`model.linear_dims()`. Call `updateDimensions` after toggling a constraint on a finalized model, and
  construct a new solver afterwards, since a solver sizes its buffers when it is created. `solvers.MPC` does both for
  you: `step()` calls `updateDimensions` and rebuilds the solver only if the sizes changed. Toggling before
  `finalize` needs neither.

## How do I control SSQP's logging output?

SSQP's output is controlled by a single global mask of independent log levels, set with the functions below.

=== "Python"

    ```python
    # No output at all
    ssqpy.setSilentMode()

    # Only warnings and errors: silence everything, then add those levels back
    ssqpy.setSilentMode()
    ssqpy.setLogLevel(ssqpy.Warning | ssqpy.Error)

    # Or remove a single level from the current mask
    ssqpy.unsetLogLevel(ssqpy.Info)

    # Turn on the (very verbose) debug output
    ssqpy.setLogLevel(ssqpy.Debug)

    # Keep the iteration table, but with fewer columns
    solver.setReducedOutput(True)
    ```

=== "C++"

    ```cpp
    // No output at all
    ssqp::set_silent_mode();

    // Only warnings and errors: silence everything, then add those levels back
    ssqp::set_silent_mode();
    ssqp::set_log_level(ssqp::Warning | ssqp::Error);

    // Or remove a single level from the current mask
    ssqp::unset_log_level(ssqp::Info);

    // Turn on the (very verbose) debug output
    ssqp::set_log_level(ssqp::Debug);

    // Keep the iteration table, but with fewer columns
    solver.setReducedOutput(true);
    ```

`setLogLevel` *adds* levels to the mask; it doesn't replace it. `Info`, `Warning` and `Error` are on by default, so
`setLogLevel(Warning | Error)` on its own changes nothing, which is why the "only warnings and errors" example
silences everything first. Each level covers a different kind of output:

| Level | On by default | What it prints |
| --- | --- | --- |
| `Info` | yes | The SSQP iteration table (iteration, step norms, constraint violation, and, unless the output is reduced, the first violated stage, QP iterations, QP runtime and step size) and the total solve time |
| `Warning` | yes | Non-fatal problems: HPIPM returning a non-zero status, NaNs in the QP solution, one dynamics object shared by several stages, or asking an MPC for its convergence data when `keep_convergence_data` is off |
| `Error` | yes | Line search failure with regularization disabled |
| `Debug` | no | The initial guess, every iterate's states and controls, the QP matrices built for each stage, and HPIPM's status; this is a lot of output for anything beyond a tiny problem |

`setReducedOutput` only trims the iteration table; it doesn't change which levels are on. The mask is global to the
process, so it applies to every model and solver at once, and it only affects printing: the solve's
[convergence data](#how-do-i-inspect-solver-convergence-or-debug-why-it-isnt-converging) is recorded either way. To remove the logging from the
build altogether, configure with `-DDISABLE_LOGGING=ON` (see [Installation](getting-started/installation.md)).

## How do I visualize a solved trajectory?

The [`visualizer` example](examples.md) renders a written-out trajectory JSON against its URDF using
[viser](https://viser.studio/main/) (Python only):

```bash
cd examples
source setup_env.sh
python visualizer/visualize.py -u path/to/robot.urdf -i path/to/trajectory.json
```

## How do I use SSQP from C++ without the Python bindings at all?

Link the `ssqp` CMake target from your own project (`add_subdirectory`/`FetchContent`, with `BUILD_PYTHON=OFF` if
you don't need the bindings). See [Installation](getting-started/installation.md) for the CMake options and
[C++ API Reference](cpp-reference/index.md) for the full class-by-class interface.
