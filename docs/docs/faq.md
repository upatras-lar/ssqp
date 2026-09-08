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

## How do I switch between the ADMM and HPIPM QP solvers?

HPIPM requires building SSQP with `-DWITH_HPIPM=ON` (see
[Installation](getting-started/installation.md#cmake-options)).

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

    model.addConstraints<ssqp::model::stagewise::predefined::constraints::ContactConstraint>(
        k, model, "foot_frame", 0.6, 200.0);
    ```

In C++, the templated `addConstraints<C>` automatically creates and wires up the force the contact needs (it reads
`required_forces()`/`force_frame()` off the constraint); this is exactly the convenience the Python
`addContactConstraint` wraps. See [Contacts and forces](concepts/model.md#contacts-and-forces) in Model.

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

## How do I control SSQP's logging output?

=== "Python"

    ```python
    ssqpy.setSilentMode()

    # or, more selectively:
    ssqpy.setLogLevel(ssqpy.Warning | ssqpy.Error)
    ```

=== "C++"

    ```cpp
    ssqp::set_silent_mode();

    // or, more selectively:
    ssqp::set_log_level(ssqp::Warning | ssqp::Error);
    ```

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
