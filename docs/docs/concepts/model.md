# Model

A `Model` (`ssqpy.model.Model` in Python, `ssqp::model::Model` in C++) is a prerequisite for solving an OCP with
SSQP. It bundles, over an optimization horizon $H$:

* a [cost function](costs-constraints-dynamics.md#costs) for each stage, $\ell_k(\boldsymbol{x}, \boldsymbol{u})$,
* [dynamics functions](costs-constraints-dynamics.md#dynamics) for each stage, $\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u})$, and
* [constraint functions](costs-constraints-dynamics.md#constraints) for each stage, $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u})$ (optional; a stage may have none).

All stage cost and dynamics functions must be defined; constraints may be left empty.

## Horizon, states and controls

A model optimizes over $H$ steps while treating the initial state $\boldsymbol{x}_0$ as fixed (only the initial
*controls* can be altered). This means there are $H$ total controls ($\boldsymbol{u}_0$ to
$\boldsymbol{u}_{H-1}$) and $H + 1$ total states ($\boldsymbol{x}_0$ to $\boldsymbol{x}_H$). At the terminal
stage $H$ there are no controls, so stagewise functions must be written to ignore $\boldsymbol{u}$ at that stage.
Predefined functions already account for this.

$\boldsymbol{x}$ need not be a Euclidean vector: SSQP also supports states that include manifold-valued
components, such as a floating-base orientation. See
[Manifold-valued states](costs-constraints-dynamics.md#manifold-valued-states) in Costs, Constraints & Dynamics.

```python
model = ssqpy.model.Model(horizon, dt)
```

## Defining a model

A model can be defined in two ways, which can also be mixed:

1. **Custom function objects**: subclass `StageCost`, `StageDynamics` and (optionally) `StageConstraints` and pass
   instances to `model.finalize(...)`, as in the [Quickstart](../getting-started/quickstart.md). This is required
   for systems that are not described by a robot XML file (e.g. simplified/reduced models).
2. **XML description files**: pass `urdf_path` or `mjcf_path` to the constructor to load a robot description via
   [Pinocchio](https://github.com/stack-of-tasks/pinocchio). This gives access to robot algorithms (forward/inverse
   dynamics, kinematics, Jacobians) that the [predefined costs and constraints](costs-constraints-dynamics.md) use
   internally, and lets you use `Model.addContactConstraint` for contact-implicit stages.

When a model is constructed from an XML file, it is not immediately usable for optimization: call `finalize()` to
assign costs/dynamics/constraints (explicitly, or letting the model build its own robot dynamics) before solving.
`model.finalized` reports whether this has happened.

### Built-in presets

A handful of common robots are baked into the ssqp binary as URDF presets, so they can be used without pointing
`urdf_path` at a file on disk. Pass `from_preset` instead:

```python
model = ssqpy.model.Model(horizon, dt, from_preset=ssqpy.model.Preset.Go2)
```

Available presets: `Go2`, `A1`, `Panda`, `G1`, `Talos`, `DoublePendulum` (`ssqpy.model.Preset.NONE` is the default,
meaning no preset). Each preset fixes its own `urdf_path` and `free_flyer`, so `from_preset` is mutually exclusive
with `urdf_path`, `mjcf_path` and `free_flyer`; passing both raises an error.

### Constructor arguments

```python
Model(horizon: int, dt: float, **kwargs)
```

| Keyword argument | Description |
| --- | --- |
| `urdf_path` / `mjcf_path` | Path to a URDF or MJCF file describing the robot |
| `free_flyer` | Whether the model has a floating base (only relevant when loaded from XML) |
| `from_preset` | A [built-in URDF preset](#built-in-presets) to build the model from, instead of `urdf_path` |
| `solver_mode` | The [`SolverMode`](#solver-mode) formulation to optimize in |
| `integration` | The [`Integration`](#integration-scheme) scheme used to propagate state |
| `ignore_innate_constraints` | Whether to ignore joint/velocity/torque limits declared in the XML |
| `actuated_joints` | Restrict actuation to a subset of joint indices |

### `finalize` arguments

`model.finalize(**kwargs)` accepts, among others: `costs` / `stage_cost` / `terminal_cost`, `dynamics`,
`constraints`, `custom_lower_position_bounds` / `custom_upper_position_bounds` / `custom_velocity_bounds` /
`custom_torque_bounds` (override XML-derived innate constraint bounds), and `init_states` / `init_controls` (initial
guesses). See the [API reference](../reference/model.md#ssqpy.model.Model.finalize) for the complete, up-to-date
list.

## Solver mode

`SolverMode` selects the NLP formulation used for the dynamics equality constraint:

* **`Kinematics`**: no system dynamics; controls are velocities and the state is the system configuration.
* **`ForwardDynamics`**: the state is configuration + velocity; controls are torques applied via forward dynamics.
* **`InverseDynamics`**: the state is configuration + velocity; controls are accelerations, with torques (needed to
  produce them) constrained to stay within effort limits. This simplifies the LQR formulation and can be more
  robust than `ForwardDynamics`.
* **`ReducedInverseDynamics`**: like `InverseDynamics`, but an unactuated floating base and/or joint subset is
  enforced implicitly through state propagation rather than via equality constraints.

## Integration scheme

`Integration` selects how velocity and configuration are propagated between stages:

* **`ExplicitEuler`**: $\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{v}_n dt$, $\boldsymbol{v}_{n+1} = \boldsymbol{v}_n + \boldsymbol{a}_n dt$
* **`SemiImplicitEuler`** (default): $\boldsymbol{x}_{n+1} = \boldsymbol{x}_n + \boldsymbol{v}_{n+1} dt$, $\boldsymbol{v}_{n+1} = \boldsymbol{v}_n + \boldsymbol{a}_n dt$

## Contacts and forces

For XML-based models, `Model.addForce(k, frame)` declares an external force at a frame for a stage, and
`Model.addContactConstraint(k, frame, mu=0.6, max_normal_force=inf, eq_tol=0.0)` is a convenience that both adds a
contact constraint (friction cone, non-penetration) *and* the associated force in one call; this is the mechanism
used by legged-robot examples (e.g. Go2, G1) to model foot contacts against the [terrain](terrain.md).

See the [Python API reference](../reference/model.md) for the full `Model` method list.
