# Costs, Constraints & Dynamics

Costs, dynamics and constraints are all *stagewise functions*: they evaluate at a single stage's
$(\boldsymbol{x}, \boldsymbol{u})$ and provide the derivatives SSQP needs to quadratize the cost and linearize the
dynamics/constraints at each iteration. `StageCost`, `StageDynamics` and `StageConstraints` all extend a common
`Function` interface (`evaluate`, `output_size`, `names`), then add their own gradient/Hessian requirements.

You can implement any of these from scratch in Python (as in the [Quickstart](../getting-started/quickstart.md)) or
C++, or use one of the predefined implementations that ship with SSQP, listed below and fully documented in the
[Python API reference](../reference/model.md).

## Costs

`StageCost` ($\ell_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}$, so `output_size()` is always `1`) requires:

* `evaluate(x, u)`: the scalar cost
* `gradient(x, u)`: $\nabla_x \ell_k$ and $\nabla_u \ell_k$
* `hessian_xx(x, u)`, `hessian_xu(x, u)`, `hessian_uu(x, u)`: the cost Hessian blocks

Costs can be named, enabled/disabled at runtime (`set_enabled`/`enabled`, though a stage always keeps at least one
cost, so disabling every cost on a stage has no effect), and multiple costs can be added to the same stage via
`Model.addCost` / `Model.addRunningCost`.

### Predefined costs

`ssqpy.model.costs` (`ssqp::model::stagewise` in C++) ships a library of squared-error costs for common OCP
objectives, all built on Pinocchio kinematics for XML-based models:

`SquaredJointAngleCost`, `SquaredJointVelocityCost`, `SquaredControlCost`, `SquaredIDTorqueCost`,
`SquaredForceCost`, `SquaredConfigurationErrorCost`, `FloatingBaseSquaredTranslationErrorCost`,
`FloatingBaseSquaredRotationErrorCost`, `FloatingBaseSquaredPoseErrorCost`,
`FloatingBaseSquaredVelocityErrorCost`, `FrameSquaredTranslationErrorCost`, `FrameSquaredTerrainHeightErrorCost`,
`FrameSquaredRotationErrorCost`,
`FrameSquaredPoseErrorCost`, `FrameSquaredVelocityErrorCost`, `CenterOfMassTrackingCost`,
`FramesInverseSquaredDistanceCost`.

See [`ssqpy.model.costs`](../reference/model-costs.md) for constructor arguments and details for each.

Predefined costs and constraints refer to the model they are built with rather than copying it, so they always see
its current forces and terrain. In C++, the model must outlive them, and a `Model` cannot be copied. In Python, each
one keeps its model alive.

## Dynamics

`StageDynamics` ($\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}^{n_x}$) requires:

* `state_size()`, `control_size()`: dimensions of $\boldsymbol{x}$ and $\boldsymbol{u}$
* `evaluate(x, u)`: the next state
* `gradient(x, u)`: $\nabla_x \boldsymbol{f}_k$, $\nabla_u \boldsymbol{f}_k$
* `hessprod_xx`, `hessprod_xu`, `hessprod_uu`: vector-Hessian products against the Lagrange multiplier
  $\boldsymbol{\lambda}_k$ (used instead of raw Hessians, since $n_x$ can be large)

Dynamics also define the manifold structure of the state through `state_right_plus`/`state_right_minus` (and
`state_diff_size`, when the tangent space differs in size from the state itself). For a Euclidean state these
default to plain vector addition/subtraction; for states that include e.g. a floating-base orientation (a Lie
group element), they implement the appropriate manifold operators. Predefined, XML-based dynamics already handle
this correctly for the selected [`SolverMode`](model.md#solver-mode) and [`Integration`](model.md#integration-scheme).

### Manifold-valued states

Not every model's state lives in a flat Euclidean space $\mathcal{R}^{n_x}$. A floating-base robot's orientation,
for example, is an element of the Lie group $SO(3)$ (or its full pose, $SE(3)$), not a vector: adding two
orientations, or taking their plain difference, isn't a well-defined operation. Rather than requiring
$\boldsymbol{x} \in \mathcal{R}^{n_x}$, SSQP treats a dynamics function's domain as an arbitrary composite
manifold: a state can combine one or more manifold-valued segments with one or more ordinary Euclidean segments,
and every place SSQP would otherwise use vector addition/subtraction instead goes through two operators:

* $\boldsymbol{x} \oplus \boldsymbol{dx}$ (`state_right_plus`): retracts a tangent-space update $\boldsymbol{dx}$
  onto the manifold at $\boldsymbol{x}$, producing a new state.
* $\boldsymbol{x}_1 \ominus \boldsymbol{x}_2$ (`state_right_minus`): the tangent-space displacement between two
  states, i.e. their local, chart-consistent difference.

For a purely Euclidean state, these reduce to plain vector addition/subtraction, and `state_diff_size()` (the
tangent space dimension $n_{dx}$) defaults to `state_size()` ($n_x$). For a manifold-valued segment, $n_{dx}$ can
differ from $n_x$: a quaternion-parameterized $SO(3)$ orientation, for instance, contributes 4 parameters to
$n_x$ but only 3 degrees of freedom to $n_{dx}$. SSQP linearizes, integrates and reports convergence norms in the
tangent space, never in the raw parameterization, so the gradients/Hessians returned by `gradient`/`hessprod_*`
are always sized against `state_diff_size()` ($n_{dx} \times n_{dx}$ or $n_{dx} \times n_u$), not `state_size()`.

XML-based (Pinocchio-backed) models handle all of this automatically: a `free_flyer=True` model's floating-base
pose is represented and integrated as its native $SE(3)$/quaternion manifold element, and the predefined
costs/constraints that touch floating-base or frame poses (e.g. `FloatingBaseSquaredPoseErrorCost`,
`FrameTranslationBound`) use the corresponding Lie-group log/exp maps internally, consistent with the selected
[`Integration`](model.md#integration-scheme) scheme. Custom dynamics only need to implement
`state_right_plus`/`state_right_minus` (and `state_diff_size`, if it differs from `state_size`) to opt into the
same treatment for their own manifold-valued states.

## Constraints

`StageConstraints` ($\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u}) \geq \boldsymbol{0}$, or bounded via
`lower_bound()`/`upper_bound()`) mirrors `StageDynamics`' derivative interface (`gradient` and
`hessprod_xx`, `hessprod_xu`, `hessprod_uu` against a multiplier $\boldsymbol{\mu}_k$) for an arbitrary output
size, and adds:

* `lower_bound()` / `upper_bound()`: per-element bounds (defaults to $[0, \infty)$, i.e. an inequality
  $\geq 0$; pass equal bounds for an equality constraint)
* `required_forces()` / `force_frame(i)`: for constraints that involve external contact forces (see
  [Contacts and forces](model.md#contacts-and-forces) in Model)

### Predefined constraints

`ssqpy.model.constraints` provides: `JointAngleConstraint`, `JointVelocityConstraint`, `JointControlConstraint`,
`FloatingBaseTranslationBound`, `FloatingBaseRotationBound`, `FloatingBasePoseBound`,
`FloatingBaseLinearVelocityBound`, `FloatingBaseAngularVelocityBound`, `FrameTranslationBound`,
`FrameTerrainHeightBound`, `FrameRotationBound`, `FramePoseBound`, `FrameLinearVelocityBound`, `FrameAngularVelocityBound`,
`CenterOfMassBound`. The C++ library additionally defines `ContactConstraint` directly, which
`Model.addContactConstraint` constructs and wires up for you (friction cone + non-penetration against the model's
[terrain](terrain.md) + no-slip zero velocity, which `no_slip=False` turns off).

See [`ssqpy.model.constraints`](../reference/model-constraints.md) for constructor arguments and details for each.

## Innate (XML-derived) constraints

When a model is built from a URDF/MJCF file, joint position, velocity and torque limits declared in the XML are
added automatically as constraints unless `ignore_innate_constraints=True` is passed to the `Model` constructor.
`finalize`'s `custom_lower_position_bounds`, `custom_upper_position_bounds`, `custom_velocity_bounds` and
`custom_torque_bounds` arguments override the XML-derived values in place.
