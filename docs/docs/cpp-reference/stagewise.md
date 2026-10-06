# Stagewise Functions

`ssqp::model::stagewise` defines the per-stage function interface shared by costs, dynamics and constraints. See
[Costs, Constraints & Dynamics](../concepts/costs-constraints-dynamics.md) in Concepts for the conceptual
explanation. The reference below is generated directly from the headers' own doc comments.

## `Function`

The common base for `StageCost`, `StageDynamics` and `StageConstraints`:

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::Function

<!-- doxy:end -->

## `StageCost`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::StageCost

<!-- doxy:end -->

## `StageDynamics`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::StageDynamics

<!-- doxy:end -->

## `StageConstraints`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::StageConstraints

<!-- doxy:end -->

## Predefined implementations

[`predefined/Costs.hpp`](https://github.com/upatras-lar/ssqp/blob/main/include/model/stagewise/predefined/Costs.hpp)
and
[`predefined/Constraints.hpp`](https://github.com/upatras-lar/ssqp/blob/main/include/model/stagewise/predefined/Constraints.hpp)
provide ready-made subclasses built on Pinocchio kinematics, matching the classes exposed in
[`ssqpy.model.costs`](../reference/model-costs.md) and [`ssqpy.model.constraints`](../reference/model-constraints.md):

**Costs**: `SquaredJointAngleCost`, `SquaredJointVelocityCost`, `SquaredControlCost`, `SquaredIDTorqueCost`,
`SquaredForceCost`, `SquaredConfigurationErrorCost`, `FloatingBaseSquaredTranslationErrorCost`,
`FloatingBaseSquaredRotationErrorCost`, `FloatingBaseSquaredPoseErrorCost`,
`FloatingBaseSquaredVelocityErrorCost`, `FrameSquaredTranslationErrorCost`, `FrameSquaredTerrainHeightErrorCost`,
`FrameSquaredRotationErrorCost`,
`FrameSquaredPoseErrorCost`, `FrameSquaredVelocityErrorCost`, `CenterOfMassTrackingCost`,
`FramesInverseSquaredDistanceCost`.

**Constraints**: `JointAngleConstraint`, `JointVelocityConstraint`, `JointControlConstraint`,
`FloatingBaseTranslationBound`, `FloatingBaseRotationBound`, `FloatingBasePoseBound`,
`FloatingBaseLinearVelocityBound`, `FloatingBaseAngularVelocityBound`, `FrameTranslationBound`,
`FrameTerrainHeightBound`, `FrameRotationBound`, `FramePoseBound`, `FrameLinearVelocityBound`, `FrameAngularVelocityBound`,
`CenterOfMassBound`, and `ContactConstraint` (friction cone + non-penetration + optional no-slip; C++-only, not
directly exposed to Python, since `Model.addContactConstraint` constructs one for you).

Their constructor arguments mirror the Python bindings; see the corresponding entry in the
[Python API reference](../reference/model-costs.md) for the parameter list of each.
