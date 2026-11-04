# Predefined Costs & Constraints

[`predefined/Costs.hpp`](https://github.com/upatras-lar/ssqp/blob/main/include/model/stagewise/predefined/Costs.hpp)
and
[`predefined/Constraints.hpp`](https://github.com/upatras-lar/ssqp/blob/main/include/model/stagewise/predefined/Constraints.hpp)
provide ready-made subclasses of [`StageCost` and `StageConstraints`](stagewise.md), built on Pinocchio kinematics
for XML-based models. They match the classes exposed in [`ssqpy.model.costs`](../reference/model-costs.md) and
[`ssqpy.model.constraints`](../reference/model-constraints.md), except for `ContactConstraint`, which is C++-only
since `Model.addContactConstraint` constructs one for you. See
[Costs, Constraints & Dynamics](../concepts/costs-constraints-dynamics.md) in Concepts for how they fit into a model,
and [Reference frames](../concepts/costs-constraints-dynamics.md#reference-frames) for the axes their errors and
bounds are expressed along. The reference below is generated directly from the headers' own doc comments.

## Costs

Costs live in `ssqp::model::stagewise::predefined::costs`.

### `SquaredJointAngleCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredJointAngleCost

<!-- doxy:end -->

### `SquaredJointVelocityCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredJointVelocityCost

<!-- doxy:end -->

### `SquaredControlCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredControlCost

<!-- doxy:end -->

### `SquaredIDTorqueCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredIDTorqueCost

<!-- doxy:end -->

### `SquaredForceCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredForceCost

<!-- doxy:end -->

### `SquaredConfigurationErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::SquaredConfigurationErrorCost

<!-- doxy:end -->

### `FloatingBaseSquaredTranslationErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FloatingBaseSquaredTranslationErrorCost

<!-- doxy:end -->

### `FloatingBaseSquaredRotationErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FloatingBaseSquaredRotationErrorCost

<!-- doxy:end -->

### `FloatingBaseSquaredPoseErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FloatingBaseSquaredPoseErrorCost

<!-- doxy:end -->

### `FloatingBaseSquaredVelocityErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FloatingBaseSquaredVelocityErrorCost

<!-- doxy:end -->

### `FrameSquaredTranslationErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FrameSquaredTranslationErrorCost

<!-- doxy:end -->

### `FrameSquaredTerrainHeightErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FrameSquaredTerrainHeightErrorCost

<!-- doxy:end -->

### `FrameSquaredRotationErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FrameSquaredRotationErrorCost

<!-- doxy:end -->

### `FrameSquaredPoseErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FrameSquaredPoseErrorCost

<!-- doxy:end -->

### `FrameSquaredVelocityErrorCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FrameSquaredVelocityErrorCost

<!-- doxy:end -->

### `CenterOfMassTrackingCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::CenterOfMassTrackingCost

<!-- doxy:end -->

### `FramesInverseSquaredDistanceCost`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::costs::FramesInverseSquaredDistanceCost

<!-- doxy:end -->

## Constraints

Constraints live in `ssqp::model::stagewise::predefined::constraints`.

### `JointAngleConstraint`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::JointAngleConstraint

<!-- doxy:end -->

### `JointVelocityConstraint`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::JointVelocityConstraint

<!-- doxy:end -->

### `JointControlConstraint`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::JointControlConstraint

<!-- doxy:end -->

### `FloatingBaseTranslationBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FloatingBaseTranslationBound

<!-- doxy:end -->

### `FloatingBaseRotationBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FloatingBaseRotationBound

<!-- doxy:end -->

### `FloatingBasePoseBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FloatingBasePoseBound

<!-- doxy:end -->

### `FloatingBaseLinearVelocityBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FloatingBaseLinearVelocityBound

<!-- doxy:end -->

### `FloatingBaseAngularVelocityBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FloatingBaseAngularVelocityBound

<!-- doxy:end -->

### `FrameTranslationBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FrameTranslationBound

<!-- doxy:end -->

### `FrameTerrainHeightBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FrameTerrainHeightBound

<!-- doxy:end -->

### `FrameRotationBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FrameRotationBound

<!-- doxy:end -->

### `FramePoseBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FramePoseBound

<!-- doxy:end -->

### `FrameLinearVelocityBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FrameLinearVelocityBound

<!-- doxy:end -->

### `FrameAngularVelocityBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::FrameAngularVelocityBound

<!-- doxy:end -->

### `CenterOfMassBound`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::CenterOfMassBound

<!-- doxy:end -->

### `ContactConstraint`

<!-- doxy:start level=3 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::constraints::ContactConstraint

<!-- doxy:end -->

## `ReferenceFrame`

Selects the axes that a predefined cost's error or a constraint's bounds are expressed along, passed as the
`reference_frame` argument of the position and velocity terms or set with their `set_reference_frame`. See
[Reference frames](../concepts/costs-constraints-dynamics.md#reference-frames) in Concepts for the options each term
supports and the position bounds' `center`.

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::stagewise::predefined::ReferenceFrame

<!-- doxy:end -->
