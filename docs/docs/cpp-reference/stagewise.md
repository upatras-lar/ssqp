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
provide ready-made subclasses of `StageCost` and `StageConstraints` built on Pinocchio kinematics. See
[Predefined Costs & Constraints](predefined.md) for the reference of each, and of the `ReferenceFrame` they take.
