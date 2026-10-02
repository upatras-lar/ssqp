# Terrain

`ssqp::model::terrain::Terrain` describes the ground a model's contacts act against. See
[Terrain](../concepts/terrain.md) in Concepts for how it's used. The reference below is generated directly from the
headers' own doc comments.

<!-- doxy:start level=1 -->
::: doxy.ssqp.class
    name: ssqp::model::terrain::Terrain

<!-- doxy:end -->

## `FlatTerrain`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::terrain::predefined::FlatTerrain

<!-- doxy:end -->

## `SlopeTerrain`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::terrain::predefined::SlopeTerrain

<!-- doxy:end -->

## `StepTerrain`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::terrain::predefined::StepTerrain

<!-- doxy:end -->

## `HeightmapTerrain`

<!-- doxy:start level=2 -->
::: doxy.ssqp.class
    name: ssqp::model::terrain::predefined::HeightmapTerrain

<!-- doxy:end -->

## Custom terrain

Subclass `Terrain` and implement `at` (required), optionally overriding `distance`, to model non-flat ground
(stairs, slopes, heightmaps) for locomotion tasks. Pass an instance via `Model::Structure::terrain`.
