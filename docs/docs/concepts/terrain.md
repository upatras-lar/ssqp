# Terrain

`Terrain` (`ssqpy.model.terrain`, `ssqp::model::terrain`) describes the ground that a model's contacts act against.
Every `Model` has an associated terrain (`Model.terrain`), defaulting to `FlatTerrain` at $z = 0$.

```python
class Terrain:
    def at(self, x: float, y: float) -> TerrainPoint: ...
    def distance(self, x: float, y: float, z: float) -> float: ...
```

`at(x, y)` returns a `TerrainPoint`: the terrain height, surface normal, two tangent directions, and the surface
slope (`dhdx`, `dhdy`) at that point. `distance(x, y, z)` gives the (default: vertical) distance from a 3D point to
the terrain surface. These are used by contact-related [constraints](costs-constraints-dynamics.md#constraints)
(e.g. `Model.addContactConstraint`) to enforce non-penetration and to build the local contact frame for the
friction cone.

## `FlatTerrain`

The only predefined terrain currently shipped:

```python
terrain = ssqpy.model.terrain.FlatTerrain(base_height=0.0)
```

## Custom terrain

Subclass `Terrain` and implement `at` (required) and, optionally, `distance` (a default implementation is provided)
to model non-flat ground, such as stairs, slopes, or heightmaps, for locomotion tasks.
