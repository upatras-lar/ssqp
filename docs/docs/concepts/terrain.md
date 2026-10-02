# Terrain

`Terrain` (`ssqpy.model.terrain`, `ssqp::model::terrain`) describes the ground that a model's contacts act against.
Every `Model` has an associated terrain (`Model.terrain`), defaulting to `FlatTerrain` at $z = 0$.

```python
class Terrain:
    def at(self, x: float, y: float) -> TerrainPoint: ...
    def distance(self, x: float, y: float, z: float) -> float: ...
```

`at(x, y)` returns a `TerrainPoint`: the terrain height, surface normal, two tangent directions, the slope
(`dhdx`, `dhdy`) and the height second derivatives (`d2hdx2`, `d2hdxdy`, `d2hdy2`) at that point. `distance(x, y, z)`
gives the (default: vertical) distance from a 3D point to the terrain surface. These are used by contact-related
[constraints](costs-constraints-dynamics.md#constraints) (e.g. `Model.addContactConstraint`) to enforce
non-penetration and to build the local contact frame for the friction cone.

The contact constraint differentiates the contact frame with respect to position using the height derivatives, so
the frame must follow the convention of `TerrainPoint.from_height(height, dhdx, dhdy, d2hdx2, d2hdxdy, d2hdy2)`:
`normal = (-dhdx, -dhdy, 1) / norm`, `t1 = (1, 0, dhdx) / norm` and `t2 = normal x t1`. Always build points with it.

## Predefined terrains

```python
terrain = ssqpy.model.terrain.FlatTerrain(base_height=0.0)
terrain = ssqpy.model.terrain.SlopeTerrain(slope_x, slope_y=0.0, base_height=0.0)
terrain = ssqpy.model.terrain.StepTerrain(
    x_start, step_height, step_length=1.0, num_steps=1, base_height=0.0
)
terrain = ssqpy.model.terrain.HeightmapTerrain(heights, resolution, x0=0.0, y0=0.0)
```

`StepTerrain` is piecewise flat. The height rises by `step_height` at `x_start` and again every `step_length` along
x, for `num_steps` steps, and stays at the top afterwards. With the default single step, `step_length` has no effect.

`HeightmapTerrain` interpolates a uniform grid, where `heights[i, j]` is the height at
`(x0 + i * resolution, y0 + j * resolution)`, with a bicubic Catmull-Rom spline. The surface passes through every grid
height and has continuous slope. Outside the grid it extends flat from the closest edge value.

Pass a terrain to a model with `Model(..., terrain=terrain)`.

`solver.write(...)` stores a description of the terrain under `properties.terrain` in the output file, with a `type`
(`flat`, `slope`, `step` or `heightmap`) and its parameters. `visualizer/visualize.py` reads it and draws the terrain
around the trajectory. Custom terrains are stored as `{"type": "custom"}` and are not drawn. Files without a terrain
are drawn as flat ground at $z = 0$.

## Custom terrain

Subclass `Terrain` and implement `at` (required) and, optionally, `distance` (a default implementation is provided).
Python subclasses are called from the solver loop for every contact at every stage, so heavy terrains are better
implemented in C++.
