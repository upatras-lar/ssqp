#ifndef SSQP_MODEL_TERRAIN_PREDEFINED_HEIGHTMAP_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_PREDEFINED_HEIGHTMAP_TERRAIN_HPP

#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
/** Smooth terrain interpolated from a uniform grid of heights
 * `heights(i, j)` is the height at $(x_0 + i r, y_0 + j r)$, where $r$ is the
 * grid resolution. Heights are interpolated with a bicubic Catmull-Rom spline,
 * which passes through every grid height and has continuous height and slope.
 * The height second derivatives are piecewise continuous.
 * Outside the grid, the terrain extends flat from its closest edge value
 */
class HeightmapTerrain : public Terrain
{
  public:
    HeightmapTerrain(const MatX &heights, const double resolution,
                     const double x0 = 0.0, const double y0 = 0.0);

    TerrainPoint at(const double x, const double y) const override;

    nlohmann::ordered_json to_json() const override;

  protected:
    const MatX   _heights;
    const double _res, _x0, _y0;
};

}
}
}
}

#endif
