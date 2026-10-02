#ifndef SSQP_MODEL_TERRAIN_PREDEFINED_STAIRS_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_PREDEFINED_STAIRS_TERRAIN_HPP

#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
/** Piecewise flat terrain with a flight of stairs along the x axis
 * Height is `base_height` for $x < x_{start}$, then rises by `step_height` at
 * every `step_length` along x, for `num_steps` steps in total. The terrain
 * stays at the top step height afterwards
 */
class StairsTerrain : public Terrain
{
  public:
    StairsTerrain(const double x_start, const double step_length,
                  const double step_height, const uint num_steps,
                  const double base_height = 0.0);

    TerrainPoint at(const double x, const double y) const override;

    nlohmann::ordered_json to_json() const override;

  protected:
    const double _x_start, _step_length, _step_height, _base_height;
    const uint   _num_steps;
};

}
}
}
}

#endif
