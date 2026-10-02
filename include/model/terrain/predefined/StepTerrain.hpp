#ifndef SSQP_MODEL_TERRAIN_PREDEFINED_STEP_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_PREDEFINED_STEP_TERRAIN_HPP

#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
/** Piecewise flat terrain with a flight of steps along the x axis
 * Height is `base_height` for $x < x_{start}$, then rises by `step_height` at
 * `x_start` and every `step_length` after that, for `num_steps` steps in total.
 * The terrain stays at the top step height afterwards. With the default of a
 * single step, `step_length` has no effect
 */
class StepTerrain : public Terrain
{
  public:
    StepTerrain(const double x_start, const double step_height,
                const double step_length = 1.0, const uint num_steps = 1,
                const double base_height = 0.0);

    TerrainPoint at(const double x, const double y) const override;

    nlohmann::ordered_json to_json() const override;

  protected:
    const double _x_start, _step_height, _step_length, _base_height;
    const uint   _num_steps;
};

}
}
}
}

#endif
