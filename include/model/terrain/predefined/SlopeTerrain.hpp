#ifndef SSQP_MODEL_TERRAIN_PREDEFINED_SLOPE_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_PREDEFINED_SLOPE_TERRAIN_HPP

#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
/** Slope terrain
 * Height is $h = h_0 + s_x x + s_y y$, where $s_x$ and $s_y$ are the slopes
 * along the x and y axes (rise over run)
 */
class SlopeTerrain : public Terrain
{
  public:
    SlopeTerrain(const double slope_x, const double slope_y = 0.0,
                 const double base_height = 0.0);

    TerrainPoint at(const double x, const double y) const override;

    nlohmann::ordered_json to_json() const override;

  protected:
    const double _slope_x, _slope_y, _base_height;
};

}
}
}
}

#endif
