#ifndef SSQP_MODEL_TERRAIN_PREDEFINED_FLAT_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_PREDEFINED_FLAT_TERRAIN_HPP

#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
class FlatTerrain : public Terrain
{
  public:
    FlatTerrain(const double base_height = 0.0);

    TerrainPoint at(const double x, const double y) const override;

    nlohmann::ordered_json to_json() const override;

  protected:
    const double _base_height;
};

}
}
}
}

#endif