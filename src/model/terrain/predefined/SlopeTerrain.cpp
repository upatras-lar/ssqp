#include "model/terrain/predefined/SlopeTerrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
SlopeTerrain::SlopeTerrain(const double slope_x, const double slope_y,
                           const double base_height)
  : _slope_x(slope_x), _slope_y(slope_y), _base_height(base_height)
{
}

Terrain::TerrainPoint SlopeTerrain::at(const double x, const double y) const
{
    return TerrainPoint::from_height(_base_height + _slope_x * x +
                                       _slope_y * y,
                                     _slope_x,
                                     _slope_y);
}

nlohmann::ordered_json SlopeTerrain::to_json() const
{
    return { { "type", "slope" },
             { "slope_x", _slope_x },
             { "slope_y", _slope_y },
             { "base_height", _base_height } };
}

}
}
}
}
