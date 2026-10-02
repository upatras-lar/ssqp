#include "model/terrain/predefined/FlatTerrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
FlatTerrain::FlatTerrain(const double base_height) : _base_height(base_height)
{
}

Terrain::TerrainPoint FlatTerrain::at(const double x, const double y) const
{
    return { _base_height, Vec3(0, 0, 1), Vec3(1, 0, 0), Vec3(0, 1, 0), 0, 0 };
}

nlohmann::ordered_json FlatTerrain::to_json() const
{
    return { { "type", "flat" },
             { "base_height", _base_height } };
}

}
}
}
}