#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
Terrain::TerrainPoint Terrain::TerrainPoint::from_height(
  double height, double dhdx, double dhdy, double d2hdx2, double d2hdxdy,
  double d2hdy2)
{
    Vec3 normal = Vec3(-dhdx, -dhdy, 1).normalized();
    Vec3 t1     = Vec3(1, 0, dhdx).normalized();
    return { height,  normal,  t1,      normal.cross(t1), dhdx,
             dhdy,    d2hdx2,  d2hdxdy, d2hdy2 };
}

Terrain::TerrainPoint Terrain::at(const Vec2 &p) const
{
    return at(p[0], p[1]);
}

double Terrain::distance(const double x, const double y, const double z) const
{
    TerrainPoint tp = at(x, y);
    return z - tp.height;
}

nlohmann::ordered_json Terrain::to_json() const
{
    return { { "type", "custom" } };
}

double Terrain::distance(const Vec3 &p) const
{
    return distance(p[0], p[1], p[2]);
}

}
}
}