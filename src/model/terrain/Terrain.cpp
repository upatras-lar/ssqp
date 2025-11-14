#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
Terrain::TerrainPoint Terrain::at(const Vec2 &p) const
{
    return at(p[0], p[1]);
}

double Terrain::distance(const double x, const double y, const double z) const
{
    TerrainPoint tp = at(x, y);
    return z - tp.height;
}

double Terrain::distance(const Vec3 &p) const
{
    return distance(p[0], p[1], p[2]);
}

}
}
}