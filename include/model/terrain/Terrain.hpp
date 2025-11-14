#ifndef SSQP_MODEL_TERRAIN_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_TERRAIN_HPP

#include "types.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
class Terrain
{
  public:
    struct TerrainPoint
    {
        double height;
        Vec3   normal;
        Vec3   t1, t2;
        double dhdx, dhdy;
    };

    virtual TerrainPoint at(const double x, const double y) const = 0;
    TerrainPoint         at(const Vec2 &p) const;

    virtual double distance(const double x, const double y,
                            const double z) const;
    double         distance(const Vec3 &p) const;
};

template<typename T, typename... Args>
std::shared_ptr<Terrain> make_terrain(Args... args);
}
}
}

#include "src/model/terrain/Terrain.tpp"

#endif