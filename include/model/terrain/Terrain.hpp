#ifndef SSQP_MODEL_TERRAIN_TERRAIN_HPP
#define SSQP_MODEL_TERRAIN_TERRAIN_HPP

#include "types.hpp"
#include "nlohmann/json.hpp"

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
        double d2hdx2 = 0, d2hdxdy = 0, d2hdy2 = 0;

        // Builds a point from the height and its derivatives, using the frame
        // convention required by contact constraints
        //  - normal = (-dhdx, -dhdy, 1) / norm
        //  - t1 = (1, 0, dhdx) / norm
        //  - t2 = normal x t1
        static TerrainPoint from_height(double height, double dhdx, double dhdy,
                                        double d2hdx2 = 0, double d2hdxdy = 0,
                                        double d2hdy2 = 0);
    };

    virtual TerrainPoint at(const double x, const double y) const = 0;
    TerrainPoint         at(const Vec2 &p) const;

    virtual double distance(const double x, const double y,
                            const double z) const;
    double         distance(const Vec3 &p) const;

    // Describes the terrain for trajectory output files
    // Terrains that do not override this are described as `custom`
    virtual nlohmann::ordered_json to_json() const;
};

template<typename T, typename... Args>
std::shared_ptr<Terrain> make_terrain(Args... args);
}
}
}

#include "src/model/terrain/Terrain.tpp"

#endif