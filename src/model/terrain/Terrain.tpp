#include "model/terrain/Terrain.hpp"

namespace ssqp
{
namespace model
{
namespace terrain
{
template<typename T, typename... Args>
std::shared_ptr<Terrain> make_terrain(Args... args)
{
    static_assert(std::is_base_of_v<Terrain, T>, "Invalid terrain type");
    return std::make_shared<T>(args...);
}
}
}
}