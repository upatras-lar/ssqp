#include "model/presets/Presets.hpp"

#include <stdexcept>

#include "model/presets/go2_urdf.hpp"
#include "model/presets/a1_urdf.hpp"
#include "model/presets/panda_urdf.hpp"
#include "model/presets/g1_urdf.hpp"
#include "model/presets/talos_urdf.hpp"
#include "model/presets/double_pendulum_urdf.hpp"

namespace ssqp
{
namespace model
{
namespace presets
{
std::string_view urdf(Preset preset)
{
    switch (preset)
    {
    case Preset::Go2:
        return go2_urdf;
    case Preset::A1:
        return a1_urdf;
    case Preset::Panda:
        return panda_urdf;
    case Preset::G1:
        return g1_urdf;
    case Preset::Talos:
        return talos_urdf;
    case Preset::DoublePendulum:
        return double_pendulum_urdf;
    default:
        throw std::invalid_argument("No URDF available for the given preset");
    }
}

bool freeFlyer(Preset preset)
{
    switch (preset)
    {
    case Preset::Go2:
    case Preset::A1:
    case Preset::G1:
    case Preset::Talos:
        return true;
    case Preset::Panda:
    case Preset::DoublePendulum:
        return false;
    default:
        throw std::invalid_argument(
          "No free_flyer setting available for the given preset");
    }
}
}
}
}
