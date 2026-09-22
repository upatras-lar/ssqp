#ifndef SSQP_MODEL_PRESETS_PRESETS_HPP
#define SSQP_MODEL_PRESETS_PRESETS_HPP

#include <string_view>

#include "types.hpp"

namespace ssqp
{
namespace model
{
namespace presets
{
// Returns the embedded URDF XML content for the given preset
std::string_view urdf(Preset preset);

// Returns whether the given preset has a floating base
bool freeFlyer(Preset preset);
}
}
}

#endif
