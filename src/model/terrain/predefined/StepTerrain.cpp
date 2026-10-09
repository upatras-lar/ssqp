#include "model/terrain/predefined/StepTerrain.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace ssqp
{
namespace model
{
namespace terrain
{
namespace predefined
{
StepTerrain::StepTerrain(const double x_start, const double step_height,
                         const double step_length, const uint num_steps,
                         const double base_height)
  : _x_start(x_start)
  , _step_height(step_height)
  , _step_length(step_length)
  , _base_height(base_height)
  , _num_steps(num_steps)
{
    assert(step_length > 0.0 && "Step length must be positive");
}

Terrain::TerrainPoint StepTerrain::at(const double x, const double y) const
{
    const double steps_climbed =
      std::clamp(std::floor((x - _x_start) / _step_length) + 1.0,
                 0.0,
                 double(_num_steps));

    return TerrainPoint::from_height(_base_height +
                                       steps_climbed * _step_height,
                                     0,
                                     0);
}

nlohmann::ordered_json StepTerrain::to_json() const
{
    return {
        {        "type",       "step" },
        {     "x_start",     _x_start },
        { "step_height", _step_height },
        { "step_length", _step_length },
        {   "num_steps",   _num_steps },
        { "base_height", _base_height }
    };
}

}
}
}
}
