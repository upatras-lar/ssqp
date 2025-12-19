#include <nanobind/nanobind.h>

#include "model/terrain/Terrain.hpp"

namespace nb = nanobind;
using namespace ssqp::model::terrain;

void bind_Terrain(nb::module_ &m)
{
    nb::class_<Terrain>(m, "Terrain");
}