#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_Terrain(nb::module_ &);

void bind_terrain(nb::module_ &m)
{
    bind_Terrain(m);
}
}
}
