#include <nanobind/nanobind.h>

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

void bind_Model(nb::module_ &);
void bind_terrain(nb::module_ &);
void bind_costs(nb::module_ &);
void bind_constraints(nb::module_ &);
void bind_dynamics(nb::module_ &);

void bind_model(nb::module_ &m)
{
    auto costs       = m.def_submodule("costs");
    auto constraints = m.def_submodule("constraints");
    auto dynamics    = m.def_submodule("dynamics");
    auto terrain     = m.def_submodule("terrain");

    bind_Model(m);
    bind_terrain(terrain);
    bind_costs(costs);
    bind_constraints(constraints);
    bind_dynamics(dynamics);
}
}
}