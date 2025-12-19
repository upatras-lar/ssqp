#include <nanobind/nanobind.h>

namespace nb = nanobind;

void bind_Model(nb::module_ &);
void bind_Terrain(nb::module_ &);
void bind_costs(nb::module_ &);
void bind_constraints(nb::module_ &);

void bind_model(nb::module_ &m)
{
    auto costs       = m.def_submodule("costs");
    auto constraints = m.def_submodule("constraints");

    bind_Model(m);
    bind_costs(costs);
    bind_constraints(constraints);
}