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
    auto costs       = m.def_submodule("costs", R"doc(
        Submodule containing classes for assigning costs to optimization stages.
        This includes the general `StageCost` class, as well as predefined costs
        for ease of use.
        )doc");
    auto constraints = m.def_submodule("constraints", R"doc(
        Submodule containing classes for assigning constraints to optimization
        stages. This includes the general `StageConstraints` class, as well as
        predefined constraints for ease of use.
        )doc");
    auto dynamics    = m.def_submodule("dynamics", R"doc(
        Submodule containing classes for assigning custom dynamics to 
        optimization stages. This includes the general `StageDynamics` class, 
        which must be subclassed and have the relevant methods overridden to 
        define custom dynamics.
        )doc");
    auto terrain     = m.def_submodule("terrain", R"doc(
        Submodule containing the `Terrain` class, used to define the 3D terrain
        that the system model acts in. By default, this is the predefined
        `FlatTerrain` class.
        )doc");

    bind_Model(m);
    bind_terrain(terrain);
    bind_costs(costs);
    bind_constraints(constraints);
    bind_dynamics(dynamics);
}
}
}