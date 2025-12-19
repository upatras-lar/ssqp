#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model/Model.hpp"
#include "model/terrain/Terrain.hpp"

namespace nb = nanobind;
using namespace ssqp::model;
using namespace ssqp::model::terrain;

void bind_SystemModel(nb::module_ &m)
{
    nb::class_<Model>(m, "SystemModel")
      .def(
        "__init__",
        [](Model             *self,
           uint               horizon,
           double             dt,
           const std::string &urdf_path,
           bool               free_flyer,
           std::shared_ptr<Terrain>
                                    terrain,
           DynamicsMode             dynamics_mode,
           IntegrationScheme        integration,
           bool                     ignore_innate_constraints,
           const std::vector<uint> &actuated_joints)
        {
            Model::Structure structure {
                urdf_path,      free_flyer,  terrain,
                dynamics_mode,  integration, ignore_innate_constraints,
                actuated_joints
            };

            new (self) Model(horizon, dt, structure);
        },
        nb::arg("horizon"),
        nb::arg("dt"),
        nb::kw_only(),
        nb::arg("urdf_path"),
        nb::arg("free_flyer"),
        nb::arg("terrain"),
        nb::arg("dynamics_mode"),
        nb::arg("integration"),
        nb::arg("ignore_innate_constraints"),
        nb::arg("actuated_joints"));
}