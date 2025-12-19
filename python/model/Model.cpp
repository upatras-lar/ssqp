#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "model/Model.hpp"
#include "model/terrain/Terrain.hpp"

namespace nb = nanobind;
using namespace ssqp::model;
using namespace ssqp::model::terrain;

void bind_Model(nb::module_ &m)
{
    nb::class_<Model>(m, "Model")
      .def("__init__",
           [](Model *self, uint horizon, double dt, nb::kwargs kwargs)
           {
               Model::Structure structure;

               if (kwargs.contains("urdf_path"))
               {
                   structure.urdf_path =
                     nb::cast<std::string>(kwargs["urdf_path"]);
               }
               if (kwargs.contains("free_flyer"))
               {
                   structure.free_flyer = nb::cast<bool>(kwargs["free_flyer"]);
               }
               if (kwargs.contains("dynamics_mode"))
               {
                   structure.dynamics_mode =
                     nb::cast<DynamicsMode>(kwargs["dynamics_mode"]);
               }
               if (kwargs.contains("integration"))
               {
                   structure.integration =
                     nb::cast<IntegrationScheme>(kwargs["integration"]);
               }
               if (kwargs.contains("ignore_innate_constraints"))
               {
                   structure.ignore_innate_constraints =
                     nb::cast<bool>(kwargs["ignore_innate_constraints"]);
               }
               if (kwargs.contains("actuated_joints"))
               {
                   structure.actuated_joints =
                     nb::cast<std::vector<uint>>(kwargs["actuated_joints"]);
               }

               new (self) Model(horizon, dt, structure);
           })
      .def(
        "finalize",
        [](Model *self, nb::kwargs kwargs)
        {
            Model::Config config;

            if (kwargs.contains("costs"))
            {
                config.costs =
                  nb::cast<std::vector<std::shared_ptr<stagewise::StageCost>>>(
                    kwargs["costs"]);
            }
            if (kwargs.contains("stage_cost"))
            {
                config.stage_cost =
                  nb::cast<std::shared_ptr<stagewise::StageCost>>(
                    kwargs["stage_cost"]);
            }
            if (kwargs.contains("terminal_cost"))
            {
                config.stage_cost =
                  nb::cast<std::shared_ptr<stagewise::StageCost>>(
                    kwargs["terminal_cost"]);
            }
            if (kwargs.contains("constraints"))
            {
                config.constraints = nb::cast<
                  std::vector<std::shared_ptr<stagewise::StageConstraints>>>(
                  kwargs["constraints"]);
            }
            if (kwargs.contains("dynamics"))
            {
                config.dynamics = nb::cast<
                  std::vector<std::shared_ptr<stagewise::StageDynamics>>>(
                  kwargs["dynamics"]);
            }
            if (kwargs.contains("init_states"))
            {
                config.init_states =
                  nb::cast<std::vector<Eigen::VectorXd>>(kwargs["init_states"]);
            }
            if (kwargs.contains("init_controls"))
            {
                config.init_controls = nb::cast<std::vector<Eigen::VectorXd>>(
                  kwargs["init_controls"]);
            }

            self->finalize(config);
        })
      .def(
        "addCost",
        static_cast<void (
          Model::*)(const uint, const std::shared_ptr<stagewise::StageCost> &)>(
          &Model::addCost),
        nb::arg("k"),
        nb::arg("cost"));
}