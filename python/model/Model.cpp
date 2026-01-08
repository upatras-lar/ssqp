#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/terrain/Terrain.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;
using namespace model::stagewise::predefined::constraints;
using namespace model::terrain;

/*
int model_tp_traverse(PyObject *self, visitproc visit, void *arg)
{
    Py_VISIT(Py_TYPE(self));

    if (!nb::inst_ready(self))
    {
        return 0;
    }

    Model *model = nb::inst_ptr<Model>(self);

    CostsContainer       costs       = model->costs();
    ConstraintsContainer constraints = model->constraints();
    DynamicsContainer    dynamics    = model->dynamics();

    for (uint k = 0; k < model->horizon(); ++k)
    {
        nb::handle value = nb::find(costs[k]);
        Py_VISIT(value.ptr());
        value = nb::find(constraints[k]);
        Py_VISIT(value.ptr());
        value = nb::find(dynamics[k]);
        Py_VISIT(value.ptr());
    }

    return 0;
}

// TODO: merged still leak
int model_tp_clear(PyObject *self)
{
    std::cout << "Running clear" << std::endl;
    Model *model = nb::inst_ptr<Model>(self);

    CostsContainer       costs       = model->costs();
    ConstraintsContainer constraints = model->constraints();
    DynamicsContainer    dynamics    = model->dynamics();

    for (uint k = 0; k < model->horizon(); ++k)
    {
        costs[k]       = nullptr;
        constraints[k] = nullptr;
        dynamics[k]    = nullptr;
    }

    return 0;
}

PyType_Slot model_slots[] = {
    { Py_tp_traverse, (void *)model_tp_traverse },
    {    Py_tp_clear,    (void *)model_tp_clear },
    {              0,                         0 }
};
*/

void bind_Model(nb::module_ &m)
{
    nb::enum_<DynamicsMode>(m, "DynamicsMode")
      .value("Forward", DynamicsMode::Forward)
      .value("Inverse", DynamicsMode::Inverse)
      .export_values();

    nb::enum_<IntegrationScheme>(m, "Integration")
      .value("ExplicitEuler", IntegrationScheme::ExplicitEuler)
      .value("SemiImplicitEuler", IntegrationScheme::SemiImplicitEuler)
      .export_values();

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
      .def("finalize",
           [](Model *self, nb::kwargs kwargs)
           {
               Model::Config config;

               if (kwargs.contains("costs"))
               {
                   config.costs =
                     nb::cast<std::vector<std::shared_ptr<StageCost>>>(
                       kwargs["costs"]);
               }
               if (kwargs.contains("stage_cost"))
               {
                   config.stage_cost =
                     nb::cast<std::shared_ptr<StageCost>>(kwargs["stage_cost"]);
               }
               if (kwargs.contains("terminal_cost"))
               {
                   config.stage_cost = nb::cast<std::shared_ptr<StageCost>>(
                     kwargs["terminal_cost"]);
               }
               if (kwargs.contains("constraints"))
               {
                   config.constraints =
                     nb::cast<std::vector<std::shared_ptr<StageConstraints>>>(
                       kwargs["constraints"]);
               }
               if (kwargs.contains("dynamics"))
               {
                   config.dynamics =
                     nb::cast<std::vector<std::shared_ptr<StageDynamics>>>(
                       kwargs["dynamics"]);
               }
               if (kwargs.contains("init_states"))
               {
                   config.init_states =
                     nb::cast<std::vector<VecX>>(kwargs["init_states"]);
               }
               if (kwargs.contains("init_controls"))
               {
                   config.init_controls =
                     nb::cast<std::vector<VecX>>(kwargs["init_controls"]);
               }

               self->finalize(config);
           })
      .def("addCost",
           static_cast<void (Model::*)(const uint,
                                       const std::shared_ptr<StageCost> &)>(
             &Model::addCost),
           nb::arg("k"),
           nb::arg("cost"))
      .def("addConstraints",
           static_cast<void (
             Model::*)(const uint, const std::shared_ptr<StageConstraints> &)>(
             &Model::addConstraints),
           nb::arg("k"),
           nb::arg("constraints"))
      .def(
        "addContactConstraint",
        [](Model &self, const uint k, const std::string &frame, const double mu)
        {
            self.addConstraints<ContactConstraint>(k,
                                                   self,
                                                   frame,
                                                   self.nq(k),
                                                   self.nv(k),
                                                   mu);
        },
        nb::arg("k"),
        nb::arg("frame"),
        nb::arg("mu") = 0.6)
      .def_prop_ro("finalized", [](Model &self) { return self.finalized(); })
      .def_prop_ro("horizon", [](Model &self) { return self.horizon(); })
      .def_prop_ro("dt", [](Model &self) { return self.dt(); })
      .def(
        "getStateSize",
        [](Model &self, const uint k)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }

            return self.dims().stage_state[k];
        },
        nb::arg("k"))
      .def("getLinearStateSize",
           [](Model &self, const uint k)
           {
               if (!self.finalized())
               {
                   throw std::runtime_error("Model is not finalized");
               }

               return self.linear_dims().stage_state[k];
           })
      .def(
        "getControlSize",
        [](Model &self, const uint k)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }

            return self.dims().stage_control[k];
        },
        nb::arg("k"))
      .def(
        "getConstraintSize",
        [](Model &self, const uint k)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }

            return self.dims().stage_constraint[k];
        },
        nb::arg("k"))
      .def("getLinearConstraintSize",
           [](Model &self, const uint k)
           {
               if (!self.finalized())
               {
                   throw std::runtime_error("Model is not finalized");
               }

               return self.linear_dims().stage_constraint[k];
           })
      .def(
        "getnq",
        [](Model &self, const uint k)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }

            return self.nq(k);
        },
        nb::arg("k"))
      .def(
        "getnv",
        [](Model &self, const uint k)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }

            return self.nv(k);
        },
        nb::arg("k"))
      .def(
        "getnjoints",
        [](Model &self, const uint k)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }

            return self.njoints(k);
        },
        nb::arg("k"))
      .def("hasFloatingBase",
           [](Model &self, const uint k)
           {
               if (!self.urdf())
               {
                   throw std::runtime_error("Model was not loaded from URDF");
               }

               return self.floatingBase();
           })
      .def(
        "getCost",
        [](Model &self, const uint k) { return self.costs()[k]; },
        nb::arg("k"))
      .def(
        "getDynamics",
        [](Model &self, const uint k) { return self.dynamics()[k]; },
        nb::arg("k"))
      .def(
        "getConstraints",
        [](Model &self, const uint k) { return self.constraints()[k]; },
        nb::arg("k"))
      .def_prop_ro("terrain", [](Model &self) { return self.terrain(); })
      .def(
        "addForce",
        [](Model &self, const uint k, const std::string &frame)
        {
            if (self.finalized())
            {
                throw std::runtime_error("Model is finalized");
            }
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }

            Force f = self.addForce(k, frame);
            return f.force_index;
        },
        nb::arg("k"),
        nb::arg("frame"))
      .def("getTotalForces", &Model::nforces, nb::arg("k"))
      .def("printFramesAndJoints", &Model::printFramesAndJoints)
      .def("getOrderedActuatedJoints", &Model::orderedActuatedJoints)
      .def(
        "forwardKinematics",
        [](Model &self, const VecX &q, const std::string &frame)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }
            return self.forwardKinematics(q, frame);
        },
        nb::arg("q"),
        nb::arg("frame"))
      .def(
        "forwardKinematicsSE3",
        [](Model &self, const VecX &q, const std::string &frame)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }
            return self.forwardKinematicsSE3(q, frame);
        },
        nb::arg("q"),
        nb::arg("frame"))
      .def("inverseDynamics",
           [](Model &self, const VecX &q, const VecX &v, const VecX &a)
           {
               if (!self.urdf())
               {
                   throw std::runtime_error("Model was not loaded from URDF");
               }
               return self.inverseDynamics(q, v, a);
           })
      .def(
        "inverseDynamics",
        [](Model &self,
           const VecX &q,
           const VecX &v,
           const VecX &a,
           const std::unordered_map<std::string, std::vector<Vec3>> &forces)
        {
            if (!self.urdf())
            {
                throw std::runtime_error("Model was not loaded from URDF");
            }
            return self.inverseDynamics(q, v, a, forces);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("a"),
        nb::arg("forces"));
}
}
}
