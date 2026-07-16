#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unordered_map.h>
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

    for (uint k = 0; k <= model->horizon(); ++k)
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

    for (uint k = 0; k <= model->horizon(); ++k)
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
    nb::enum_<SolverMode>(m, "SolverMode", R"doc(
        Solver Mode
        The mode the solver uses for the system formulation, corresponding to the
        equality constraint of the NLP
        This can either be full dynamics, or a kinematics formulation
        Options:
         - Kinematics: The solver optimizes in a kinematic mode. This means there are
           no system dynamics. Instead, the generated controls correspond to velocities,
           while the state corresponds to the system configuration.
         - ForwardDynamics: The solver optimizes with a forward dynamics approach. The 
           state corresponds to a stacked vector of the system configuration and velocity.
           The generated controls correspond to torques which, when applied over time
           to the system, generate the output trajectory.
         - InverseDynamics: The solver optimizes with an inverse dynamics approach. The 
           state corresponds to a stacked vector of the system configuration and velocity.
           The generated controls correspond to accelerations. This simplifies the
           LQR formulation and can sometimes lead to more robust solutions. Inverse
           dynamics is used as a constraint, ensuring the torques producing the
           generated acceleration controls are within the required effort limits.
         - ReducedInverseDynamics: This mode is similar to the Inverse dynamics mode. In 
           this mode, the floating base (if present) and/or some subset of the model's
           joints are unactuated. While regular Inverse mode enforces this via
           equality constraints, this mode instead enforces this implicitly through
           state propagation.
        )doc")
      .value("Kinematics", SolverMode::Kinematics)
      .value("ForwardDynamics", SolverMode::ForwardDynamics)
      .value("InverseDynamics", SolverMode::InverseDynamics)
      .value("ReducedInverseDynamics", SolverMode::ReducedInverseDynamics)
      .export_values();

    nb::enum_<IntegrationScheme>(m, "Integration", R"doc(
        The type of integration used for the system dynamics
        Options:
        - ExplicitEuler:
            $\boldsymbol{x}_{n+1}=\boldsymbol{x}_n+\boldsymbol{v}_ndt$
            $\boldsymbol{v}_{n+1}=\boldsymbol{v}_n+\boldsymbol{a}_ndt$
        - SemiImplicitEuler:
            $\boldsymbol{x}_{n+1}=\boldsymbol{x}_n+\boldsymbol{v}_{n+1}dt$
            $\boldsymbol{v}_{n+1}=\boldsymbol{v}_n+\boldsymbol{a}_ndt$
        )doc")
      .value("ExplicitEuler", IntegrationScheme::ExplicitEuler)
      .value("SemiImplicitEuler", IntegrationScheme::SemiImplicitEuler)
      .export_values();

    nb::class_<Model::CustomProperties>(
      m,
      "CustomProperties",
      "Optional custom properties for non-XML models.")
      .def(nb::init())
      .def_rw("nq", &Model::CustomProperties::nq, "Joint configuration size.")
      .def_rw("nv", &Model::CustomProperties::nv, "Joint velocity size")
      .def_rw("nu", &Model::CustomProperties::nu, "Control size")
      .def_rw("njoints",
              &Model::CustomProperties::njoints,
              "Number of actuated joints.")
      .def_rw("floating_base",
              &Model::CustomProperties::floating_base,
              "Whether the model has a floating base.")
      .def_rw("ordered_joints",
              &Model::CustomProperties::ordered_joints,
              "Ordered joint names.")
      .def_rw(
        "joint_types",
        &Model::CustomProperties::joint_types,
        "The types of joints corresponding to each of the ordered joints.");

    nb::class_<Model>(m, "Model", R"doc(
        Defines an SSQP model
        An SSQP model object is a prerequisite to performing optimization using SSQP
        The model contains the following:
         - An optimization horizon, $H$
         - A cost function for each stage of the horizon, $\ell_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}$, $0 \leq k \leq H$
         - Dynamics functions for each stage of the horizon, $\boldsymbol{f}_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}^n$, $0 \leq k \leq H$
         - Constraint functions for each stage of the horizon, $\boldsymbol{c}_k(\boldsymbol{x}, \boldsymbol{u}) \in \mathcal{R}^c$, $0 \leq k \leq H$

        Of the above, all stage cost and dynamics functions must be defined, but
        constraints may be left empty.

        Note that models optimize for $H$ total steps while considering the initial
        state to be constant, whereas the initial controls can be altered. This means
        that there are $H$ total controls ($\boldsymbol{u}_0$ to $\boldsymbol{u}_{H-1}$) and $H + 1$ total states ($\boldsymbol{x}_0$ to 
        $\boldsymbol{x}_H$), with $\boldsymbol{x}_0$ being constant. At the terminal stage (stage $H$), there are no
        controls and, as such, stagewise functions must be appropriately defined to
        ignore them. Predefined functions all have constructors to account for this.

        Models may be defined either by passing custom function implementations, or
        by using XML description files. Some constructors allow for defining a model
        using an XML file, but without passing the required cost functions. This is 
        done to allow access to robot-related algorithms such as forward dynamics,
        which may be used to define model costs. Note that, if such constructors are
        used, the `finalize` method must be called before attempting to use the model
        for optimization. The user is responsible for either manually assigning all
        required costs, or passing them to the `finalize` method.
        )doc")
      .def(
        "__init__",
        [](Model *self, uint horizon, double dt, nb::kwargs kwargs)
        {
            Model::Structure structure;

            if (kwargs.contains("urdf_path"))
            {
                structure.urdf_path =
                  nb::cast<std::string>(kwargs["urdf_path"]);
            }
            if (kwargs.contains("mjcf_path"))
            {
                structure.mjcf_path =
                  nb::cast<std::string>(kwargs["mjcf_path"]);
            }
            if (kwargs.contains("free_flyer"))
            {
                structure.free_flyer = nb::cast<bool>(kwargs["free_flyer"]);
            }
            if (kwargs.contains("solver_mode"))
            {
                structure.solver_mode =
                  nb::cast<SolverMode>(kwargs["solver_mode"]);
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
        },
        nb::arg("horizon"),
        nb::arg("dt"),
        nb::arg("kwargs"),
        R"doc(
        Initialize an SSQP Model.

        Positional arguments (required):
            `horizon` (int): Optimization horizon (excluding initial stage)
            `dt` (float): Optimization time step

        Keyword arguments (optional):
            `urdf_path` (str): Path to URDF file to use for the system dynamics
            `mjcf_path` (str): Path to MJCF file to use for the system dynamics
            `free_flyer` (bool): Whether the model has a floating base
            `solver_mode` (ssqpy.model.SolverMode): The solver mode to solve in
            `integration` (ssqpy.model.Integration): The integration method to use
            `ignore_innate_constraints` (bool): Whether to ignore XML constraints
            `actuated_joints` (list of int): List of joints to apply controls to;
                if non-empty, all joints whose indices are not included will not be
                actuated
           )doc")
      .def(
        "finalize",
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
                config.terminal_cost =
                  nb::cast<std::shared_ptr<StageCost>>(kwargs["terminal_cost"]);
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
                if (nb::isinstance<std::vector<VecX>>(kwargs["init_states"]))
                {
                    config.init_states =
                      nb::cast<std::vector<VecX>>(kwargs["init_states"]);
                }
                else //if (nb::isinstance<VecX>(kwargs["init_states"]))
                {
                    config.init_states = { nb::cast<VecX>(
                      kwargs["init_states"]) };
                }
            }
            if (kwargs.contains("init_controls"))
            {
                if (nb::isinstance<std::vector<VecX>>(kwargs["init_controls"]))
                {
                    config.init_controls =
                      nb::cast<std::vector<VecX>>(kwargs["init_controls"]);
                }
                else //if (nb::isinstance<VecX>(kwargs["init_controls"]))
                {
                    config.init_controls = { nb::cast<VecX>(
                      kwargs["init_controls"]) };
                }
            }

            self->finalize(config);
        },
        R"doc(
        Finalize the model with the provided configuration.

        Keyword arguments (optional):
            `costs` (list of StageCost): Costs for each optimization stage
            `stage_cost` (StageCost): Cost to add to each non-terminal stage
            `terminal_cost` (StageCost): Cost for the terminal stage
            `constraints` (list of StageConstraints): Constraints for each optimization stage
            `dynamics` (list of StageDynamics): Dynamics for each optimization stage
            `init_states` (np.array or list of np.array): Initial states for each optimization stage;
                if np.array: initial state to use for every stage
            `init_controls` (np.array or list of np.array): Initial controls for each optimization stage;
                if np.array: initial controls to use for every stage
        )doc")
      .def("updateDimensions", &Model::updateDimensions, R"doc(
        Updates the model's dimensions and linear dimensions
        This is required to update constraint sizes, when constraints' enabled
        state changes
        )doc")
      .def(
        "addCost",
        [](Model &self, const uint k, nb::object py_cost)
        {
            auto *raw  = nb::cast<StageCost *>(py_cost);
            auto  cost = std::shared_ptr<StageCost>(
              raw,
              [ref = nb::object(py_cost)](StageCost *) {});

            self.addCost(k, cost);
        },
        nb::arg("k"),
        nb::arg("cost"),
        "Add a cost to a stage.")
      .def("addConstraints",
           static_cast<void (
             Model::*)(const uint, const std::shared_ptr<StageConstraints> &)>(
             &Model::addConstraints),
           nb::arg("k"),
           nb::arg("constraints"),
           "Add constraints to a stage.")
      .def(
        "addContactConstraint",
        [](Model             &self,
           const uint         k,
           const std::string &frame,
           const double       mu,
           const double       max_normal_force,
           const double       eq_tol,
           const std::string &name)
        {
            return self.addConstraints<ContactConstraint>(k,
                                                          self,
                                                          frame,
                                                          mu,
                                                          max_normal_force,
                                                          eq_tol,
                                                          name);
        },
        nb::arg("k"),
        nb::arg("frame"),
        nb::arg("mu")               = 0.6,
        nb::arg("max_normal_force") = std::numeric_limits<double>::infinity(),
        nb::arg("eq_tol")           = 0.0,
        nb::arg("name")             = "",
        R"doc(
        Add a contact constraint for a frame to a stage.
        This includes adding a force for the contact constraint at that stage.
        )doc")
      .def_prop_ro(
        "finalized",
        [](Model &self) { return self.finalized(); },
        "Whether the model has been finalized.")
      .def_prop_ro(
        "xml",
        [](Model &self) { return self.xml(); },
        "Whether the model has been loaded from an XML file.")
      .def_prop_ro(
        "horizon",
        [](Model &self) { return self.horizon(); },
        "Get the optimization horizon.")
      .def_prop_ro(
        "dt",
        [](Model &self) { return self.dt(); },
        "Get the optimization time step.")
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
        nb::arg("k"),
        "Get the state size of the model.")
      .def(
        "getLinearStateSize",
        [](Model &self, const uint k)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }

            return self.linear_dims().stage_state[k];
        },
        "Get the state size of the model after state linearization.")
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
        nb::arg("k"),
        "Get the control size of the model.")
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
        nb::arg("k"),
        "Get the constraint size of the model.")
      .def(
        "getLinearConstraintSize",
        [](Model &self, const uint k)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }

            return self.linear_dims().stage_constraint[k];
        },
        "Get the constraint size of the model after linearization.")
      .def(
        "getnq",
        [](Model &self)
        {
            if (!self.xml() && !self.hasCustomProperties())
            {
                throw std::runtime_error(
                  "Model was not loaded from XML and has no custom properties");
            }

            return self.nq();
        },
        R"doc(
        Get the configuration size of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "getnv",
        [](Model &self)
        {
            if (!self.xml() && !self.hasCustomProperties())
            {
                throw std::runtime_error(
                  "Model was not loaded from XML and has no custom properties");
            }

            return self.nv();
        },
        R"doc(
        Get the velocity size of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "getnu",
        [](Model &self)
        {
            if (!self.xml() && !self.hasCustomProperties())
            {
                throw std::runtime_error(
                  "Model was not loaded from XML and has no custom properties");
            }

            return self.nu();
        },
        R"doc(
        Get the non-force control size of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "getnjoints",
        [](Model &self)
        {
            if (!self.xml() && !self.hasCustomProperties())
            {
                throw std::runtime_error(
                  "Model was not loaded from XML and has no custom properties");
            }

            return self.njoints();
        },
        R"doc(
        Get the number of actuated joints of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "hasFloatingBase",
        [](Model &self, const uint k)
        {
            if (!self.xml() && !self.hasCustomProperties())
            {
                throw std::runtime_error(
                  "Model was not loaded from XML and has no custom properties");
            }

            return self.floatingBase();
        },
        R"doc(
        Whether the model has a floating base.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def("getSolverMode",
           &Model::solverMode,
           "Get the solver mode formulation")
      .def(
        "getCost",
        [](Model &self, const uint k) { return self.costs()[k]; },
        nb::arg("k"),
        "Get the current model cost at a stage.")
      .def(
        "getDynamics",
        [](Model &self, const uint k) { return self.dynamics()[k]; },
        nb::arg("k"),
        "Get the current model dynamics at a stage.")
      .def(
        "getConstraints",
        [](Model &self, const uint k) { return self.constraints()[k]; },
        nb::arg("k"),
        "Get the current model constraints at a stage.")
      .def_prop_ro(
        "terrain",
        [](Model &self) { return self.terrain(); },
        "Get the model's terrain.")
      .def(
        "setInitState",
        [](Model &self, const uint k, const VecX &init_state)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }
            if (init_state.size() != self.dynamics()[k]->state_size())
            {
                throw std::runtime_error("Invalid initial state size");
            }
            self.setInitState(k, init_state);
        },
        nb::arg("k"),
        nb::arg("init_state"),
        "Set initial state for stage k (must be finalized)")
      .def(
        "setInitControl",
        [](Model &self, const uint k, const VecX &init_control)
        {
            if (!self.finalized())
            {
                throw std::runtime_error("Model is not finalized");
            }
            if (init_control.size() != self.dynamics()[k]->control_size())
            {
                throw std::runtime_error("Invalid initial controls size");
            }
            self.setInitControl(k, init_control);
        },
        nb::arg("k"),
        nb::arg("init_state"),
        "Set initial state for stage k (must be finalized)")
      .def(
        "addForce",
        [](Model &self, const uint k, const std::string &frame)
        {
            if (self.finalized())
            {
                throw std::runtime_error("Model is finalized");
            }
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }

            std::shared_ptr<Force> f = self.addForce(k, frame);
            return f->force_index;
        },
        nb::arg("k"),
        nb::arg("frame"),
        R"doc(
        Add a force on a frame at a stage.
        Only defined if the model was initialized using an XML file.
        For this method to be called, the model must not be finalized.
        )doc")
      .def("getTotalForces", &Model::nforces, nb::arg("k"), R"doc(
        Get the current number of forces for a stage.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def("getOrderedForceFrames",
           &Model::orderedForceFrames,
           nb::arg("k"),
           "Vector containing frame names, ordered 1-1 with forces for stage k")
      .def("printFramesAndJoints", &Model::printFramesAndJoints, R"doc(
        Print the frames and joints of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def("getOrderedActuatedJoints", &Model::orderedActuatedJoints, R"doc(
        Get an ordered list of the actuated joints of the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "getJointTypes",
        &Model::jointTypes,
        "Types for each joint that is part of the state (including unactuated)")
      .def("hasFrame",
           &Model::hasFrame,
           nb::arg("frame"),
           "Whether the model has a frame with the given name")
      .def("setActuatedJoints",
           &Model::setActuatedJoints,
           nb::arg("actuated_joints"),
           "Set the actuated joint indices vector")
      .def(
        "forwardKinematics",
        [](Model &self, const VecX &q, const std::string &frame)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }
            return self.forwardKinematics(q, frame);
        },
        nb::arg("q"),
        nb::arg("frame"),
        R"doc(
        Perform forward kinematics on the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "forwardKinematicsSE3",
        [](Model &self, const VecX &q, const std::string &frame)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            if (!self.hasFrame(frame))
            {
                throw std::runtime_error("Provided frame does not exist");
            }
            return self.forwardKinematicsSE3(q, frame);
        },
        nb::arg("q"),
        nb::arg("frame"),
        R"doc(
        Perform forward kinematics on the model and return a transformation matrix.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "forwardDynamics",
        [](Model &self, const VecX &q, const VecX &v, const VecX &tau)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.forwardDynamics(q, v, tau);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("a"),
        R"doc(
        Perform forward dynamics on the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "forwardDynamics",
        [](Model &self,
           const VecX &q,
           const VecX &v,
           const VecX &tau,
           const std::unordered_map<std::string, std::vector<Vec3>> &forces)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.forwardDynamics(q, v, tau, forces);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("tau"),
        nb::arg("forces"),
        R"doc(
        Perform forward dynamics on the model with forces.
        Only defined if the model was initialized using an XML file.
        )doc")

      .def(
        "inverseDynamics",
        [](Model &self, const VecX &q, const VecX &v, const VecX &a)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.inverseDynamics(q, v, a);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("a"),
        R"doc(
        Perform inverse dynamics on the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "inverseDynamics",
        [](Model &self,
           const VecX &q,
           const VecX &v,
           const VecX &a,
           const std::unordered_map<std::string, std::vector<Vec3>> &forces)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.inverseDynamics(q, v, a, forces);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("a"),
        nb::arg("forces"),
        R"doc(
        Perform inverse dynamics on the model with forces.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def(
        "centerOfMass",
        [](Model &self, const VecX &q)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.centerOfMass(q);
        },
        nb::arg("q"),
        "Get model center of mass for the given configuration")
      .def(
        "frameJacobian",
        [](Model &self,
           const VecX &q,
           const std::string &frame,
           const bool local)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.frameJacobian(q, frame, local);
        },
        nb::arg("q"),
        nb::arg("frame"),
        nb::arg("local") = false,
        "Get a frame jacobian based on the XML model dynamics")
      .def(
        "jacobianCenterOfMass",
        [](Model &self, const VecX &q)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.jacobianCenterOfMass(q);
        },
        nb::arg("q"),
        "Get model center of mass jacobian for the given configuration")
      .def(
        "getAllDynamicParameters",
        [](Model &self)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.getAllDynamicParameters();
        },
        R"doc(
        Returns a stacked vector of the current values of the loaded model's
        dynamic parameters for every link
        This consist of n vectors of size 10, containing the following:
        $\boldsymbol{v} = [m, m_{c,x}, m_{c,y}, m_{c_z}, I_{xx}, I_{xy}, I_{yy}, I_{xz}, I_{yz}, I_{zz}]^\top$ for each link
        )doc")
      .def(
        "setAllDynamicParameters",
        [](Model &self, const VecX &parameters)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            self.setAllDynamicParameters(parameters);
        },
        nb::arg("parameters"),
        R"doc(
        Sets the dynamic parameters for every link of the loaded model to the
        provided stacked vector
        Input:
         - `parameters`: A stacked vector of the dynamic parameters for each 
            link of the model
           )doc")
      .def(
        "getJointTorqueRegressor",
        [](Model &self, const VecX &q, const VecX &v, const VecX &a)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.getJointTorqueRegressor(q, v, a);
        },
        nb::arg("q"),
        nb::arg("v"),
        nb::arg("a"),
        R"doc(
        Computes and returns the joint torque regressor matrix linking the joint 
        torque to the dynamic parameters of each link for the given 
        configuration, velocity and acceleration
        Input:
         - `q`: The model configuration
         - `v`: The model velocity
         - `a`: The model acceleration
            
        Returns: The joint torque regressor matrix  
           )doc")
      .def(
        "getFullAccelerationVector",
        [](Model &self, const VecX &x, const VecX &u, const uint k)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.getFullAccelerationVector(x, u, k);
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("k") = 0,
        R"doc(
        Returns the full acceleration vector for a given stage
        Input:
         - `x`: The model state
         - `u`: The model controls
         - `k`: The model stage (only required if model is in Reduced Inverse
            mode and there are present forces)
            
        Note: For Inverse dynamics mode, this function is trivial, since it only
        returns the non-force part of the control vector
        Note: XML-based function
        )doc")
      .def(
        "neutralPose",
        [](Model &self)
        {
            if (!self.xml())
            {
                throw std::runtime_error("Model was not loaded from XML");
            }
            return self.neutralPose();
        },
        R"doc(
        Get a configuration for the model.
        Only defined if the model was initialized using an XML file.
        )doc")
      .def("setCustomProperties",
           &Model::setCustomProperties,
           "Set custom properties for non-XML models")
      .def("hasCustomProperties",
           &Model::hasCustomProperties,
           "Whether custom properties have been set");
}
}
}
