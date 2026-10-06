#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/predefined/Costs.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;
using namespace model::stagewise::predefined::costs;

struct __attribute__((visibility("default"))) PyStageCost : StageCost
{
    NB_TRAMPOLINE(StageCost, 5);

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline, "evaluate", true);
            return (Scalar() << nb::cast<double>(
                      nb_trampoline.base().attr(nb_ticket.key)(x, u)))
              .finished();
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error("Type mismatch in Python override of "
                                     "`StageCost::evaluate`: expected a float");
        }
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline, "gradient", true);
            std::tie(dx, du) = nb::cast<std::pair<VecX, VecX>>(
              nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageCost::gradient`: expected a vector (numpy.ndarray)");
        }
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessian_xx",
                                               true);
            Hxx =
              nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageCost::hessian_xx`: expected a matrix (numpy.ndarray)");
        }
    }

    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessian_xu",
                                               true);
            Hxu =
              nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageCost::hessian_xu`: expected a matrix (numpy.ndarray)");
        }
    }

    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        try
        {
            nanobind::detail::ticket nb_ticket(nb_trampoline,
                                               "hessian_uu",
                                               true);
            Huu =
              nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
        }
        catch (const nb::cast_error &e)
        {
            throw std::runtime_error(
              "Type mismatch in Python override of "
              "`StageCost::hessian_uu`: expected a matrix (numpy.ndarray)");
        }
    }
};

void bind_Costs(nb::module_ &m)
{
    nb::class_<StageCost, PyStageCost>(m,
                                       "StageCost",
                                       R"doc(
        General abstract constraint function for an optimization stage $k$: $\ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Defines the required functions for a stage cost, while providing 
        default implementations for some
        )doc")
      .def(nb::init())
      .def(nb::init<const std::string &>(), nb::arg("name") = "")
      .def("names", &StageCost::names, "Get all cost names for StageCost.")
      .def("set_enabled", &StageCost::set_enabled, nb::arg("enabled"), R"doc(
        Set enabled status for StageCost
        A stage always keeps at least one cost, so this has no effect if every
        cost on the stage is disabled
      )doc")
      .def("enabled", &StageCost::enabled, "Whether this cost is enabled")
      .def("evaluate", &StageCost::evaluate, nb::arg("x"), nb::arg("u"), R"doc(
        Evaluate function $f(\boldsymbol{x}, \boldsymbol{u})$
        Returns: output vector, $n \times 1$
      )doc")
      .def(
        "gradient",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            VecX dx, du;
            self.gradient(x, u, dx, du);
            return nb::make_tuple(dx, du);
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get cost Jacobians with respect to $\boldsymbol{x}$ and $\boldsymbol{u}$, evaluated at
        given $(\boldsymbol{x}, \boldsymbol{u})$: $\nabla_x \ell_k(\boldsymbol{x}, \boldsymbol{u})$ and $\nabla_u \ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
            dx: gradient with respect to $\boldsymbol{x}$, $n_{dx} \times 1$
            du: gradient with respect to $\boldsymbol{u}$, $n_u \times 1$
        )doc")
      .def(
        "hessian_xx",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Hxx;
            self.hessian_xx(x, u, Hxx);
            return Hxx;
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get cost function hessian matrix with respect to $\boldsymbol{xx}$, evaluated at given
        $(\boldsymbol{x}, \boldsymbol{u})$:
        $\nabla_{xx}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Hxx: Hessian matrix, $n_{dx} \times n_{dx}$
        )doc")
      .def(
        "hessian_xu",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Hxu;
            self.hessian_xu(x, u, Hxu);
            return Hxu;
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get cost function hessian matrix with respect to $\boldsymbol{xu}$, evaluated at given
        $(\boldsymbol{x}, \boldsymbol{u})$:
        $\nabla_{xu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Hxu: Hessian matrix, $n_{dx} \times n_u$
        )doc")
      .def(
        "hessian_uu",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Huu;
            self.hessian_uu(x, u, Huu);
            return Huu;
        },
        nb::arg("x"),
        nb::arg("u"),
        R"doc(
        Get cost function hessian matrix with respect to $\boldsymbol{uu}$, evaluated at given
        $(\boldsymbol{x}, \boldsymbol{u})$:
        $\nabla_{uu}^2 \ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Returns:
         Huu: Hessian matrix, $n_u \times n_u$
        )doc");

    nb::class_<SquaredJointAngleCost, StageCost>(m,
                                                 "SquaredJointAngleCost",
                                                 R"doc(
        Predefined squared joint angle cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{ang} \rVert_2^2$

        where:
         - $\boldsymbol{x}_{ang}$ is the angular (position) part of the state
           Note that this will ignore the first 7 elements if the model has a 
           floating base
         - $\boldsymbol{w}$ is the weight on the angular term

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`), appearing after the floating base (if present)
            )doc")
      .def(nb::init<const Model &, const double, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &, const VecX &, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_weight",
           static_cast<void (SquaredJointAngleCost::*)(const double)>(
             &SquaredJointAngleCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredJointAngleCost::*)(const VecX &)>(
             &SquaredJointAngleCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredJointVelocityCost, StageCost>(m,
                                                    "SquaredJointVelocityCost",
                                                    R"doc(
        Predefined squared joint velocity cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{x}_{vel} \rVert_2^2$

        where:
         - $\boldsymbol{x}_{vel}$ is the velocity part of the state
         - $\boldsymbol{w}$ is the weight on the velocity term

        Assumes:
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)    
                                                    )doc")
      .def(nb::init<const Model &, const double, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &, const VecX &, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_weight",
           static_cast<void (SquaredJointVelocityCost::*)(const double)>(
             &SquaredJointVelocityCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredJointVelocityCost::*)(const VecX &)>(
             &SquaredJointVelocityCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredControlCost, StageCost>(m, "SquaredControlCost", R"doc(
        Predefined squared control cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{ctrl} \rVert_2^2$

        where:
         - $\boldsymbol{u}_{ctrl}$ is the control part of the controls
         - $\boldsymbol{w}$ is the weight on the control term

        Assumes:
         - The model control is the first part of the control vector (size 
           `model.nu()`)
        )doc")
      .def(nb::init<const Model &, const double, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &, const VecX &, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_weight",
           static_cast<void (SquaredControlCost::*)(const double)>(
             &SquaredControlCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredControlCost::*)(const VecX &)>(
             &SquaredControlCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredIDTorqueCost, StageCost>(m, "SquaredIDTorqueCost", R"doc(
        Predefined squared inverse dynamics torque cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{\tau} \rVert_2^2$

        where:
         - $\boldsymbol{\tau}$ is the actuated-joint torque caused by the controls
         - $\boldsymbol{w}$ is the weight on the torque term

        Note that this cost assumes inverse dynamics is being used (i.e. treats
        controls as accelerations)

        Assumes:
         - The model is in Inverse Dynamics or Reduced Inverse Dynamics mode,
           such that the state is made up of configuration and velocity and
           the controls are made up of accelerations (including base, for
           Inverse Dynamics; actuated joints only, for Reduced Inverse
           Dynamics) and potential forces

        `k` (the model stage) only matters when there are forces present at
        that stage. Construct one instance per stage (rather than sharing a
        single instance across stages via Model.addCost) whenever forces
        vary between the stages a cost is added to
        )doc")
      .def(nb::init<const Model &, const double, const std::string &,
                    const uint>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "",
           nb::arg("k")    = 0)
      .def(nb::init<const Model &, const VecX &, const std::string &,
                    const uint>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "",
           nb::arg("k")    = 0)
      .def("set_weight",
           static_cast<void (SquaredIDTorqueCost::*)(const double)>(
             &SquaredIDTorqueCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredIDTorqueCost::*)(const VecX &)>(
             &SquaredIDTorqueCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredForceCost, StageCost>(m, "SquaredForceCost", R"doc(
        Predefined squared force cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{u}_{f} \rVert_2^2$

        where:
         - $\boldsymbol{u}_{f}$ is the force part of the controls
         - $\boldsymbol{w}$ is the weight on the force term

        Assumes:
         - The model forces make up the second and final part of the control vector,
           following the control part of size `model.nu()`, for any remaining
           elements
        )doc")
      .def(nb::init<const Model &, const double, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &, const VecX &, const std::string &>(),
           nb::arg("model"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_weight",
           static_cast<void (SquaredForceCost::*)(const double)>(
             &SquaredForceCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredForceCost::*)(const VecX &)>(
             &SquaredForceCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredConfigurationErrorCost, StageCost>(
      m,
      "SquaredConfigurationErrorCost",
      R"doc(
        Predefined squared configuration error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{q} - \boldsymbol{q}_d)\rVert_2^2$

        where:
         - $\boldsymbol{q}$ is the configuration part of the state
         - $\boldsymbol{q}_d$ is the desired configuration
         - $\boldsymbol{w}$ is the weight for the configuration error

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const VecX &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_configuration"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const VecX &,
                    const VecX &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_configuration"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_configuration",
           &SquaredConfigurationErrorCost::set_desired_configuration,
           nb::arg("desired_configuration"))
      .def("set_weight",
           static_cast<void (SquaredConfigurationErrorCost::*)(const double)>(
             &SquaredConfigurationErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredConfigurationErrorCost::*)(const VecX &)>(
             &SquaredConfigurationErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FloatingBaseSquaredTranslationErrorCost, StageCost>(
      m,
      "FloatingBaseSquaredTranslationErrorCost",
      R"doc(
        Predefined squared floating base translation error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{t}_f - \boldsymbol{t}_d)\rVert_2^2$

        where:
        - $\boldsymbol{t}_f$ is the floating base translation (in $\mathcal{R}^3$)
        - $\boldsymbol{t}_d$ is the desired translation of the floating base (in $\mathcal{R}^3$)
        - $\boldsymbol{w}$ is the weight on the translation error

        Assumes:
        - The model joint configuration is the first part of the state vector (size
        `model.nq()`)
        - The floating base translation is extracted from the base pose of the model
        state representation
      )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_translation",
           &FloatingBaseSquaredTranslationErrorCost::set_desired_translation,
           nb::arg("desired_translation"))
      .def(
        "set_weight",
        static_cast<void (FloatingBaseSquaredTranslationErrorCost::*)(
          const double)>(&FloatingBaseSquaredTranslationErrorCost::set_weight),
        nb::arg("weight"))
      .def(
        "set_weight",
        static_cast<void (FloatingBaseSquaredTranslationErrorCost::*)(
          const Vec3 &)>(&FloatingBaseSquaredTranslationErrorCost::set_weight),
        nb::arg("weight"));

    nb::class_<FloatingBaseSquaredRotationErrorCost, StageCost>(
      m,
      "FloatingBaseSquaredRotationErrorCost",
      R"doc(
        Predefined squared floating base rotation error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \log_{SO(3)}(R_f^\top R_d)\rVert_2^2$

        where:
         - $R_f$ is the floating base rotation (in $\mathit{SO3}$)
         - $R_d$ is the desired rotation of the floating base (in $\mathit{SO3}$)
         - $\boldsymbol{w}$ is the weight on the rotation error

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The floating base orientation is extracted from the base pose of the model
           state representation

      )doc")
      .def(nb::init<const Model &,
                    const Mat3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Mat3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_rotation",
           &FloatingBaseSquaredRotationErrorCost::set_desired_rotation,
           nb::arg("desired_rotation"))
      .def("set_weight",
           static_cast<void (FloatingBaseSquaredRotationErrorCost::*)(
             const double)>(&FloatingBaseSquaredRotationErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FloatingBaseSquaredRotationErrorCost::*)(
             const Vec3 &)>(&FloatingBaseSquaredRotationErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FloatingBaseSquaredPoseErrorCost, StageCost>(
      m,
      "FloatingBaseSquaredPoseErrorCost",
      R"doc(
        Predefined squared floating base pose error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} D^\top W D$

        where:
         - $D = \log_{SE(3)} T^{-1}T_d$
         - $T$ is the floating base pose (in $\mathit{SE3}$)
         - $T_d$ is the desired floating base pose (in $\mathit{SE3}$)
         - $W$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
           for rotation

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)

      )doc")
      .def(nb::init<const Model &,
                    const Mat4 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_pose"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Mat4 &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_pose"),
           nb::arg("translation_weight"),
           nb::arg("rotation_weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Mat4 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_pose"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Mat3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Mat3 &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("translation_weight"),
           nb::arg("rotation_weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Mat3 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_pose",
           &FloatingBaseSquaredPoseErrorCost::set_desired_pose,
           nb::arg("desired_pose"))
      .def("set_desired_translation",
           &FloatingBaseSquaredPoseErrorCost::set_desired_translation,
           nb::arg("desired_translation"))
      .def("set_desired_rotation",
           &FloatingBaseSquaredPoseErrorCost::set_desired_rotation,
           nb::arg("desired_rotation"))
      .def("set_translation_weight",
           &FloatingBaseSquaredPoseErrorCost::set_translation_weight,
           nb::arg("translation_weight"))
      .def("set_rotation_weight",
           &FloatingBaseSquaredPoseErrorCost::set_rotation_weight,
           nb::arg("rotation_weight"))
      .def(
        "set_weight",
        static_cast<void (FloatingBaseSquaredPoseErrorCost::*)(const double)>(
          &FloatingBaseSquaredPoseErrorCost::set_weight),
        nb::arg("weight"))
      .def(
        "set_weight",
        static_cast<void (FloatingBaseSquaredPoseErrorCost::*)(const Vec6 &)>(
          &FloatingBaseSquaredPoseErrorCost::set_weight),
        nb::arg("weight"));

    nb::class_<FloatingBaseSquaredVelocityErrorCost, StageCost>(
      m,
      "FloatingBaseSquaredVelocityErrorCost",
      R"doc(
        Predefined squared floating base velocity error cost

        Defines the cost term
        $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{v}_b - \boldsymbol{v}_d)\rVert_2^2$

        where:
         - $\boldsymbol{v}_b$ is the floating base velocity (in $\mathcal{R}^6$)
         - $\boldsymbol{v}_d$ is the desired floating base velocity (in $\mathcal{R}^6$)
         - $\boldsymbol{w}$ is the weight on the velocity error

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
      )doc")
      .def(nb::init<const Model &,
                    const Vec6 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_velocity"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("desired_velocity"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_velocity",
           &FloatingBaseSquaredVelocityErrorCost::set_desired_velocity,
           nb::arg("desired_velocity"))
      .def("set_weight",
           static_cast<void (FloatingBaseSquaredVelocityErrorCost::*)(
             const double)>(&FloatingBaseSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FloatingBaseSquaredVelocityErrorCost::*)(
             const Vec6 &)>(&FloatingBaseSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FrameSquaredTranslationErrorCost, StageCost>(
      m,
      "FrameSquaredTranslationErrorCost",
      R"doc(
        Predefined squared frame translation error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{t} - \boldsymbol{t}_d)\rVert_2^2$

        where:
         - $\boldsymbol{t}$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
         - $\boldsymbol{t}_d$ is the desired translation for that frame (in $\mathcal{R}^3$)
         - $\boldsymbol{w}$ is the weight on the translation error

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_translation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_translation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_translation",
           &FrameSquaredTranslationErrorCost::set_desired_translation,
           nb::arg("desired_translation"))
      .def(
        "set_weight",
        static_cast<void (FrameSquaredTranslationErrorCost::*)(const double)>(
          &FrameSquaredTranslationErrorCost::set_weight),
        nb::arg("weight"))
      .def(
        "set_weight",
        static_cast<void (FrameSquaredTranslationErrorCost::*)(const Vec3 &)>(
          &FrameSquaredTranslationErrorCost::set_weight),
        nb::arg("weight"));

    nb::class_<FrameSquaredTerrainHeightErrorCost, StageCost>(
      m,
      "FrameSquaredTerrainHeightErrorCost",
      R"doc(
        Predefined squared frame terrain height error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} w^2 (p_z - h(p_x, p_y) - d_d)^2$

        where:
         - $\boldsymbol{p}$ is the frame translation for a given frame of the model (in $\mathcal{R}^3$)
         - $h$ is the height of the model's terrain at the frame's horizontal position
         - $d_d$ is the desired height of the frame above the terrain
         - $w$ is the weight on the height error

        Assumes:
         - The model joint configuration is the first part of the state vector (size
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_height"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_height",
           &FrameSquaredTerrainHeightErrorCost::set_desired_height,
           nb::arg("desired_height"))
      .def("set_weight",
           &FrameSquaredTerrainHeightErrorCost::set_weight,
           nb::arg("weight"));

    nb::class_<FrameSquaredRotationErrorCost, StageCost>(
      m,
      "FrameSquaredRotationErrorCost",
      R"doc(
        Predefined squared frame rotation error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert w \log_{SO(3)}R^\top R_d\rVert_2^2$

        where:
         - $R$ is the frame rotation for a given frame of the model (in $\mathit{SO3}$)
         - $R_d$ is the desired rotation for that frame (in $\mathit{SO3}$)
         - $\boldsymbol{w}$ is the weight on the rotation error

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_rotation",
           &FrameSquaredRotationErrorCost::set_desired_rotation,
           nb::arg("desired_rotation"))
      .def("set_weight",
           static_cast<void (FrameSquaredRotationErrorCost::*)(const double)>(
             &FrameSquaredRotationErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredRotationErrorCost::*)(const Vec3 &)>(
             &FrameSquaredRotationErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FrameSquaredPoseErrorCost, StageCost>(
      m,
      "FrameSquaredPoseErrorCost",
      R"doc(
        Predefined squared frame pose error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} D^\top W D$

        where:
         - $D = \log_{SE(3)} T^{-1}T_d$
         - $T$ is the frame pose for a given frame of the model (in $\mathit{SE3}$)
         - $T_d$ is the desired pose for that frame (in $\mathit{SE3}$)
         - $W$ is a weight matrix, made up of parameters $w_p$ for translation and $w_r$
           for rotation

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat4 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_pose"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat4 &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_pose"),
           nb::arg("translation_weight"),
           nb::arg("rotation_weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Mat4 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_pose"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Mat3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Mat3 &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("translation_weight"),
           nb::arg("rotation_weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec3 &,
                    const Mat3 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_translation"),
           nb::arg("desired_rotation"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_pose",
           &FrameSquaredPoseErrorCost::set_desired_pose,
           nb::arg("desired_pose"))
      .def("set_desired_translation",
           &FrameSquaredPoseErrorCost::set_desired_translation,
           nb::arg("desired_translation"))
      .def("set_desired_rotation",
           &FrameSquaredPoseErrorCost::set_desired_rotation,
           nb::arg("desired_rotation"))
      .def("set_translation_weight",
           &FrameSquaredPoseErrorCost::set_translation_weight,
           nb::arg("translation_weight"))
      .def("set_rotation_weight",
           &FrameSquaredPoseErrorCost::set_rotation_weight,
           nb::arg("rotation_weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredPoseErrorCost::*)(const double)>(
             &FrameSquaredPoseErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredPoseErrorCost::*)(const Vec6 &)>(
             &FrameSquaredPoseErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FrameSquaredVelocityErrorCost, StageCost>(
      m,
      "FrameSquaredVelocityErrorCost",
      R"doc(
        Predefined squared frame velocity error cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{v}_f - \boldsymbol{v}_d)\rVert_2^2$

        where:
         - $\boldsymbol{v}_f$ is the frame velocity for a given frame of the model (in $\mathcal{R}^6$)
         - $\boldsymbol{v}_d$ is the desired velocity for that frame (in $\mathcal{R}^6$)
         - $\boldsymbol{w}$ is the weight on the velocity error

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
         - The model joint velocity is the final part of the state vector (size
           `model.nv()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec6 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_velocity"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Vec6 &,
                    const Vec6 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("desired_velocity"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_desired_velocity",
           &FrameSquaredVelocityErrorCost::set_desired_velocity,
           nb::arg("desired_velocity"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityErrorCost::*)(const double)>(
             &FrameSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityErrorCost::*)(const Vec6 &)>(
             &FrameSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<CenterOfMassTrackingCost, StageCost>(m,
                                                    "CenterOfMassTrackingCost",
                                                    R"doc(
        Predefined center of mass (CoM) tracking cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{c} - \boldsymbol{c}_r)\rVert_2^2$

        where:
         - $\boldsymbol{c}$ is the center of mass for the model (in $\mathcal{R}^3$)
         - $\boldsymbol{c}_r$ is the reference point for the center of mass (in $\mathcal{R}^3$)

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
                                                    )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("cref"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("cref"),
           nb::arg("weight"),
           nb::arg("name") = "")
      .def("set_weight",
           static_cast<void (CenterOfMassTrackingCost::*)(const double)>(
             &CenterOfMassTrackingCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (CenterOfMassTrackingCost::*)(const Vec3 &)>(
             &CenterOfMassTrackingCost::set_weight),
           nb::arg("weight"))
      .def("set_cref", &CenterOfMassTrackingCost::set_cref, nb::arg("cref"));

    nb::class_<FramesInverseSquaredDistanceCost, StageCost>(
      m,
      "FramesInverseSquaredDistanceCost",
      R"doc(
        Predefined frames squared inverse distance cost

        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \sum_{i < j} \frac{w}{2 \lVert \boldsymbol{\sigma} \odot (\boldsymbol{t}_i - \boldsymbol{t}_j)\rVert_2^2}$

        where:
         - $\boldsymbol{t}_i$ is the frame translation for frame $i$ of the model (in $\mathcal{R}^3$)
         - $w$ is the weight on the translation error
         - $\boldsymbol{\sigma}$ is the distance scale, a per-axis vector (in $\mathcal{R}^3$) determining the
           scaling factor for each axis; the factors work in an inverse manner, i.e.
           larger values place less importance on an axis, while smaller ones place
           more

        Assumes:
         - The model joint configuration is the first part of the state vector (size 
           `model.nq()`)
      )doc")
      .def(nb::init<const Model &,
                    const std::vector<std::string> &,
                    const double,
                    const double,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frames"),
           nb::arg("weight"),
           nb::arg("distance_scale") = 100.0,
           nb::arg("name")           = "")
      .def(nb::init<const Model &,
                    const std::vector<std::string> &,
                    const double,
                    const Vec3 &,
                    const std::string &>(),
           nb::arg("model"),
           nb::arg("frames"),
           nb::arg("weight"),
           nb::arg("distance_scale"),
           nb::arg("name") = "")
      .def(
        "set_weight",
        static_cast<void (FramesInverseSquaredDistanceCost::*)(const double)>(
          &FramesInverseSquaredDistanceCost::set_weight),
        nb::arg("weight"))
      .def(
        "set_distance_scale",
        static_cast<void (FramesInverseSquaredDistanceCost::*)(const double)>(
          &FramesInverseSquaredDistanceCost::set_distance_scale),
        nb::arg("distance_scale"))
      .def(
        "set_distance_scale",
        static_cast<void (FramesInverseSquaredDistanceCost::*)(const Vec3 &)>(
          &FramesInverseSquaredDistanceCost::set_distance_scale),
        nb::arg("distance_scale"));
}
}
}
