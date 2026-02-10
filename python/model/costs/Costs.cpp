#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
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
        nanobind::detail::ticket nb_ticket(nb_trampoline, "evaluate", true);
        return (Scalar() << nb::cast<double>(
                  nb_trampoline.base().attr(nb_ticket.key)(x, u)))
          .finished();
    }

    void gradient(const VecX &x, const VecX &u, VecX &dx,
                  VecX &du) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "gradient", true);
        std::tie(dx, du) = nb::cast<std::pair<VecX, VecX>>(
          nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessian_xx(const VecX &x, const VecX &u, MatX &Hxx) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessian_xx", true);
        Hxx = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessian_xu(const VecX &x, const VecX &u, MatX &Hxu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessian_xu", true);
        Hxu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessian_uu(const VecX &x, const VecX &u, MatX &Huu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessian_uu", true);
        Huu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }
};

void bind_Costs(nb::module_ &m)
{
    nb::class_<StageCost, PyStageCost>(m, "StageCost", R"doc(
        General abstract constraint function for an optimization stage $k$: $\ell_k(\boldsymbol{x}, \boldsymbol{u})$
        Defines the required functions for a stage cost, while providing 
        default implementations for some
        )doc")
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
            )doc")
      .def(nb::init<uint, uint, double, bool>(),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
      .def(nb::init<uint, uint, const VecX &, bool>(),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
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
                                                    )doc")
      .def(nb::init<uint, double, bool>(),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
      .def(nb::init<uint, const VecX &, bool>(),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
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
        )doc")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"))
      .def(nb::init<uint, const VecX &>(), nb::arg("nv"), nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredControlCost::*)(const double)>(
             &SquaredControlCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredControlCost::*)(const VecX &)>(
             &SquaredControlCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredForceCost, StageCost>(m, "SquaredForceCost", R"doc(
        Predefined squared force cost
        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u})=w_f \boldsymbol{u}_f$
        where $\boldsymbol{u}_f$ is the force part of the controls
        and $w_f$ is the weight on the term
        )doc")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"))
      .def(nb::init<uint, const VecX &>(), nb::arg("nv"), nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredForceCost::*)(const double)>(
             &SquaredForceCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (SquaredForceCost::*)(const VecX &)>(
             &SquaredForceCost::set_weight),
           nb::arg("weight"));

    nb::class_<SquaredConfigurationDifferenceCost, StageCost>(
      m,
      "SquaredConfigurationDifferenceCost",
      R"doc(
        Predefined squared configuration difference cost
        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot (\boldsymbol{q} - \boldsymbol{q}_d)\rVert_2^2$
        where:
         - $\boldsymbol{q}$ is the configuration part of the state
         - $\boldsymbol{q}_d$ is the desired configuration
         - $\boldsymbol{w}$ is the weight for the difference
      )doc")
      .def(nb::init<const VecX &, uint, uint, double, bool>(),
           nb::arg("desired_configuration"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer"))
      .def(nb::init<const VecX &, uint, uint, const VecX &, bool>(),
           nb::arg("desired_configuration"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer"))
      .def("set_desired_configuration",
           &SquaredConfigurationDifferenceCost::set_desired_configuration,
           nb::arg("desired_configuration"))
      .def(
        "set_weight",
        static_cast<void (SquaredConfigurationDifferenceCost::*)(const double)>(
          &SquaredConfigurationDifferenceCost::set_weight),
        nb::arg("weight"))
      .def(
        "set_weight",
        static_cast<void (SquaredConfigurationDifferenceCost::*)(const VecX &)>(
          &SquaredConfigurationDifferenceCost::set_weight),
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
      )doc")
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const std::string &,
                    uint,
                    uint,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const std::string &,
                    uint,
                    uint,
                    const Vec3 &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
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
      )doc")
      .def(nb::init<const Model &,
                    const Mat3 &,
                    const std::string &,
                    uint,
                    uint,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_rotation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def("set_desired_rotation",
           &FrameSquaredRotationErrorCost::set_desired_rotation,
           nb::arg("desired_rotation"))
      .def("set_weight",
           &FrameSquaredRotationErrorCost::set_weight,
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
      )doc")
      .def(nb::init<const Model &,
                    const Mat4 &,
                    const std::string &,
                    uint,
                    uint,
                    double,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_pose"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("position_weight"),
           nb::arg("rotation_weight"))
      .def(nb::init<const Model &,
                    const Vec3 &,
                    const Mat3 &,
                    const std::string &,
                    uint,
                    uint,
                    double,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_position"),
           nb::arg("desired_rotation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("position_weight"),
           nb::arg("rotation_weight"))
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
           nb::arg("rotation_weight"));

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
      )doc")
      .def(nb::init<const Model &,
                    const Vec6 &,
                    const std::string &,
                    uint,
                    uint,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def(nb::init<const Model &,
                    const Vec6 &,
                    const std::string &,
                    uint,
                    uint,
                    const Vec6 &>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def("set_desired_velocity",
           &FrameSquaredVelocityErrorCost::set_desired_velocity,
           nb::arg("desired_translation"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityErrorCost::*)(const double)>(
             &FrameSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityErrorCost::*)(const Vec6 &)>(
             &FrameSquaredVelocityErrorCost::set_weight),
           nb::arg("weight"));

    nb::class_<FrameSquaredVelocityCost, StageCost>(m,
                                                    "FrameSquaredVelocityCost",
                                                    R"doc(
        Predefined squared frame velocity cost
        Defines the cost term $\ell(\boldsymbol{x}, \boldsymbol{u}) = \frac{1}{2} \lVert \boldsymbol{w} \odot \boldsymbol{v}_{f} \rVert_2^2$
        where:
         - $\boldsymbol{v}_f$ is the (6D) velocity for a frame
         - $\boldsymbol{w}$ is the weight on the velocity term
                                                    )doc")
      .def(nb::init<const Model &, const std::string &, uint, uint, double>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def(
        nb::
          init<const Model &, const std::string &, uint, uint, const Vec6 &>(),
        nb::arg("model"),
        nb::arg("frame"),
        nb::arg("nq"),
        nb::arg("nv"),
        nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityCost::*)(const double)>(
             &FrameSquaredVelocityCost::set_weight),
           nb::arg("weight"))
      .def("set_weight",
           static_cast<void (FrameSquaredVelocityCost::*)(const Vec6 &)>(
             &FrameSquaredVelocityCost::set_weight),
           nb::arg("weight"));
}
}
}
