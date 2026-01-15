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
    nb::class_<StageCost, PyStageCost>(m, "StageCost")
      .def(nb::init())
      .def("evaluate", &StageCost::evaluate, nb::arg("x"), nb::arg("u"))
      .def(
        "gradient",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            VecX dx, du;
            self.gradient(x, u, dx, du);
            return nb::make_tuple(dx, du);
        },
        nb::arg("x"),
        nb::arg("u"))
      .def(
        "hessian_xx",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Hxx;
            self.hessian_xx(x, u, Hxx);
            return Hxx;
        },
        nb::arg("x"),
        nb::arg("u"))
      .def(
        "hessian_xu",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Hxu;
            self.hessian_xu(x, u, Hxu);
            return Hxu;
        },
        nb::arg("x"),
        nb::arg("u"))
      .def(
        "hessian_uu",
        [](const StageCost &self, const VecX &x, const VecX &u)
        {
            MatX Huu;
            self.hessian_uu(x, u, Huu);
            return Huu;
        },
        nb::arg("x"),
        nb::arg("u"));

    nb::class_<SquaredJointAngleCost, StageCost>(m, "SquaredJointAngleCost")
      .def(nb::init<uint, uint, double, bool>(),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
      .def("set_weight", &SquaredJointAngleCost::set_weight, nb::arg("weight"));

    nb::class_<SquaredJointVelocityCost, StageCost>(m,
                                                    "SquaredJointVelocityCost")
      .def(nb::init<uint, double, bool>(),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false)
      .def("set_weight",
           &SquaredJointVelocityCost::set_weight,
           nb::arg("weight"));

    nb::class_<SquaredControlCost, StageCost>(m, "SquaredControlCost")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"))
      .def("set_weight", &SquaredControlCost::set_weight, nb::arg("weight"));

    nb::class_<SquaredForceCost, StageCost>(m, "SquaredForceCost")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"))
      .def("set_weight", &SquaredForceCost::set_weight, nb::arg("weight"));

    nb::class_<FrameSquaredTranslationErrorCost, StageCost>(
      m,
      "FrameSquaredTranslationErrorCost")
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
      .def("set_desired_translation",
           &FrameSquaredTranslationErrorCost::set_desired_translation,
           nb::arg("desired_translation"))
      .def("set_weight",
           &FrameSquaredTranslationErrorCost::set_weight,
           nb::arg("weight"));

    nb::class_<FrameSquaredRotationErrorCost, StageCost>(
      m,
      "FrameSquaredRotationErrorCost")
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
      "FrameSquaredPoseErrorCost")
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

    nb::class_<FrameSquaredVelocityCost, StageCost>(m,
                                                    "FrameSquaredVelocityCost")
      .def(nb::init<const Model &, const std::string &, uint, uint, double>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"))
      .def("set_weight",
           &FrameSquaredVelocityCost::set_weight,
           nb::arg("weight"));
}
}
}
