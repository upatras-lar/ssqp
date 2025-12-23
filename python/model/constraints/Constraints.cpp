#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;
using namespace model::stagewise::predefined::constraints;

struct __attribute__((visibility("default"))) PyStageConstraints
  : StageConstraints
{
    NB_TRAMPOLINE(StageConstraints, 10);

    uint output_size() const override
    {
        NB_OVERRIDE_PURE(output_size);
    }

    VecX lower_bound() const override
    {
        NB_OVERRIDE(lower_bound);
    }

    VecX upper_bound() const override
    {
        NB_OVERRIDE(upper_bound);
    }

    uint constraint_diff_size() const override
    {
        NB_OVERRIDE(constraint_diff_size);
    }

    VecX right_minus(const VecX &x, const VecX &y) const override
    {
        NB_OVERRIDE(right_minus, x, y);
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "evaluate", true);
        return (Scalar() << nb::cast<double>(
                  nb_trampoline.base().attr(nb_ticket.key)(x, u)))
          .finished();
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "gradient", true);
        std::tie(dX, dU) = nb::cast<std::pair<MatX, MatX>>(
          nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_xx", true);
        Pxx = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_xu", true);
        Pxu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        nanobind::detail::ticket nb_ticket(nb_trampoline, "hessprod_uu", true);
        Puu = nb::cast<MatX>(nb_trampoline.base().attr(nb_ticket.key)(x, u));
    }
};

void bind_Constraints(nb::module_ &m)
{
    nb::class_<StageConstraints, PyStageConstraints>(m, "StageConstraints")
      .def(nb::init())
      .def("output_size", &StageConstraints::output_size)
      .def("lower_bound", &StageConstraints::lower_bound)
      .def("upper_bound", &StageConstraints::upper_bound)
      .def("constraint_diff_size", &StageConstraints::constraint_diff_size)
      .def("right_minus",
           &StageConstraints::right_minus,
           nb::arg("x"),
           nb::arg("u"))
      .def("evaluate", &StageConstraints::evaluate, nb::arg("x"), nb::arg("u"))
      .def(
        "gradient",
        [](const StageConstraints &self, const VecX &x, const VecX &u)
        {
            MatX dX, dU;
            self.gradient(x, u, dX, dU);
            return nb::make_tuple(dX, dU);
        },
        nb::arg("x"),
        nb::arg("u"))
      .def(
        "hessprod_xx",
        [](const StageConstraints &self,
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
        {
            MatX Pxx;
            self.hessprod_xx(x, u, mu, Pxx);
            return Pxx;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"))
      .def(
        "hessprod_xu",
        [](const StageConstraints &self,
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
        {
            MatX Pxu;
            self.hessprod_xu(x, u, mu, Pxu);
            return Pxu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"))
      .def(
        "hessprod_uu",
        [](const StageConstraints &self,
           const VecX             &x,
           const VecX             &u,
           const VecX             &mu)
        {
            MatX Puu;
            self.hessprod_uu(x, u, mu, Puu);
            return Puu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("mu"));

    nb::class_<FramePositionBound, StageConstraints>(m, "FramePositionBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const VecX &,
                    const VecX &,
                    uint,
                    uint>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"));

    nb::class_<FrameVelocityBound, StageConstraints>(m, "FrameVelocityBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const VecX &,
                    const VecX &,
                    uint,
                    uint,
                    bool>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("terminal") = false);
}
}
}
