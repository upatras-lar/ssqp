#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageDynamics.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model;
using namespace model::stagewise;

struct __attribute__((visibility("default"))) PyStageDynamics : StageDynamics
{
    NB_TRAMPOLINE(StageDynamics, 13);

    uint output_size() const override
    {
        NB_OVERRIDE_PURE(output_size);
    }

    uint state_size() const override
    {
        NB_OVERRIDE_PURE(state_size);
    }

    uint control_size() const override
    {
        NB_OVERRIDE_PURE(control_size);
    }

    uint state_diff_size() const override
    {
        NB_OVERRIDE(state_diff_size);
    }

    VecX init_state() const override
    {
        NB_OVERRIDE(init_state);
    }

    VecX init_control() const override
    {
        NB_OVERRIDE(init_control);
    }

    VecX state_right_plus(const VecX &x, const VecX &dx) const override
    {
        NB_OVERRIDE(state_right_plus, x, dx);
    }

    VecX state_right_minus(const VecX &x, const VecX &y) const override
    {
        NB_OVERRIDE(state_right_minus, x, y);
    }

    VecX evaluate(const VecX &x, const VecX &u) const override
    {
        NB_OVERRIDE_PURE(evaluate, x, u);
    }

    void gradient(const VecX &x, const VecX &u, MatX &dX,
                  MatX &dU) const override
    {
        NB_OVERRIDE_PURE(gradient, x, u, dX, dU);
    }

    void hessprod_xx(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxx) const override
    {
        NB_OVERRIDE_PURE(hessprod_xx, x, u, lambda, Pxx);
    }

    void hessprod_xu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Pxu) const override
    {
        NB_OVERRIDE_PURE(hessprod_xu, x, u, lambda, Pxu);
    }

    void hessprod_uu(const VecX &x, const VecX &u, const VecX &lambda,
                     MatX &Puu) const override
    {
        NB_OVERRIDE_PURE(hessprod_uu, x, u, lambda, Puu);
    }
};

void bind_Dynamics(nb::module_ &m)
{
    nb::class_<StageDynamics, PyStageDynamics>(m, "StageDynamics")
      .def(nb::init())
      .def("output_size", &StageDynamics::output_size)
      .def("state_size", &StageDynamics::state_size)
      .def("control_size", &StageDynamics::control_size)
      .def("state_diff_size", &StageDynamics::state_diff_size)
      .def("init_state", &StageDynamics::init_state)
      .def("init_control", &StageDynamics::init_control)
      .def("state_right_plus",
           &StageDynamics::state_right_plus,
           nb::arg("x"),
           nb::arg("dx"))
      .def("state_right_minus",
           &StageDynamics::state_right_minus,
           nb::arg("x"),
           nb::arg("y"))
      .def("evaluate", &StageDynamics::evaluate, nb::arg("x"), nb::arg("u"))
      .def(
        "gradient",
        [](const StageDynamics &self, const VecX &x, const VecX &u)
        {
            MatX dX, dU;
            self.gradient(x, u, dX, dU);
            return nb::make_tuple(dX, dU);
        },
        nb::arg("x"),
        nb::arg("u"))
      .def(
        "hessprod_xx",
        [](const StageDynamics &self,
           const VecX          &x,
           const VecX          &u,
           const VecX          &lambda)
        {
            MatX Pxx;
            self.hessprod_xx(x, u, lambda, Pxx);
            return Pxx;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lambda"))
      .def(
        "hessprod_xu",
        [](const StageDynamics &self,
           const VecX          &x,
           const VecX          &u,
           const VecX          &lambda)
        {
            MatX Pxu;
            self.hessprod_xu(x, u, lambda, Pxu);
            return Pxu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lambda"))
      .def(
        "hessprod_uu",
        [](const StageDynamics &self,
           const VecX          &x,
           const VecX          &u,
           const VecX          &lambda)
        {
            MatX Puu;
            self.hessprod_uu(x, u, lambda, Puu);
            return Puu;
        },
        nb::arg("x"),
        nb::arg("u"),
        nb::arg("lambda"));
}
}
}
