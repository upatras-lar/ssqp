#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/eigen/dense.h>

#include "model/stagewise/StageDynamics.hpp"
#include "model/stagewise/StageCost.hpp"
#include "models/double_pendulum/DoublePendulumAmbient.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model::stagewise;
using namespace model::custom;

void bind_Custom(nb::module_ &m)
{
    nb::module_ double_pendulum = m.def_submodule("double_pendulum");

    nb::enum_<double_pendulum::ActuationMode>(double_pendulum,
                                              "ActuationMode",
                                              R"doc(
        Enum used to select the mode of actuation for the double pendulum ambient
        model

        Options:
          - FULLY_ACTUATED : both joints actuated (classic double pendulum)
          - PENDUBOT       : only the first joint actuated, second passive
          - ACROBOT        : only the second joint actuated, first passive
        )doc")
      .value("Full", double_pendulum::ActuationMode::Full)
      .value("Pendubot", double_pendulum::ActuationMode::Pendubot)
      .value("Acrobot", double_pendulum::ActuationMode::Acrobot)
      .export_values();

    nb::class_<double_pendulum::DoublePendulumAmbientDynamics, StageDynamics>(
      double_pendulum,
      "DoublePendulumAmbientDynamics",
      R"doc(
        Two-Link Pendulum Ambient Dynamics
        Represents the system's state in ambient space: (sin(a), cos(a))
        for each angle a.

        A single class spanning all three actuation patterns of a two-link
        pendulum, selected via ActuationMode at construction time. 
      )doc")
      .def(nb::init<const Eigen::Vector<double, 9> &,
                    const double_pendulum::ActuationMode,
                    const double,
                    const std::string &>(),
           nb::arg("params"),
           nb::arg("actuation_mode") = double_pendulum::ActuationMode::Full,
           nb::arg("dt")             = 0.01,
           nb::arg("name")           = "")
      .def(
        "computeRegressorAndTorque",
        [](double_pendulum::DoublePendulumAmbientDynamics *self,
           const VecX                                     &x,
           const Vec2                                     &ddq,
           const VecX                                     &u)
        {
            Eigen::Matrix<double, 2, 9> Y;
            Vec2                        tau;
            self->compute_regressor_and_torque(x, ddq, u, Y, tau);
            return std::pair<MatX, VecX>(Y, tau);
        },
        nb::arg("x"),
        nb::arg("ddq"),
        nb::arg("u"));

    nb::class_<double_pendulum::DoublePendulumAmbientTrackingCost, StageCost>(
      double_pendulum,
      "DoublePendulumAmbientTrackingCost")
      .def(nb::init<const Vec6 &, const Vec4 &>(),
           nb::arg("x_des"),
           nb::arg("q_weight"))
      .def("setTarget",
           &double_pendulum::DoublePendulumAmbientTrackingCost::set_target,
           nb::arg("x_des"));
}
}
}
