#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageConstraints.hpp"
#include "model/stagewise/predefined/Constraints.hpp"

namespace nb = nanobind;
using namespace ssqp::model;
using namespace ssqp::model::stagewise;
using namespace ssqp::model::stagewise::predefined::constraints;

NB_MAKE_OPAQUE(std::shared_ptr<StageConstraints>);

void bind_Constraints(nb::module_ &m)
{
    nb::class_<FramePositionBound>(m, "FramePositionBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Eigen::VectorXd &,
                    const Eigen::VectorXd &,
                    uint,
                    uint>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("lb"),
           nb::arg("ub"),
           nb::arg("nq"),
           nb::arg("nv"));

    nb::class_<FrameVelocityBound>(m, "FrameVelocityBound")
      .def(nb::init<const Model &,
                    const std::string &,
                    const Eigen::VectorXd &,
                    const Eigen::VectorXd &,
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

    nb::class_<ContactConstraint>(m, "ContactConstraint")
      .def(nb::init<const Model &, const std::string &, uint, uint, double>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("mu"));
}
