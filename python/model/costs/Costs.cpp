#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include "model/Model.hpp"
#include "model/stagewise/StageCost.hpp"
#include "model/stagewise/predefined/Costs.hpp"

namespace nb = nanobind;
using namespace ssqp::model;
using namespace ssqp::model::stagewise;
using namespace ssqp::model::stagewise::predefined::costs;

NB_MAKE_OPAQUE(std::shared_ptr<StageCost>);

void bind_Costs(nb::module_ &m)
{
    nb::class_<SquaredJointAngleCost>(m, "SquaredJointAngleCost")
      .def(nb::init<uint, uint, double, bool>(),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false);

    nb::class_<SquaredJointVelocityCost>(m, "SquaredJointVelocityCost")
      .def(nb::init<uint, double, bool>(),
           nb::arg("nv"),
           nb::arg("weight"),
           nb::arg("free_flyer") = false);

    nb::class_<SquaredControlCost>(m, "SquaredControlCost")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"));

    nb::class_<SquaredForceCost>(m, "SquaredForceCost")
      .def(nb::init<uint, double>(), nb::arg("nv"), nb::arg("weight"));

    nb::class_<FrameSquaredPositionErrorCost>(m,
                                              "FrameSquaredPositionErrorCost")
      .def(nb::init<const Model &,
                    const Eigen::Vector3d &,
                    const std::string &,
                    uint,
                    uint,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_translation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"));

    nb::class_<FrameSquaredRotationErrorCost>(m,
                                              "FrameSquaredRotationErrorCost")
      .def(nb::init<const Model &,
                    const Eigen::Matrix3d &,
                    const std::string &,
                    uint,
                    uint,
                    double>(),
           nb::arg("model"),
           nb::arg("desired_rotation"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"));

    nb::class_<FrameSquaredPoseErrorCost, StageCost>(
      m,
      "FrameSquaredPoseErrorCost")
      .def(nb::init<const Model &,
                    const Eigen::Matrix4d &,
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
                    const Eigen::Vector3d &,
                    const Eigen::Matrix3d &,
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
           nb::arg("rotation_weight"));

    nb::class_<FrameSquaredVelocityCost>(m, "FrameSquaredVelocityCost")
      .def(nb::init<const Model &, const std::string &, uint, uint, double>(),
           nb::arg("model"),
           nb::arg("frame"),
           nb::arg("nq"),
           nb::arg("nv"),
           nb::arg("weight"));
}
