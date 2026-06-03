#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>

#include "model/stagewise/StageDynamics.hpp"
#include "models/double_pendulum/DoublePendulum.hpp"

namespace ssqp
{
namespace python
{
namespace nb = nanobind;

using namespace model::stagewise;
using namespace model::custom;

void bind_Custom(nb::module_ &m)
{
    nb::class_<DoublePendulumDynamics, StageDynamics>(m,
                                                      "DoublePendulumDynamics",
                                                      R"doc(
        Custom stagewise dynamics class for the double pendulum system
        Defines the double pendulum dynamics relative to 6 parameters:
         1. Mass 1 (m1)
         2. Mass 2 (m2)
         3. Length 1 (l1)
         4. Length 2 (l2)
         5. Inertia 1 (I1)
         6. Inertia 2 (I2)    
                                                      )doc")
      .def(nb::init<const Vec6 &, const std::string &>(),
           nb::arg("params"),
           nb::arg("name") = "");
}
}
}
